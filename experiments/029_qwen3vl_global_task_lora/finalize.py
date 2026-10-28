"""Unseal labels and score Experiment 029 only after every candidate exists."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from core import cell_accuracy, file_sha256

HERE = Path(__file__).resolve().parent


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def load_sealed_candidates(
    config_path: Path, challenge_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = json.loads(config_path.read_text())
    config_hash = file_sha256(config_path)
    challenge_hash = file_sha256(challenge_path)
    if challenge_hash != config["challenge_sha256"]:
        raise ValueError("challenge hash differs from fixed config")
    records: dict[str, dict[str, Any]] = {}
    for arm in config["arms"]:
        records[arm] = {}
        for task_id in config["task_ids"]:
            run_dir = output_root / arm / task_id
            candidate_path = run_dir / "candidate.json"
            manifest_path = run_dir / "manifest.json"
            if not candidate_path.exists() or not manifest_path.exists():
                raise RuntimeError(
                    "official evaluation is sealed until all 32 candidate "
                    f"artifacts exist; missing {arm}/{task_id}"
                )
            manifest = json.loads(manifest_path.read_text())
            if file_sha256(candidate_path) != manifest.get("candidate_sha256"):
                raise ValueError(f"candidate hash mismatch: {arm}/{task_id}")
            adapter_hashes = manifest.get("task_adapter_sha256", {})
            if {Path(name).name for name in adapter_hashes} != {
                "adapter_config.json",
                "adapter_model.safetensors",
            }:
                raise ValueError(f"adapter hash set is incomplete: {arm}/{task_id}")
            for name, expected in adapter_hashes.items():
                if file_sha256(run_dir / "adapter" / name) != expected:
                    raise ValueError(f"adapter hash mismatch: {arm}/{task_id}/{name}")
            candidate = json.loads(candidate_path.read_text())
            fingerprint = candidate.get("fingerprint", {})
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError(f"manifest fingerprint mismatch: {arm}/{task_id}")
            expected_fields = {
                "experiment": 29,
                "arm": arm,
                "task_id": task_id,
                "config_sha256": config_hash,
                "challenge_sha256": challenge_hash,
                "worker_sha256": file_sha256(HERE / "worker.py"),
                "core_sha256": file_sha256(HERE / "core.py"),
            }
            for key, expected in expected_fields.items():
                if fingerprint.get(key) != expected:
                    raise ValueError(
                        f"candidate fingerprint mismatch: {arm}/{task_id}/{key}"
                    )
            if fingerprint.get("model") is None or fingerprint.get("runtime") is None:
                raise ValueError(f"incomplete runtime identity: {arm}/{task_id}")
            if arm == "perception_task_lora":
                if fingerprint.get("global_adapter_sha256") != {
                    "adapter_config.json": config[
                        "global_adapter_config_sha256"
                    ],
                    "adapter_model.safetensors": config[
                        "global_adapter_model_sha256"
                    ],
                }:
                    raise ValueError(
                        f"global adapter identity mismatch: {arm}/{task_id}"
                    )
            if candidate["optimizer_steps"] != config["optimizer_steps"]:
                raise ValueError(f"incomplete optimization: {arm}/{task_id}")
            records[arm][task_id] = candidate
    for task_id in config["task_ids"]:
        initial = {
            records[arm][task_id]["initial_task_adapter_digest"]
            for arm in config["arms"]
        }
        if len(initial) != 1:
            raise ValueError(f"task adapter initialization differs for {task_id}")
    for identity_key in ("model", "runtime"):
        identities = {
            json.dumps(
                records[arm][task_id]["fingerprint"][identity_key],
                sort_keys=True,
            )
            for arm in config["arms"]
            for task_id in config["task_ids"]
        }
        if len(identities) != 1:
            raise ValueError(f"mixed {identity_key} identities across candidates")
    return config, records


def score_arm(
    config: dict[str, Any],
    records: dict[str, Any],
    solutions: dict[str, list[list[list[int]]]],
) -> dict[str, Any]:
    task_rows = []
    total_items = 0
    exact_items = 0
    valid_items = 0
    valid_beams = 0
    total_beams = 0
    cell_sum = 0.0
    for task_id in config["task_ids"]:
        truths = solutions[task_id]
        predictions = records[task_id]["test_predictions"]
        if len(predictions) != len(truths):
            raise ValueError(f"test item count mismatch: {task_id}")
        item_rows = []
        for prediction, truth in zip(predictions, truths):
            candidates = prediction["candidates"]
            item_exact = any(candidate == truth for candidate in candidates)
            best_cell = max(
                (cell_accuracy(candidate, truth) for candidate in candidates),
                default=0.0,
            )
            raw = prediction["raw_generations"]
            beam_valid = sum(bool(row["parsed_grids"]) for row in raw)
            exact_items += int(item_exact)
            valid_items += int(bool(candidates))
            valid_beams += beam_valid
            total_beams += len(raw)
            cell_sum += best_cell
            total_items += 1
            item_rows.append(
                {
                    "index": prediction["index"],
                    "top2_exact": item_exact,
                    "best_cell_accuracy": best_cell,
                    "candidate_count": len(candidates),
                    "valid_beams": beam_valid,
                }
            )
        task_exact = all(row["top2_exact"] for row in item_rows)
        task_rows.append(
            {
                "task_id": task_id,
                "top2_exact": task_exact,
                "items": item_rows,
                "train_reconstruction_rate": sum(
                    row["top2_exact"]
                    for row in records[task_id]["train_reconstructions"]
                )
                / len(records[task_id]["train_reconstructions"]),
                "train_seconds": records[task_id]["train_seconds"],
                "peak_allocated_bytes": records[task_id]["peak_allocated_bytes"],
            }
        )
    return {
        "task_top2_exact": sum(row["top2_exact"] for row in task_rows),
        "task_count": len(task_rows),
        "item_top2_exact": exact_items,
        "item_count": total_items,
        "mean_best_cell_accuracy": cell_sum / total_items,
        "valid_item_rate": valid_items / total_items,
        "valid_beam_rate": valid_beams / total_beams,
        "tasks": task_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    # This operation must finish before the solution path is touched.
    config, records = load_sealed_candidates(
        args.config, args.challenge, args.output_root
    )
    if file_sha256(args.solutions) != config["solution_sha256"]:
        raise ValueError("solution file differs from fixed SHA-256")
    solutions = json.loads(args.solutions.read_text())
    if not set(config["task_ids"]).issubset(solutions):
        raise ValueError("solution file is missing a task from the fixed hard set")

    arms = {
        arm: score_arm(config, records[arm], solutions) for arm in config["arms"]
    }
    base_tasks = {
        row["task_id"]: row["top2_exact"]
        for row in arms["base_task_lora"]["tasks"]
    }
    treatment_tasks = {
        row["task_id"]: row["top2_exact"]
        for row in arms["perception_task_lora"]["tasks"]
    }
    wins = [
        task_id
        for task_id in config["task_ids"]
        if treatment_tasks[task_id] and not base_tasks[task_id]
    ]
    losses = [
        task_id
        for task_id in config["task_ids"]
        if base_tasks[task_id] and not treatment_tasks[task_id]
    ]
    report = {
        "experiment": 29,
        "config_sha256": file_sha256(args.config),
        "challenge_sha256": file_sha256(args.challenge),
        "solution_sha256": file_sha256(args.solutions),
        "arms": arms,
        "paired": {
            "perception_wins": wins,
            "perception_losses": losses,
            "ties": len(config["task_ids"]) - len(wins) - len(losses),
            "net_task_gain": len(wins) - len(losses),
        },
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
