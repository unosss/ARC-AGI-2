"""Restart-safe, label-gated parent runner for Experiment 035."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical(value) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(canonical(row) + "\n" for row in rows))
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment") != 35:
        raise ValueError("config.experiment must be 35")
    if config.get("arms") != ["baseline", "global_refiner", "task_refiner"]:
        raise ValueError("Experiment 035 arms changed")
    if config["refiner"].get("num_stages") != 3:
        raise ValueError("the iterative refiner must have three stages")
    if config["calibration"].get("mode") != "representative_subset":
        raise ValueError("calibration.mode must be representative_subset")
    if config.get("primary_metric") != "paired_task_level_top2_exact":
        raise ValueError("primary metric changed")


def bounded(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    from arc_solver.qwen2d_runtime import serialize_chatml_bounded

    result = []
    for source in rows:
        _, _, dropped = serialize_chatml_bounded(
            source.get("train", []), source["query"], source["target"],
            max_length=maximum,
        )
        row = dict(source)
        row["train"] = source.get("train", [])[dropped:]
        row["dropped_oldest_demos"] = dropped
        result.append(row)
    return result


def prepare(
    config: dict[str, Any], repo: Path, challenges_path: Path, work: Path
) -> dict[str, Any]:
    """Materialize all label-blind partitions and deterministic corpora."""
    sys.path.insert(0, str(repo))
    from arc_solver.qwen2d_experiment import (
        INTERPRETATION_FAMILIES,
        PERCEPTION_FAMILIES,
        fixed_token_budget,
        make_arc_loo_rows,
        make_refiner_rows,
        make_synthetic_rows,
        make_task_manifest,
        refiner_corpus_audit,
        sha256_json,
    )
    from kaggle_runs.qwen_staged_adapters_runpod.run_experiment import exposed_task_ids

    challenges = read_json(challenges_path)
    if not isinstance(challenges, dict) or not challenges:
        raise ValueError("challenges must be a non-empty task mapping")
    if any("output" in item for task in challenges.values() for item in task["test"]):
        raise ValueError("challenge file contains test outputs")
    excluded, exclusion_sources = exposed_task_ids(repo)
    # Experiment 034 deliberately omitted its own preregistration while
    # deriving its panel, so carry that frozen exclusion list forward here.
    prior_preregistration = (
        repo / "experiments" / "034_qwen_staged_adapters"
        / "preregistered_manifest.json"
    )
    prior = read_json(prior_preregistration)
    excluded.update(prior["exclusion_derivation"]["excluded_ids"])
    exclusion_sources.append({
        "path": str(prior_preregistration.relative_to(repo)),
        "sha256": digest(prior_preregistration),
        "id_count": str(len(prior["exclusion_derivation"]["excluded_ids"])),
    })
    manifest = make_task_manifest(
        challenges,
        excluded,
        seed=config["split_seed"],
        reserve_count=config["panel"]["reserve_tasks"],
        primary_count=config["panel"]["primary_tasks"],
        negative_count=1,
        replay_count=0,
    )
    manifest["pilot_ids"] = manifest["primary_ids"][: config["panel"]["pilot_tasks"]]

    synthetic_train, synthetic_probe = make_synthetic_rows(
        seeds_per_family=config["corpus"]["synthetic_seeds_per_family"],
        seed=config["seed"],
        heldout_modulus=config["corpus"]["synthetic_holdout_modulus"],
    )
    arc_rows = make_arc_loo_rows(challenges, manifest["global_train_ids"])
    maximum = config["corpus"]["global_max_sequence_length"]
    staged_sources = {
        "staged_perception": [
            row for row in synthetic_train if row["family"] in PERCEPTION_FAMILIES
        ],
        "staged_interpretation": [
            row for row in synthetic_train if row["family"] in INTERPRETATION_FAMILIES
        ] + arc_rows,
    }
    artifact_hashes: dict[str, str] = {}
    corpus_root = work / "manifests" / "corpora"
    corpus_manifests = {}
    for name, source in staged_sources.items():
        stage = name.removeprefix("staged_")
        rows, row_manifest = fixed_token_budget(
            bounded(source, maximum), config["stage_budgets"][stage]
        )
        path = corpus_root / f"{name}.jsonl"
        write_jsonl(path, rows)
        artifact_hashes[str(path.relative_to(work))] = digest(path)
        corpus_manifests[name] = row_manifest

    # Refiner prompts contain train/query/draft only. Targets remain separate
    # teacher labels consumed by refiner_loss.
    refiner_source = bounded(synthetic_train + arc_rows, maximum)
    refiner_rows = make_refiner_rows(
        refiner_source,
        seed=config["seed"],
        variants_per_row=config["corpus"]["refiner_variants_per_row"],
    )[: config["corpus"]["global_refiner_rows"]]
    audit = refiner_corpus_audit(
        refiner_rows,
        global_task_ids=manifest["global_train_ids"],
        heldout_task_ids=manifest["reserved_ids"],
    )
    refiner_path = corpus_root / "global_refiner.jsonl"
    write_jsonl(refiner_path, refiner_rows)
    artifact_hashes[str(refiner_path.relative_to(work))] = digest(refiner_path)

    docs = {
        "task_manifest.json": manifest,
        "corpus_manifests.json": corpus_manifests,
        "refiner_audit.json": audit,
        "probe_rows.json": {
            "perception": [
                row for row in synthetic_probe if row["family"] in PERCEPTION_FAMILIES
            ],
            "interpretation": [
                row for row in synthetic_probe if row["family"] in INTERPRETATION_FAMILIES
            ],
        },
        "exclusion_sources.json": exclusion_sources,
    }
    for name, value in docs.items():
        path = work / "manifests" / name
        atomic_json(path, value)
        artifact_hashes[str(path.relative_to(work))] = digest(path)
    prepared = {
        "schema_version": 1,
        "experiment": 35,
        "created_without_solution_file": True,
        "selection_uses_test_outputs": False,
        "challenge_sha256": digest(challenges_path),
        "config_sha256": digest(HERE / "config.json"),
        "panel_sha256": sha256_json(manifest["primary_ids"]),
        "artifacts": artifact_hashes,
    }
    atomic_json(work / "manifests" / "prepared.json", prepared)
    return manifest


def verify_prepared(work: Path, challenges: Path) -> dict[str, Any]:
    prepared = read_json(work / "manifests" / "prepared.json")
    if prepared["challenge_sha256"] != digest(challenges):
        raise RuntimeError("challenge file changed after preparation")
    if prepared["config_sha256"] != digest(HERE / "config.json"):
        raise RuntimeError("config changed after preparation")
    for relative, expected in prepared["artifacts"].items():
        path = work / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"prepared artifact changed: {relative}")
    return read_json(work / "manifests" / "task_manifest.json")


def run_worker(
    mode: str, repo: Path, challenges: Path, model: Path, work: Path
) -> None:
    config = read_json(HERE / "config.json")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["ARC035_DEADLINE"] = str(time.time() + config["wall_cap_seconds"])
    command = [
        sys.executable, str(HERE / "worker.py"), "--mode", mode,
        "--config", str(HERE / "config.json"), "--challenges", str(challenges),
        "--model-dir", str(model), "--work", str(work),
    ]
    log_path = work / "logs" / f"{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"{mode} worker exited {code}; rerun to resume")


def calibration_gate(config: dict[str, Any], work: Path) -> dict[str, Any]:
    result = read_json(work / "calibration" / "result.json")
    margin = 1.0 + config["calibration"]["safety_margin"]
    gate = {
        **result,
        "safety_margin": config["calibration"]["safety_margin"],
        "projected_with_margin_seconds": result["projected_full_seconds"] * margin,
        "peak_with_margin_bytes": result["peak_allocated_bytes"] * margin,
    }
    gate["time_passed"] = (
        gate["projected_with_margin_seconds"]
        <= config["calibration"]["maximum_projected_seconds"]
    )
    gate["memory_passed"] = (
        gate["peak_with_margin_bytes"] <= result["total_gpu_memory_bytes"]
    )
    gate["passed"] = gate["time_passed"] and gate["memory_passed"]
    atomic_json(work / "calibration" / "gate.json", gate)
    return gate


def cell_accuracy(candidate: list[list[int]], truth: list[list[int]]) -> float:
    if len(candidate) != len(truth) or any(
        len(a) != len(b) for a, b in zip(candidate, truth)
    ):
        return 0.0
    total = sum(map(len, truth))
    return sum(
        x == y for a, b in zip(candidate, truth) for x, y in zip(a, b)
    ) / total


def evaluate(
    config: dict[str, Any],
    manifest: dict[str, Any],
    solutions_path: Path,
    work: Path,
    output: Path,
) -> None:
    """Open solutions only after every arm/task candidate has a valid hash."""
    from arc_solver.qwen2d_experiment import LabelLifecycle, paired_contrast

    lifecycle = LabelLifecycle(work, config["arms"], manifest["primary_ids"])
    for arm, task_id in itertools.product(config["arms"], manifest["primary_ids"]):
        path = work / "arms" / arm / "candidates" / f"{task_id}.json"
        marker = path.with_suffix(".sha256")
        if not path.is_file() or not marker.is_file():
            raise RuntimeError(f"missing candidate: {arm}/{task_id}")
        metadata = read_json(marker)
        if metadata["sha256"] != digest(path):
            raise RuntimeError(f"candidate digest mismatch: {arm}/{task_id}")
        lifecycle.record_candidates(arm, task_id, path)
    lifecycle.assert_ready_for_labels()
    solutions = read_json(solutions_path)
    selected = {task_id: solutions[task_id] for task_id in manifest["primary_ids"]}
    lifecycle.materialize_labels(selected)
    arms, metrics = {}, {}
    for arm in config["arms"]:
        tasks, metrics[arm] = [], {}
        for task_id in manifest["primary_ids"]:
            payload = read_json(work / "arms" / arm / "candidates" / f"{task_id}.json")
            if len(payload["ranked_candidates"]) != len(selected[task_id]):
                raise RuntimeError(f"test-item count mismatch: {arm}/{task_id}")
            items = []
            for candidates, truth in zip(payload["ranked_candidates"], selected[task_id]):
                grids = [candidate["grid"] for candidate in candidates]
                rank = grids.index(truth) + 1 if truth in grids else None
                items.append({
                    "top1_exact": rank == 1,
                    "top2_exact": rank is not None and rank <= 2,
                    "truth_rank": rank,
                    "top2_best_cell_accuracy": max(
                        (cell_accuracy(grid, truth) for grid in grids[:2]), default=0.0
                    ),
                })
            exact = all(item["top2_exact"] for item in items)
            metrics[arm][task_id] = float(exact)
            tasks.append({"task_id": task_id, "task_top2_exact": exact, "items": items})
        arms[arm] = {
            "tasks": tasks,
            "aggregate": {
                "tasks": len(tasks),
                "task_top2_exact": sum(item["task_top2_exact"] for item in tasks),
            },
        }
    report = {
        "schema_version": 1,
        "experiment": 35,
        "labels_opened_after_all_candidate_hashes": True,
        "solution_sha256": digest(solutions_path),
        "primary_metric": config["primary_metric"],
        "arms": arms,
        "pairwise": [
            paired_contrast(metrics, "baseline", arm)
            for arm in ("global_refiner", "task_refiner")
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)


def grade_exploratory_subset(
    config: dict[str, Any],
    task_ids: list[str],
    solutions_path: Path,
    work: Path,
    output: Path,
) -> dict[str, Any]:
    """Grade a preregistered pilot after all subset candidates are immutable."""
    solutions = read_json(solutions_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": 35,
        "exploratory": True,
        "task_ids": task_ids,
        "arms": {},
    }
    for arm in config["arms"]:
        task_reports = []
        for task_id in task_ids:
            path = work / "arms" / arm / "candidates" / f"{task_id}.json"
            marker = path.with_suffix(".sha256")
            if not path.is_file() or not marker.is_file():
                raise RuntimeError(f"missing pilot candidate: {arm}/{task_id}")
            metadata = read_json(marker)
            if metadata.get("sha256") != digest(path):
                raise RuntimeError(f"pilot candidate digest mismatch: {arm}/{task_id}")
            payload = read_json(path)
            truths = solutions[task_id]
            items = []
            for item_index, truth in enumerate(truths):
                candidates = payload["ranked_candidates"][item_index]
                top2 = [candidate["grid"] for candidate in candidates[:2]]
                trajectories = payload["stage_trajectories"][item_index]
                stage_metrics = {}
                monotonic = []
                for trajectory in trajectories:
                    accuracies = []
                    for stage_index in range(4):
                        key = f"G{stage_index}"
                        if key not in trajectory:
                            continue
                        grid = trajectory[key]
                        accuracy = cell_accuracy(grid, truth)
                        accuracies.append(accuracy)
                        stage_metrics.setdefault(key, []).append(
                            {
                                "exact": grid == truth,
                                "cell_accuracy": accuracy,
                                "shape_exact": (
                                    len(grid) == len(truth)
                                    and len(grid[0]) == len(truth[0])
                                ),
                            }
                        )
                    monotonic.append(
                        all(
                            right + 1e-12 >= left
                            for left, right in zip(accuracies, accuracies[1:])
                        )
                    )
                items.append(
                    {
                        "top2_exact": truth in top2,
                        "top2_best_cell_accuracy": max(
                            (cell_accuracy(grid, truth) for grid in top2),
                            default=0.0,
                        ),
                        "stage_best": {
                            stage: {
                                "exact": any(value["exact"] for value in values),
                                "cell_accuracy": max(
                                    value["cell_accuracy"] for value in values
                                ),
                                "shape_exact": any(
                                    value["shape_exact"] for value in values
                                ),
                            }
                            for stage, values in stage_metrics.items()
                        },
                        "trajectory_monotonic": monotonic,
                    }
                )
            task_reports.append(
                {
                    "task_id": task_id,
                    "task_top2_exact": all(item["top2_exact"] for item in items),
                    "items": items,
                }
            )
        report["arms"][arm] = {
            "tasks": task_reports,
            "task_top2_exact": sum(
                task["task_top2_exact"] for task in task_reports
            ),
            "mean_top2_best_cell_accuracy": sum(
                item["top2_best_cell_accuracy"]
                for task in task_reports
                for item in task["items"]
            )
            / sum(len(task["items"]) for task in task_reports),
        }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "pilot_report.json", report)
    return report


def smoke(repo: Path) -> None:
    """Exercise the CPU path without loading or downloading a Qwen model."""
    with tempfile.TemporaryDirectory(prefix="arc035-smoke-") as directory:
        command = [
            sys.executable, str(HERE / "worker.py"), "--smoke",
            "--config", str(HERE / "config.json"), "--work", directory,
        ]
        env = {**os.environ, "PYTHONPATH": str(repo)}
        subprocess.run(command, check=True, env=env)
        artifact = Path(directory) / "calibration" / "smoke.json"
        if not read_json(artifact).get("passed"):
            raise RuntimeError("offline smoke artifact did not pass")
        print(canonical({"smoke": "passed", "artifact": str(artifact)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--challenges", type=Path)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--calibrate-only", action="store_true")
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--grade-pilot", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config = read_json(HERE / "config.json")
    validate_config(config)
    repo = args.repo.resolve()
    if args.smoke:
        smoke(repo)
        return
    if args.challenges is None or args.work is None:
        parser.error("--challenges and --work are required outside smoke mode")
    challenges, work = args.challenges.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / "manifests" / "prepared.json"
    manifest = (
        verify_prepared(work, challenges)
        if prepared.is_file()
        else prepare(config, repo, challenges, work)
    )
    if args.prepare_only:
        print(canonical({"prepared": True, "primary_ids": manifest["primary_ids"]}))
        return
    if args.grade_pilot:
        if args.solutions is None or args.output is None:
            parser.error("--grade-pilot requires --solutions and --output")
        report = grade_exploratory_subset(
            config,
            manifest["pilot_ids"],
            args.solutions.resolve(),
            work,
            args.output.resolve(),
        )
        print(canonical(report))
        return
    if args.model is None or not args.model.joinpath("config.json").is_file():
        raise ValueError("--model must be a local Transformers checkpoint")
    if args.calibrate_only:
        run_worker("calibrate", repo, challenges, args.model.resolve(), work)
        print(canonical(calibration_gate(config, work)))
        return
    gate_path = work / "calibration" / "gate.json"
    if not gate_path.is_file() or not read_json(gate_path).get("passed"):
        raise RuntimeError("a passing --calibrate-only run is required")
    mode = "pilot" if args.pilot else "full"
    run_worker(mode, repo, challenges, args.model.resolve(), work)
    if args.full:
        if args.solutions is None or args.output is None:
            parser.error("--full requires --solutions and --output")
        evaluate(config, manifest, args.solutions.resolve(), work, args.output.resolve())


if __name__ == "__main__":
    main()
