"""Restart-safe parent lifecycle for Experiment 034.

The parent alone knows the solution path.  The worker receives a label-free
challenge file and cannot evaluate or select candidates against test answers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
PERCEPTION_FAMILIES = frozenset(
    {
        "reconstruction",
        "indexed_cell",
        "row_extraction",
        "column_extraction",
        "component_count",
        "component_bbox",
        "component_membership",
    }
)
INTERPRETATION_FAMILIES = frozenset(
    {"transpose", "d4", "translation", "copy_component", "recolor_component"}
)
HARD_EXPLORATORY_IDS = [
    "195c6913",
    "221dfab4",
    "20a9e565",
    "13e47133",
    "142ca369",
    "269e22fb",
    "16b78196",
    "21897d95",
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical(value) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "experiment": 34,
        "arms": ["staged_late", "mixed_late", "staged_early_collision"],
        "stage_budgets": {"perception": 600_000, "interpretation": 900_000},
        "primary_metric": "paired_task_level_top2_exact",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"config.{key} must be {value!r}")
    if config["runtime"].get("peft") != "0.21.2":
        raise ValueError("Experiment 034 requires PEFT 0.21.2")
    if config["runtime"].get("transformers") != ">=4.57,<5":
        raise ValueError("Transformers constraint changed")
    if config["runtime"].get("minimum_gpu_memory_bytes") != 24_000_000_000:
        raise ValueError("minimum GPU memory must be exactly 24 GB")
    if config["panel"].get("primary_tasks") != 32:
        raise ValueError("confirmatory panel must contain 32 tasks")
    if config["panel"].get("reserve_tasks") != 100:
        raise ValueError("reserve count must be 100")
    if config.get("control_arms") != [
        "reversed_bands",
        "shuffled_interpretation",
    ]:
        raise ValueError("diagnostic control arms changed")
    if config.get("structure_learning_rate") != 0.001:
        raise ValueError("2D structure-adapter learning rate changed")
    global_lora = config["global_lora"]
    task_lora = config["task_local"]
    if (global_lora["rank"], global_lora["alpha"]) != (64, 16):
        raise ValueError("global stages must use rank 64 / alpha 16")
    if (task_lora["rank"], task_lora["alpha"], task_lora["steps"]) != (256, 32, 128):
        raise ValueError("task stage must use rank 256 / alpha 32 / 128 steps")
    required_dfs = {
        "batch_size": 4,
        "max_nodes_per_view": 4096,
        "max_model_forward_tokens_per_view": 12288,
        "max_candidates_per_view": 128,
        "max_new_tokens": 931,
        "emergency_seconds_per_view": 900,
        "min_prob": 0.2,
    }
    if config.get("dfs") != required_dfs:
        raise ValueError("production DFS budgets changed")
    if config["calibration"].get("safety_margin") != 0.25:
        raise ValueError("calibration safety margin must be 25%")


def _ids_from_value(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"primary_ids", "task_ids", "pilot_task_ids"} and isinstance(
                nested, list
            ):
                found.update(
                    item
                    for item in nested
                    if isinstance(item, str) and len(item) == 8
                )
            else:
                found.update(_ids_from_value(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_ids_from_value(nested))
    return found


def exposed_task_ids(repo: Path) -> tuple[set[str], list[dict[str, str]]]:
    """Derive exclusions from preregistrations/configs, never solution files."""
    records: list[dict[str, str]] = []
    excluded: set[str] = set()
    experiment_root = repo / "experiments"
    patterns = ("preregistered_manifest.json", "config.json")
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in experiment_root.glob(f"*/{pattern}")
            if "034_qwen_staged_adapters" not in str(path)
        }
    )
    for path in paths:
        try:
            ids = _ids_from_value(read_json(path))
        except (json.JSONDecodeError, OSError):
            continue
        if ids:
            excluded.update(ids)
            records.append(
                {
                    "path": str(path.relative_to(repo)),
                    "sha256": digest_file(path),
                    "id_count": str(len(ids)),
                }
            )
    # Manually reviewed blind traces are exposed even though they are not a manifest.
    trace_root = experiment_root / "030_frontier_structured_reasoning" / "traces"
    if trace_root.is_dir():
        trace_ids = {path.stem for path in trace_root.glob("*.json")}
        excluded.update(trace_ids)
        records.append(
            {
                "path": str(trace_root.relative_to(repo)),
                "sha256": digest_bytes(canonical(sorted(trace_ids)).encode()),
                "id_count": str(len(trace_ids)),
            }
        )
    return excluded, records


def _bounded_rows(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    from arc_solver.qwen2d_runtime import serialize_chatml_bounded

    output = []
    for source in rows:
        _, _, dropped = serialize_chatml_bounded(
            source.get("train", []),
            source["query"],
            source["target"],
            max_length=maximum,
        )
        row = dict(source)
        row["train"] = source.get("train", [])[dropped:]
        row["dropped_oldest_demos"] = dropped
        output.append(row)
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(canonical(row) + "\n" for row in rows))
    os.replace(temporary, path)


def prepare(
    config: dict[str, Any], repo: Path, challenges_path: Path, work: Path
) -> dict[str, Any]:
    """Create the panel and all corpora using challenge/train data only."""
    sys.path.insert(0, str(repo))
    from arc_solver.qwen2d_experiment import (
        SYNTHETIC_FAMILIES,
        fixed_token_budget,
        make_arc_loo_rows,
        make_synthetic_rows,
        make_task_manifest,
        sha256_json,
    )

    challenges = read_json(challenges_path)
    if not isinstance(challenges, dict) or not challenges:
        raise ValueError("challenge file must be a non-empty task mapping")
    if (PERCEPTION_FAMILIES | INTERPRETATION_FAMILIES) != set(SYNTHETIC_FAMILIES):
        raise AssertionError("every synthetic family must be assigned exactly once")
    if PERCEPTION_FAMILIES & INTERPRETATION_FAMILIES:
        raise AssertionError("synthetic stage families overlap")

    exclusions, exclusion_sources = exposed_task_ids(repo)
    preregistration = read_json(
        repo
        / "experiments"
        / "034_qwen_staged_adapters"
        / "preregistered_manifest.json"
    )
    fixed_exclusions = set(
        preregistration["exclusion_derivation"]["excluded_ids"]
    )
    exclusions.update(fixed_exclusions)
    pilot_ids = sorted(fixed_exclusions & set(challenges))[:2]
    if len(pilot_ids) != 2:
        raise RuntimeError(
            "two previously exposed challenge tasks are required for the pilot"
        )
    manifest = make_task_manifest(
        challenges,
        exclusions,
        seed=config["split_seed"],
        reserve_count=config["panel"]["reserve_tasks"],
        primary_count=config["panel"]["primary_tasks"],
        negative_count=8,
        replay_count=0,
    )
    if len(manifest["primary_ids"]) != 32:
        raise AssertionError("panel selection did not produce 32 tasks")
    controls = sorted(
        manifest["primary_ids"],
        key=lambda task_id: (
            digest_bytes(f"experiment-034-control:{task_id}".encode()),
            task_id,
        ),
    )[:8]
    manifest["control_ids"] = controls
    manifest["negative_control_ids"] = controls
    manifest["pilot_ids"] = pilot_ids
    panel = set(manifest["primary_ids"])
    if panel & exclusions:
        raise AssertionError("new panel overlaps a previously exposed task")
    if panel & set(manifest["global_train_ids"]):
        raise AssertionError("panel leaked into global ARC training")

    synthetic, probes = make_synthetic_rows(
        seeds_per_family=config["corpus"]["synthetic_seeds_per_family"],
        seed=config["seed"],
        heldout_modulus=config["corpus"]["synthetic_holdout_modulus"],
    )
    perception = [row for row in synthetic if row["family"] in PERCEPTION_FAMILIES]
    interpretation_synthetic = [
        row for row in synthetic if row["family"] in INTERPRETATION_FAMILIES
    ]
    arc_rows = make_arc_loo_rows(challenges, manifest["global_train_ids"])
    interpretation = interpretation_synthetic + arc_rows
    shuffled_order = sorted(
        range(len(interpretation)),
        key=lambda index: (
            digest_bytes(
                f"experiment-034-shuffle:{interpretation[index]['row_id']}".encode()
            ),
            interpretation[index]["row_id"],
        ),
    )
    shuffled_targets = [
        interpretation[shuffled_order[(offset + 1) % len(shuffled_order)]]["target"]
        for offset in range(len(shuffled_order))
    ]
    shuffled_interpretation = []
    for offset, source_index in enumerate(shuffled_order):
        row = dict(interpretation[source_index])
        row["row_id"] = f"shuffled_interpretation:{row['row_id']}"
        row["target"] = shuffled_targets[offset]
        row["negative_control"] = "deterministically_rotated_target"
        shuffled_interpretation.append(row)
    # A deterministic interleave makes both mixed stages see the same balanced
    # objective ordering before exact fixed-token truncation.
    if not perception or not interpretation:
        raise AssertionError("both stage objectives require non-empty corpora")
    mixed = []
    for index in range(max(len(perception), len(interpretation))):
        for objective, pool in (
            ("perception", perception),
            ("interpretation", interpretation),
        ):
            row = dict(pool[index % len(pool)])
            row["row_id"] = f"mixed:{objective}:{index:06d}:{row['row_id']}"
            row["mixed_objective"] = objective
            mixed.append(row)

    maximum = config["corpus"]["global_max_sequence_length"]
    sources = {
        "staged_perception": _bounded_rows(perception, maximum),
        "staged_interpretation": _bounded_rows(interpretation, maximum),
        "mixed_perception": _bounded_rows(mixed, maximum),
        "mixed_interpretation": _bounded_rows(mixed, maximum),
        "shuffled_interpretation": _bounded_rows(
            shuffled_interpretation, maximum
        ),
    }
    budgets = {
        "staged_perception": config["stage_budgets"]["perception"],
        "staged_interpretation": config["stage_budgets"]["interpretation"],
        "mixed_perception": config["stage_budgets"]["perception"],
        "mixed_interpretation": config["stage_budgets"]["interpretation"],
        "shuffled_interpretation": config["stage_budgets"]["interpretation"],
    }
    manifest_root = work / "manifests"
    corpus_root = manifest_root / "corpora"
    corpus_manifests = {}
    artifact_hashes = {}
    for name, rows in sources.items():
        corpus, corpus_manifest = fixed_token_budget(rows, budgets[name])
        path = corpus_root / f"{name}.jsonl"
        _write_jsonl(path, corpus)
        corpus_manifests[name] = corpus_manifest
        artifact_hashes[str(path.relative_to(work))] = digest_file(path)

    probe_partitions = {
        "perception": [row for row in probes if row["family"] in PERCEPTION_FAMILIES],
        "interpretation": [
            row for row in probes if row["family"] in INTERPRETATION_FAMILIES
        ],
    }
    documents = {
        "task_manifest.json": manifest,
        "corpus_manifests.json": corpus_manifests,
        "probe_rows.json": probe_partitions,
        "exclusion_sources.json": exclusion_sources,
    }
    for name, value in documents.items():
        path = manifest_root / name
        atomic_json(path, value)
        artifact_hashes[str(path.relative_to(work))] = digest_file(path)
    prepared = {
        "schema_version": 1,
        "experiment": 34,
        "created_without_solution_file": True,
        "challenge_sha256": digest_file(challenges_path),
        "config_sha256": digest_file(HERE / "config.json"),
        "panel_sha256": sha256_json(manifest["primary_ids"]),
        "exclusions_sha256": sha256_json(sorted(exclusions)),
        "artifacts": artifact_hashes,
    }
    atomic_json(manifest_root / "prepared.json", prepared)
    return manifest


def verify_prepared(work: Path, challenges_path: Path) -> dict[str, Any]:
    prepared = read_json(work / "manifests" / "prepared.json")
    if prepared["challenge_sha256"] != digest_file(challenges_path):
        raise RuntimeError("challenge file changed since preparation")
    if prepared["config_sha256"] != digest_file(HERE / "config.json"):
        raise RuntimeError("configuration changed since preparation")
    for relative, expected in prepared["artifacts"].items():
        path = work / relative
        if not path.is_file() or digest_file(path) != expected:
            raise RuntimeError(f"prepared artifact mismatch: {relative}")
    return read_json(work / "manifests" / "task_manifest.json")


def require_24gb_gpu(config: dict[str, Any]) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.check_output(command, text=True).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("a visible NVIDIA GPU is required") from error
    if not values:
        raise RuntimeError("nvidia-smi reported no GPU")
    available = max(int(value.strip()) * 1024**2 for value in values)
    required = int(config["runtime"]["minimum_gpu_memory_bytes"])
    if available < required:
        raise RuntimeError(
            f"GPU has {available / 1024**3:.1f} GiB; at least 24 GiB is required"
        )


def require_runtime_versions() -> None:
    try:
        peft = importlib.metadata.version("peft")
        transformers = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("run setup.sh before GPU execution") from error
    if peft != "0.21.2":
        raise RuntimeError(f"PEFT 0.21.2 is required, found {peft}")
    pieces = transformers.split(".")
    if len(pieces) < 2 or int(pieces[0]) != 4 or int(pieces[1]) < 57:
        raise RuntimeError(
            f"Transformers >=4.57,<5 is required, found {transformers}"
        )


def run_worker(
    mode: str,
    config: dict[str, Any],
    repo: Path,
    challenges: Path,
    model: Path,
    work: Path,
) -> None:
    log_path = work / "logs" / f"{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["ARC034_DEADLINE"] = str(time.time() + int(config["wall_cap_seconds"]))
    command = [
        sys.executable,
        str(HERE / "worker.py"),
        "--mode",
        mode,
        "--config",
        str(HERE / "config.json"),
        "--challenges",
        str(challenges),
        "--model-dir",
        str(model),
        "--work",
        str(work),
    ]
    with log_path.open("a") as log:
        log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
        log.flush()
        code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"{mode} worker exited {code}; rerunning is restart-safe")


def calibration_gate(config: dict[str, Any], work: Path) -> dict[str, Any]:
    result = read_json(work / "calibration" / "result.json")
    margin = 1.0 + config["calibration"]["safety_margin"]
    projected = float(result["projected_full_seconds"]) * margin
    peak = float(result.get("peak_allocated_bytes", 0))
    total_memory = float(result.get("total_gpu_memory_bytes", float("inf")))
    memory_with_margin = peak * margin
    gate = {
        **result,
        "safety_margin": config["calibration"]["safety_margin"],
        "projected_with_margin_seconds": projected,
        "peak_with_margin_bytes": memory_with_margin,
        "maximum_projected_seconds": config["calibration"][
            "maximum_projected_seconds"
        ],
        "time_passed": projected
        <= config["calibration"]["maximum_projected_seconds"],
        "memory_passed": memory_with_margin <= total_memory,
    }
    gate["passed"] = gate["time_passed"] and gate["memory_passed"]
    atomic_json(work / "calibration" / "gate.json", gate)
    return gate


def hash_all_candidates(
    config: dict[str, Any], manifest: dict[str, Any], work: Path
) -> None:
    from arc_solver.qwen2d_experiment import LabelLifecycle

    lifecycle = LabelLifecycle(work, config["arms"], manifest["primary_ids"])
    for arm, task_id in itertools.product(config["arms"], manifest["primary_ids"]):
        path = work / "arms" / arm / "candidates" / f"{task_id}.json"
        if not path.is_file():
            raise RuntimeError(f"candidate missing: {arm}/{task_id}")
        lifecycle.record_candidates(arm, task_id, path)
    lifecycle.assert_ready_for_labels()
    control_hashes = {}
    for arm, task_id in itertools.product(
        config["control_arms"], manifest["control_ids"]
    ):
        path = work / "arms" / arm / "candidates" / f"{task_id}.json"
        marker = path.with_suffix(".sha256")
        if not path.is_file() or not marker.is_file():
            raise RuntimeError(f"control candidate missing: {arm}/{task_id}")
        metadata = read_json(marker)
        actual = digest_file(path)
        if metadata.get("sha256") != actual:
            raise RuntimeError(f"control candidate digest mismatch: {arm}/{task_id}")
        control_hashes[f"{arm}/{task_id}"] = actual
    atomic_json(work / "labels" / "control_candidate_hashes.json", control_hashes)
    probe_hashes = {}
    for arm in [*config["arms"], *config["control_arms"]]:
        path = work / "arms" / arm / "global_probe_metrics.json"
        if not path.is_file():
            raise RuntimeError(f"probe metrics missing: {arm}")
        probe_hashes[arm] = digest_file(path)
    atomic_json(work / "labels" / "probe_hashes.json", probe_hashes)


def _cell_accuracy(candidate: list[list[int]], truth: list[list[int]]) -> float:
    if (
        len(candidate) != len(truth)
        or any(len(left) != len(right) for left, right in zip(candidate, truth))
    ):
        return 0.0
    total = sum(len(row) for row in truth)
    return (
        sum(
            left == right
            for candidate_row, truth_row in zip(candidate, truth)
            for left, right in zip(candidate_row, truth_row)
        )
        / total
        if total
        else 1.0
    )


def _score(candidates: list[dict[str, Any]], truth: list[list[int]]) -> dict[str, Any]:
    grids = [candidate["grid"] for candidate in candidates]
    rank = grids.index(truth) + 1 if truth in grids else None
    return {
        "top1_exact": rank == 1,
        "top2_exact": rank is not None and rank <= 2,
        "top2_best_cell_accuracy": max(
            (_cell_accuracy(grid, truth) for grid in grids[:2]),
            default=0.0,
        ),
        "truth_in_pool": rank is not None,
        "truth_rank": rank,
        "generation_miss": rank is None,
        "selection_miss": rank is not None and rank > 2,
    }


def evaluate(
    config: dict[str, Any],
    manifest: dict[str, Any],
    solutions_path: Path,
    work: Path,
    output: Path,
) -> None:
    """Open labels only after the complete candidate-hash barrier."""
    from arc_solver.qwen2d_experiment import LabelLifecycle, paired_contrast

    hash_all_candidates(config, manifest, work)
    solutions = read_json(solutions_path)
    selected = {task_id: solutions[task_id] for task_id in manifest["primary_ids"]}
    LabelLifecycle(work, config["arms"], manifest["primary_ids"]).materialize_labels(
        selected
    )
    arms: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}
    for arm in config["arms"]:
        tasks = []
        metrics[arm] = {}
        for task_id in manifest["primary_ids"]:
            payload = read_json(
                work / "arms" / arm / "candidates" / f"{task_id}.json"
            )
            truths = selected[task_id]
            if len(payload["ranked_candidates"]) != len(truths):
                raise RuntimeError(f"test-item count mismatch: {arm}/{task_id}")
            items = [
                _score(candidates, truth)
                for candidates, truth in zip(payload["ranked_candidates"], truths)
            ]
            exact = all(item["top2_exact"] for item in items)
            metrics[arm][task_id] = float(exact)
            tasks.append(
                {
                    "task_id": task_id,
                    "task_top1_exact": all(item["top1_exact"] for item in items),
                    "task_top2_exact": exact,
                    "items": items,
                }
            )
        arms[arm] = {
            "tasks": tasks,
            "aggregate": {
                "tasks": len(tasks),
                "task_top2_exact": sum(task["task_top2_exact"] for task in tasks),
                "task_top1_exact": sum(task["task_top1_exact"] for task in tasks),
                "item_top2_exact": sum(
                    item["top2_exact"] for task in tasks for item in task["items"]
                ),
                "mean_top2_best_cell_accuracy": sum(
                    item["top2_best_cell_accuracy"]
                    for task in tasks
                    for item in task["items"]
                )
                / sum(len(task["items"]) for task in tasks),
                "truth_in_pool": sum(
                    item["truth_in_pool"] for task in tasks for item in task["items"]
                ),
            },
        }
    pairwise = [
        paired_contrast(metrics, control, "staged_late")
        for control in ("mixed_late", "staged_early_collision")
    ]
    ordered = sorted(
        range(len(pairwise)),
        key=lambda index: pairwise[index]["paired_randomization_p"],
    )
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(
            1.0,
            (len(pairwise) - rank)
            * pairwise[index]["paired_randomization_p"],
        )
        running = max(running, adjusted)
        pairwise[index]["holm_adjusted_p"] = running
        low, high = pairwise[index]["bootstrap_95_ci"]
        pairwise[index]["supports_claim"] = (
            pairwise[index]["mean_difference"] > 0
            and running <= 0.05
            and low > 0
        )
    controls = {}
    for arm in config["control_arms"]:
        tasks = []
        for task_id in manifest["control_ids"]:
            payload = read_json(
                work / "arms" / arm / "candidates" / f"{task_id}.json"
            )
            items = [
                _score(candidates, truth)
                for candidates, truth in zip(
                    payload["ranked_candidates"], selected[task_id]
                )
            ]
            tasks.append(
                {
                    "task_id": task_id,
                    "task_top2_exact": all(item["top2_exact"] for item in items),
                    "items": items,
                }
            )
        controls[arm] = {"tasks": tasks}
    report = {
        "schema_version": 1,
        "experiment": 34,
        "labels_opened_after_all_candidate_hashes": True,
        "solution_sha256": digest_file(solutions_path),
        "primary_metric": config["primary_metric"],
        "arms": arms,
        "diagnostic_controls": controls,
        "pairwise": pairwise,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--calibrate-only", action="store_true")
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--exploratory", action="store_true")
    modes.add_argument("--hard-exploratory", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = parser.parse_args()

    config = read_json(HERE / "config.json")
    validate_config(config)
    args.repo = args.repo.resolve()
    args.challenges = args.challenges.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    if not (args.work / "manifests" / "prepared.json").is_file():
        manifest = prepare(config, args.repo, args.challenges, args.work)
    else:
        manifest = verify_prepared(args.work, args.challenges)
    if args.prepare_only:
        print(canonical({"prepared": True, "primary_ids": manifest["primary_ids"]}))
        return
    if args.model is None or not args.model.joinpath("config.json").is_file():
        raise ValueError("--model must point to a local Transformers model")
    require_24gb_gpu(config)
    require_runtime_versions()
    if args.calibrate_only:
        run_worker("calibrate", config, args.repo, args.challenges, args.model, args.work)
        gate = calibration_gate(config, args.work)
        print(canonical(gate))
        return
    gate_path = args.work / "calibration" / "gate.json"
    if not gate_path.is_file():
        raise RuntimeError("run --calibrate-only before pilot/full")
    gate = read_json(gate_path)
    if not gate.get("passed"):
        raise RuntimeError("calibration gate failed; confirmatory execution is blocked")
    if args.pilot:
        run_worker("pilot", config, args.repo, args.challenges, args.model, args.work)
        return
    if args.exploratory:
        if args.solutions is None or args.output is None:
            raise ValueError("--exploratory requires --solutions and --output")
        run_worker(
            "exploratory",
            config,
            args.repo,
            args.challenges,
            args.model,
            args.work,
        )
        exploratory_config = {**config, "control_arms": []}
        exploratory_manifest = {
            **manifest,
            "primary_ids": manifest["control_ids"],
            "control_ids": [],
        }
        evaluate(
            exploratory_config,
            exploratory_manifest,
            args.solutions.resolve(),
            args.work,
            args.output.resolve(),
        )
        return
    if args.hard_exploratory:
        if args.solutions is None or args.output is None:
            raise ValueError("--hard-exploratory requires --solutions and --output")
        missing = sorted(set(HARD_EXPLORATORY_IDS) - set(read_json(args.challenges)))
        if missing:
            raise RuntimeError(f"hard exploratory task IDs missing: {missing}")
        atomic_json(
            args.work / "manifests" / "hard_exploratory_ids.json",
            HARD_EXPLORATORY_IDS,
        )
        run_worker(
            "hard_exploratory",
            config,
            args.repo,
            args.challenges,
            args.model,
            args.work,
        )
        exploratory_config = {**config, "control_arms": []}
        exploratory_manifest = {
            **manifest,
            "primary_ids": HARD_EXPLORATORY_IDS,
            "control_ids": [],
        }
        evaluate(
            exploratory_config,
            exploratory_manifest,
            args.solutions.resolve(),
            args.work,
            args.output.resolve(),
        )
        return
    if args.solutions is None or args.output is None:
        raise ValueError("--full requires --solutions and --output")
    run_worker("full", config, args.repo, args.challenges, args.model, args.work)
    evaluate(config, manifest, args.solutions.resolve(), args.work, args.output.resolve())


if __name__ == "__main__":
    main()
