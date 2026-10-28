"""Experiment 018 Kaggle runner with a process-enforced label boundary."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


SOURCE_ARCHIVE = "__SOURCE_ARCHIVE__"
CONFIG = json.loads(r"""__RUN_CONFIG__""")
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/experiment_018_v2")


def extract_sources() -> Path:
    root = Path("/tmp/experiment_018_source")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
        archive.extractall(root)
    return root


def find_arc_file(fragment: str) -> Path:
    matches = sorted(path for path in INPUT_ROOT.rglob("*.json") if fragment in path.name)
    if not matches:
        raise FileNotFoundError(f"ARC file containing {fragment!r} not found")
    return matches[0]


def find_package_root(package: str) -> Path:
    roots = [INPUT_ROOT, Path("/kaggle/usr/lib")]
    matches = sorted(
        match
        for root in roots
        if root.exists()
        for match in root.rglob(f"{package}/__init__.py")
    )
    if not matches:
        raise FileNotFoundError(f"offline package {package!r} not found")
    return matches[0].parent.parent


def append_event(path: Path, name: str) -> None:
    with path.open("a") as handle:
        handle.write(
            json.dumps({"event": name, "time": time.time()}, separators=(",", ":"))
            + "\n"
        )


def run_phase(
    phase: str,
    source_root: Path,
    dependency_root: Path,
    solver_path: Path,
    lifecycle_path: Path,
    *,
    oracle_payload: Path | None = None,
) -> dict:
    result_path = WORK_ROOT / f"{phase}_result.json"
    log_path = WORK_ROOT / f"{phase}.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(source_root), str(dependency_root))),
            "ARC_STAGE_PHASE": phase,
            "ARC_STAGE_CONFIG": json.dumps(CONFIG, separators=(",", ":")),
            "ARC_STAGE_SOLVER_TASK": str(solver_path),
            "ARC_STAGE_RESULT": str(result_path),
            "ARC_STAGE_LIFECYCLE": str(lifecycle_path),
            "ARC_N_TRAIN_AUG": str(CONFIG["n_train_aug"]),
            "ARC_EPOCHS": str(CONFIG["epochs"]),
            "ARC_LR": str(CONFIG["learning_rate"]),
            "ARC_LORA_SEED": str(CONFIG["lora_seed"]),
            "ARC_TRAIN_AUG_SEED": str(CONFIG["train_aug_seed"]),
            "ARC_MIN_PROB": str(CONFIG["min_prob"]),
            "ARC_DFS_WINDOW": str(CONFIG["dfs_window_seconds"]),
            "UNSLOTH_DISABLE_STATISTICS": "1",
            "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    if oracle_payload is not None:
        env["ARC_STAGE_ORACLE_PAYLOAD"] = str(oracle_payload)
    started = time.time()
    with log_path.open("w") as log:
        process = subprocess.run(
            [sys.executable, str(source_root / "stage_worker.py")],
            cwd=source_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=CONFIG["phase_timeout_seconds"],
        )
    if process.returncode:
        raise RuntimeError(f"{phase} worker failed with exit code {process.returncode}")
    append_event(lifecycle_path, f"{phase}_worker_exited")
    result = json.loads(result_path.read_text())
    result["phase_seconds"] = round(time.time() - started, 3)
    return result


def main() -> None:
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError("Experiment 018 is a labeled diagnostic, not a submission")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    source_root = extract_sources()
    dependency_root = find_package_root("unsloth")
    sys.path.insert(0, str(source_root))
    from arc_solver.nvarc_stage_diagnosis import (
        TASK_ID,
        aggregate_report,
        build_oracle_conditions,
        build_training_bundle,
    )

    challenges_path = find_arc_file("evaluation_challenges")
    challenges = json.loads(challenges_path.read_text())
    solver_bundle = build_training_bundle(challenges, TASK_ID)
    solver_dir = WORK_ROOT / "solver_input"
    solver_dir.mkdir(parents=True, exist_ok=True)
    solver_path = solver_dir / "challenge.json"
    solver_path.write_text(json.dumps(solver_bundle, separators=(",", ":")))
    lifecycle_path = WORK_ROOT / "lifecycle.jsonl"

    # No solution path, label path, truth, or oracle payload exists in this phase.
    ordinary = run_phase(
        "ordinary", source_root, dependency_root, solver_path, lifecycle_path
    )
    if ordinary["labels_available"] is not False:
        raise AssertionError("ordinary worker reported label availability")

    # Parent-only label access begins strictly after ordinary subprocess exit.
    solutions_path = find_arc_file("evaluation_solutions")
    solutions = json.loads(solutions_path.read_text())
    truth = solutions[TASK_ID][CONFIG["test_index"]]
    append_event(lifecycle_path, "labels_opened")
    ordinary["baseline"]["truth_present"] = any(
        candidate["grid"] == truth
        for candidate in ordinary["baseline"]["candidates"]
    )
    ordinary["baseline"]["exact_candidate_count"] = sum(
        candidate["grid"] == truth
        for candidate in ordinary["baseline"]["candidates"]
    )
    ordinary["baseline_exp015_comparable"]["truth_present"] = any(
        candidate["grid"] == truth
        for candidate in ordinary["baseline_exp015_comparable"]["candidates"]
    )
    ordinary["baseline_exp015_comparable"]["exact_candidate_count"] = sum(
        candidate["grid"] == truth
        for candidate in ordinary["baseline_exp015_comparable"]["candidates"]
    )
    parent_dir = WORK_ROOT / "parent_oracle_only"
    parent_dir.mkdir(parents=True, exist_ok=True)
    payload = build_oracle_conditions(solver_bundle[TASK_ID], truth)
    oracle_path = parent_dir / "oracle_payload.json"
    oracle_path.write_text(json.dumps(payload, separators=(",", ":")))
    append_event(lifecycle_path, "oracle_payload_written")

    # This worker retrains from the label-free solver bundle. Its code opens the
    # oracle payload only after asserting that all 128 optimizer steps completed.
    oracle = run_phase(
        "oracle",
        source_root,
        dependency_root,
        solver_path,
        lifecycle_path,
        oracle_payload=oracle_path,
    )
    lifecycle = [
        json.loads(line) for line in lifecycle_path.read_text().splitlines() if line
    ]
    report = aggregate_report(CONFIG, ordinary, oracle, lifecycle)
    report_path = Path("/kaggle/working") / CONFIG["report_filename"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (WORK_ROOT / CONFIG["report_filename"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
