"""Parent-only orchestration and grading for Experiment 039."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kaggle_runs.qwen_diarc_paired_runpod.protocol import atomic_json
from kaggle_runs.qwen_scale_lattice_lora_runpod.protocol import (
    ARMS,
    DEFAULT_HARD_TASK,
    grade,
    reject_test_outputs,
    validate_config,
    verify_hash_barrier,
)


def label_blind_tasks(
    challenges: dict[str, Any], task_ids: list[str]
) -> dict[str, Any]:
    missing = sorted(set(task_ids) - set(challenges))
    if missing:
        raise ValueError(f"challenge tasks missing: {missing}")
    selected = json.loads(json.dumps({key: challenges[key] for key in task_ids}))
    reject_test_outputs(selected)
    return selected


def worker_command(
    *,
    arm: str,
    mode: str,
    config_path: Path,
    challenges_path: Path,
    task_ids: list[str],
    work: Path,
    model_dir: Path | None,
    adapter: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "worker.py"),
        "--arm",
        arm,
        "--mode",
        mode,
        "--config",
        str(config_path),
        "--challenges",
        str(challenges_path),
        "--work",
        str(work),
    ]
    for task_id in task_ids:
        command.extend(["--task-id", task_id])
    if model_dir is not None:
        command.extend(["--model-dir", str(model_dir)])
    if adapter is not None:
        command.extend(["--adapter", adapter])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--calibrate", action="store_true")
    modes.add_argument("--hard-exploratory", action="store_true")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--adapter")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mode = (
        "smoke"
        if args.smoke
        else "calibrate"
        if args.calibrate
        else "hard-exploratory"
    )
    config = json.loads(args.config.read_text())
    validate_config(config)
    if mode != "smoke" and args.model_dir is None:
        raise ValueError("--model-dir is required outside smoke mode")
    task_ids = args.task_id or (
        [config["calibration"]["task_id"]]
        if mode == "calibrate"
        else config["hard_exploratory"].get("default_task_ids", [DEFAULT_HARD_TASK])
    )
    if len(task_ids) != 1:
        raise ValueError("Experiment 039 modes are paired one-task comparisons")

    challenges = json.loads(args.challenges.read_text())
    blind = label_blind_tasks(challenges, task_ids)
    args.work.mkdir(parents=True, exist_ok=True)
    blind_path = args.work / "label_blind_challenges.json"
    atomic_json(blind_path, blind)
    atomic_json(
        args.work / "run_manifest.json",
        {"experiment": 39, "mode": mode, "arms": list(ARMS), "task_ids": task_ids},
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO)
    for arm in ARMS:
        command = worker_command(
            arm=arm,
            mode=mode,
            config_path=args.config,
            challenges_path=blind_path,
            task_ids=task_ids,
            work=args.work,
            model_dir=args.model_dir,
            adapter=args.adapter,
        )
        if any("solution" in argument.lower() for argument in command):
            raise AssertionError("solution path must never reach label-blind worker")
        subprocess.run(command, check=True, env=environment)

    verify_hash_barrier(args.work, task_ids)
    report = grade(args.work, task_ids, json.loads(args.solutions.read_text()))
    report["mode"] = mode
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
