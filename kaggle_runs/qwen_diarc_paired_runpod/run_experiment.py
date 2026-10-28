"""Parent-only orchestration and grading for Experiment 036."""

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

from kaggle_runs.qwen_diarc_paired_runpod.protocol import (
    ARMS,
    DEFAULT_HARD_TASK,
    atomic_json,
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
    selected = json.loads(json.dumps({task_id: challenges[task_id] for task_id in task_ids}))
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


def run_workers(
    mode: str,
    config_path: Path,
    challenges_path: Path,
    task_ids: list[str],
    work: Path,
    *,
    model_dir: Path | None,
    adapter: str | None,
) -> None:
    for arm in ARMS:
        command = worker_command(
            arm=arm,
            mode=mode,
            config_path=config_path,
            challenges_path=challenges_path,
            task_ids=task_ids,
            work=work,
            model_dir=model_dir,
            adapter=adapter,
        )
        if any("solution" in argument.lower() for argument in command):
            raise AssertionError("solution path must never reach label-blind worker")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO)
        subprocess.run(command, check=True, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true", help="CPU lifecycle, no model load")
    modes.add_argument("--calibrate", action="store_true", help="reduced one-task GPU run")
    modes.add_argument(
        "--hard-exploratory", action="store_true", help="full registered one-task run"
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        raise ValueError("--model-dir is required for calibration and hard runs")
    if args.task_id:
        task_ids = args.task_id
    elif mode == "calibrate":
        task_ids = [config["calibration"]["task_id"]]
    else:
        task_ids = config["hard_exploratory"].get(
            "default_task_ids", [DEFAULT_HARD_TASK]
        )
    if len(task_ids) != 1:
        raise ValueError("Experiment 036 modes are paired one-task comparisons")

    # Parent creates the worker mount without ever copying a held-out output.
    challenges = json.loads(args.challenges.read_text())
    blind = label_blind_tasks(challenges, task_ids)
    args.work.mkdir(parents=True, exist_ok=True)
    blind_path = args.work / "label_blind_challenges.json"
    atomic_json(blind_path, blind)
    atomic_json(
        args.work / "run_manifest.json",
        {"experiment": 36, "mode": mode, "arms": list(ARMS), "task_ids": task_ids},
    )

    run_workers(
        mode,
        args.config,
        blind_path,
        task_ids,
        args.work,
        model_dir=args.model_dir,
        adapter=args.adapter,
    )

    # This is the label-opening line: both arms' bytes are immutable first.
    verify_hash_barrier(args.work, task_ids)
    solutions = json.loads(args.solutions.read_text())
    report = grade(args.work, task_ids, solutions)
    report["mode"] = mode
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
