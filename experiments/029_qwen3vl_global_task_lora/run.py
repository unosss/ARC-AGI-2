"""Sequential pilot/full orchestrator for one 24 GiB GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from core import file_sha256


HERE = Path(__file__).resolve().parent


def worker_command(args, arm: str, task_id: str) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "worker.py"),
        "--config",
        str(args.config),
        "--challenge",
        str(args.challenge),
        "--output-root",
        str(args.output_root),
        "--arm",
        arm,
        "--task-id",
        task_id,
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    if arm == "perception_task_lora":
        command.extend(["--global-adapter", str(args.global_adapter)])
    return command


def run_worker(args, arm: str, task_id: str) -> None:
    subprocess.run(worker_command(args, arm, task_id), check=True)


def load_candidate(root: Path, arm: str, task_id: str) -> dict:
    return json.loads((root / arm / task_id / "candidate.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--global-adapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    import torch

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "bitsandbytes",
                "qwen-vl-utils",
            )
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment_path = args.output_root / "environment.json"
    if environment_path.exists():
        if json.loads(environment_path.read_text()) != environment:
            raise RuntimeError("run environment changed; refusing mixed resume")
    else:
        environment_path.write_text(
            json.dumps(environment, ensure_ascii=False, indent=2)
        )
    if file_sha256(args.challenge) != config["challenge_sha256"]:
        raise ValueError("challenge does not match fixed SHA-256")
    preflight_command = [
        sys.executable,
        str(HERE / "preflight.py"),
        "--config",
        str(args.config),
        "--challenge",
        str(args.challenge),
        "--output",
        str(args.output_root / "sequence_preflight.json"),
    ]
    if args.model_path:
        preflight_command.extend(["--model-path", args.model_path])
    subprocess.run(preflight_command, check=True)
    probe_command = [
        sys.executable,
        str(HERE / "adapter_probe.py"),
        "--config",
        str(args.config),
        "--global-adapter",
        str(args.global_adapter),
        "--output",
        str(args.output_root / "adapter_probe.json"),
    ]
    if args.model_path:
        probe_command.extend(["--model-path", args.model_path])
    subprocess.run(probe_command, check=True)
    pilot = config["pilot_task_id"]
    for arm in config["arms"]:
        run_worker(args, arm, pilot)

    pilot_rows = [
        load_candidate(args.output_root, arm, pilot) for arm in config["arms"]
    ]
    projected = (
        sum(row["total_seconds"] for row in pilot_rows)
        * len(config["task_ids"])
        * 1.25
    )
    peak = max(row["peak_allocated_bytes"] for row in pilot_rows)
    initialization_match = (
        len({row["initial_task_adapter_digest"] for row in pilot_rows}) == 1
    )
    gate = {
        "pilot_task_id": pilot,
        "arm_total_seconds": {
            arm: row["total_seconds"]
            for arm, row in zip(config["arms"], pilot_rows)
        },
        "projected_seconds_with_25_percent_margin": projected,
        "peak_allocated_bytes": peak,
        "train_reconstruction": {
            arm: {
                "exact": sum(
                    row["top2_exact"]
                    for row in candidate["train_reconstructions"]
                ),
                "count": len(candidate["train_reconstructions"]),
                "truncated_sequences": candidate["truncated_sequences"],
            }
            for arm, candidate in zip(config["arms"], pilot_rows)
        },
        "passed_runtime": projected <= config["max_projected_seconds"],
        "passed_memory": peak <= config["max_peak_allocated_bytes"],
        "passed_cross_arm_initialization_parity": initialization_match,
    }
    (args.output_root / "pilot_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2)
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not all(
        (
            gate["passed_runtime"],
            gate["passed_memory"],
            gate["passed_cross_arm_initialization_parity"],
        )
    ):
        raise RuntimeError("pilot feasibility gate failed; full run is aborted")
    if args.pilot_only:
        return
    if args.solutions is None:
        raise ValueError("--solutions is required for the full run")

    for task_id in config["task_ids"]:
        for arm in config["arms"]:
            run_worker(args, arm, task_id)
    report = args.output_root / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "finalize.py"),
            "--config",
            str(args.config),
            "--challenge",
            str(args.challenge),
            "--solutions",
            str(args.solutions),
            "--output-root",
            str(args.output_root),
            "--report",
            str(report),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "analyze.py"),
            "--report",
            str(report),
            "--output",
            str(args.output_root / "REPORT.md"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
