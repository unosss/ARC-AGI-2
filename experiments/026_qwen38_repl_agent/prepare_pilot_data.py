#!/usr/bin/env python3
"""Prepare challenge-only pilot input while keeping the label separate."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads((args.repo / "experiments/026_qwen38_repl_agent/pilot_config.json").read_text())
    task_id = cfg["task_id"]
    challenges = json.loads((args.repo / cfg["source_challenges"]).read_text())
    solutions = json.loads((args.repo / cfg["source_solutions"]).read_text())
    if task_id not in challenges or task_id not in solutions:
        raise KeyError(f"{task_id} absent from historical data")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "challenge.json").write_text(
        json.dumps({task_id: challenges[task_id]}, separators=(",", ":")) + "\n"
    )
    label_dir = args.out / "sealed_labels"
    label_dir.mkdir(exist_ok=True)
    (label_dir / "solution.json").write_text(
        json.dumps({task_id: solutions[task_id]}, separators=(",", ":")) + "\n"
    )
    print(f"prepared {task_id}: challenge.json and sealed_labels/solution.json")


if __name__ == "__main__":
    main()
