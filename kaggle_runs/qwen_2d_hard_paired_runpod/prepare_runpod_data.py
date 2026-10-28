"""Build Experiment 024's merged RunPod challenge and sealed solution files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-challenges", type=Path, required=True)
    parser.add_argument("--base-solutions", type=Path, required=True)
    parser.add_argument("--historical-challenges", type=Path, required=True)
    parser.add_argument("--historical-labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = read_json(args.manifest)["primary_ids"]
    if len(expected) != 16 or len(set(expected)) != 16:
        raise AssertionError("manifest must contain exactly 16 unique primary IDs")
    base_challenges = read_json(args.base_challenges)
    base_solutions = read_json(args.base_solutions)
    historical_challenges = read_json(args.historical_challenges)
    historical_labels = read_json(args.historical_labels)
    missing_tasks = sorted(set(expected) - historical_challenges.keys())
    missing_labels = sorted(set(expected) - historical_labels.keys())
    if missing_tasks or missing_labels:
        raise AssertionError(
            f"historical source incomplete: tasks={missing_tasks}, labels={missing_labels}"
        )

    merged_challenges = dict(base_challenges)
    merged_solutions = dict(base_solutions)
    for task_id in expected:
        merged_challenges[task_id] = historical_challenges[task_id]
        merged_solutions[task_id] = historical_labels[task_id]
        task = merged_challenges[task_id]
        if not task.get("train") or not task.get("test"):
            raise AssertionError(f"invalid challenge payload for {task_id}")
        if len(task["test"]) != len(merged_solutions[task_id]):
            raise AssertionError(f"test/label count mismatch for {task_id}")

    args.output.mkdir(parents=True, exist_ok=True)
    challenge_path = args.output / "arc-agi_training_challenges.json"
    solution_path = args.output / "arc-agi_training_solutions.json"
    challenge_path.write_text(json.dumps(merged_challenges, sort_keys=True))
    solution_path.write_text(json.dumps(merged_solutions, sort_keys=True))
    metadata = {
        "base_challenges": str(args.base_challenges),
        "base_solutions": str(args.base_solutions),
        "historical_challenges": str(args.historical_challenges),
        "historical_labels": str(args.historical_labels),
        "manifest": str(args.manifest),
        "primary_ids": expected,
        "merged_challenge_count": len(merged_challenges),
        "merged_solution_count": len(merged_solutions),
        "challenge_sha256": sha256(challenge_path),
        "solution_sha256": sha256(solution_path),
    }
    (args.output / "merged_data_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
