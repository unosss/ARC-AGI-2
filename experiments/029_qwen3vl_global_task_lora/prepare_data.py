"""Reconstruct the exact Experiment 024 merged artifacts from their sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def matching_serialization(value, expected: str) -> bytes:
    encodings = [
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode(),
        json.dumps(
            value, ensure_ascii=False, sort_keys=False, separators=(",", ":")
        ).encode(),
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode(),
        json.dumps(value, ensure_ascii=False, sort_keys=False).encode(),
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
        (json.dumps(value, ensure_ascii=False, sort_keys=False, indent=2) + "\n").encode(),
    ]
    for payload in encodings:
        if hashlib.sha256(payload).hexdigest() == expected:
            return payload
    observed = [hashlib.sha256(payload).hexdigest() for payload in encodings]
    raise RuntimeError(
        f"no known serialization matched {expected}; observed hashes: {observed}"
    )


def merge(base_path: Path, additional_path: Path) -> dict:
    base = json.loads(base_path.read_text())
    additional = json.loads(additional_path.read_text())
    overlap = set(base) & set(additional)
    if overlap:
        raise ValueError(f"task IDs overlap across sources: {sorted(overlap)[:5]}")
    return {**base, **additional}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-challenges", type=Path, required=True)
    parser.add_argument("--base-solutions", type=Path, required=True)
    parser.add_argument("--additional-challenges", type=Path, required=True)
    parser.add_argument("--additional-solutions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    challenges = merge(args.base_challenges, args.additional_challenges)
    solutions = merge(args.base_solutions, args.additional_solutions)
    if len(challenges) != 1016 or len(solutions) != 1016:
        raise ValueError("Experiment 024 merged artifacts must each have 1,016 tasks")
    if not set(config["task_ids"]).issubset(challenges):
        raise ValueError("merged challenge is missing fixed task IDs")
    if not set(config["task_ids"]).issubset(solutions):
        raise ValueError("merged solutions are missing fixed task IDs")
    challenge_bytes = matching_serialization(
        challenges, config["challenge_sha256"]
    )
    solution_bytes = matching_serialization(solutions, config["solution_sha256"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "merged_challenges.json").write_bytes(challenge_bytes)
    (args.output_dir / "merged_solutions.json").write_bytes(solution_bytes)
    print(
        json.dumps(
            {
                "challenge_count": len(challenges),
                "solution_count": len(solutions),
                "challenge_sha256": hashlib.sha256(challenge_bytes).hexdigest(),
                "solution_sha256": hashlib.sha256(solution_bytes).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
