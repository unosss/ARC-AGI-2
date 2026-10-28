"""Create a deterministic twenty-task label-free panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE_DIR = (
    ROOT
    / "kaggle_runs/codet5_small_eval_l4/output_v4/experiment_014/data"
)
CHALLENGE_SOURCE = SOURCE_DIR / "phase1_challenges.json"
SOLUTION_SOURCE = SOURCE_DIR / "phase1_solutions.json"
EXPLICIT_EXCLUSIONS = {
    "42f14c03",
    "72207abc",
    "0d87d2a6",
    "bf32578f",
    "7ee1c6ea",
    "ad38a9d0",
    "7039b2d7",
    "b27ca6d3",
    "f8a8fe49",
    "fcc82909",
    "0934a4d8",
    "135a2760",
}


def selection_key(task_id: str) -> str:
    return hashlib.sha256(f"experiment-033:{task_id}".encode()).hexdigest()


def main() -> None:
    challenges = json.loads(CHALLENGE_SOURCE.read_text())
    solutions = json.loads(SOLUTION_SOURCE.read_text())
    excluded = EXPLICIT_EXCLUSIONS
    eligible = sorted(
        set(challenges) & set(solutions) - excluded,
        key=selection_key,
    )
    if len(eligible) < 20:
        raise RuntimeError(
            f"Only {len(eligible)} unseen tasks remain after exclusions"
        )
    selected = eligible[:20]
    challenge_dir = EXP / "challenges"
    challenge_dir.mkdir(parents=True, exist_ok=True)
    for stale in challenge_dir.glob("*.json"):
        stale.unlink()
    for task_id in selected:
        (challenge_dir / f"{task_id}.json").write_text(
            json.dumps(challenges[task_id], separators=(",", ":")) + "\n"
        )
    (EXP / "task_ids.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    (EXP / "private_answer_key.json").write_text(
        json.dumps(
            {task_id: solutions[task_id] for task_id in selected},
            separators=(",", ":"),
        )
        + "\n"
    )
    (EXP / "selection_metadata.json").write_text(
        json.dumps(
            {
                "method": "lowest sha256(experiment-033:<task_id>)",
                "source_challenge_count": len(challenges),
                "source_solution_count": len(solutions),
                "excluded_frontier_or_manual_task_count": len(excluded),
                "eligible_count": len(eligible),
                "selected_count": len(selected),
                "selected": selected,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

