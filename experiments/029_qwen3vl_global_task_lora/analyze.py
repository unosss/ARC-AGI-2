"""Create the human-readable paired analysis from the sealed report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def stage(task: dict) -> str:
    if task["train_reconstruction_rate"] < 1.0:
        return "train_reconstruction_failure"
    if not all(item["candidate_count"] for item in task["items"]):
        return "invalid_generation"
    if not task["top2_exact"]:
        return "test_generalization_failure"
    return "solved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    arms = report["arms"]
    lines = [
        "# Experiment 029 Results",
        "",
        "This is a paired result on the preregistered 16-task hard set, not a "
        "population or leaderboard accuracy estimate.",
        "",
        "## Aggregate",
        "",
    ]
    for arm, result in arms.items():
        total_seconds = sum(row["train_seconds"] for row in result["tasks"])
        peak = max(row["peak_allocated_bytes"] for row in result["tasks"])
        lines.extend(
            [
                f"- `{arm}`: task top-2 {result['task_top2_exact']}/"
                f"{result['task_count']}; item top-2 {result['item_top2_exact']}/"
                f"{result['item_count']}; mean best-cell "
                f"{result['mean_best_cell_accuracy']:.4f}; valid items "
                f"{result['valid_item_rate']:.4f}; train time "
                f"{total_seconds:.1f}s; peak {peak / 2**30:.2f} GiB",
            ]
        )
    paired = report["paired"]
    lines.extend(
        [
            "",
            "## Paired effect",
            "",
            f"- perception wins: {len(paired['perception_wins'])} "
            f"({', '.join(paired['perception_wins']) or 'none'})",
            f"- perception losses: {len(paired['perception_losses'])} "
            f"({', '.join(paired['perception_losses']) or 'none'})",
            f"- ties: {paired['ties']}",
            f"- net task gain: {paired['net_task_gain']:+d}",
            "",
            "## Failure stage by task",
            "",
        ]
    )
    task_maps = {
        arm: {row["task_id"]: row for row in result["tasks"]}
        for arm, result in arms.items()
    }
    for task_id in task_maps["base_task_lora"]:
        base = task_maps["base_task_lora"][task_id]
        perception = task_maps["perception_task_lora"][task_id]
        lines.append(
            f"- `{task_id}`: base={int(base['top2_exact'])} ({stage(base)}); "
            f"perception={int(perception['top2_exact'])} ({stage(perception)})"
        )
    lines.extend(
        [
            "",
            "A positive net gain supports a causal contribution from the frozen "
            "perception adapter under this task-training and decoder budget. A "
            "tie or loss does not establish that perception pretraining is "
            "generally useless; it rejects this specific checkpoint/composition.",
            "",
        ]
    )
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
