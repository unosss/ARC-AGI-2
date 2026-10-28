"""Command-line runner for the deterministic hybrid 2D solver."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from .io import (
    evaluate_posthoc,
    load_challenges,
    load_labels,
    solve_challenges,
    write_json,
)


def _challenge_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    for name in ("challenges.json", "challenge.json"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no challenge JSON found at {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="challenge JSON or a directory containing it")
    parser.add_argument("--task-ids", nargs="*", help="selected task IDs")
    parser.add_argument("--labels", help="optional evaluator-only labels JSON")
    parser.add_argument("--report", required=True, help="output report JSON")
    parser.add_argument("--submission", required=True, help="output predictions JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    challenges = load_challenges(_challenge_path(args.source))
    selected = args.task_ids or sorted(challenges)
    submission, task_reports = solve_challenges(challenges, selected)
    # Predictions are fully materialized and serialized before evaluator labels open.
    write_json(args.submission, submission)
    report = {
        "experiment": "020_component_scene_programs",
        "phase": "deterministic_dual_scene_program_synthesis",
        "label_free_inference": True,
        "task_ids": selected,
        "task_reports": task_reports,
        "train_fit_tasks": sum(
            all(item["train_exact_program_count"] > 0 for item in reports)
            for reports in task_reports.values()
        ),
        "abstained_items": sum(
            item["abstained"] for reports in task_reports.values() for item in reports
        ),
        "runtime_seconds_before_label_load": time.perf_counter() - started,
    }
    if args.labels:
        labels = load_labels(args.labels)
        report["posthoc_evaluation"] = evaluate_posthoc(submission, labels)
    report["runtime_seconds_total"] = time.perf_counter() - started
    write_json(args.report, report)
    return 0
