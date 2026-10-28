"""Leakage-safe loading, evaluation, and report serialization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from .solver import SolveResult, solve_task
from .types import Grid, validate_grid


def load_challenges(path: str | Path) -> dict[str, Any]:
    """Load challenges and remove test outputs before returning them."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("challenge file must contain an object")
    public: dict[str, Any] = {}
    for task_id, task in raw.items():
        if not isinstance(task, dict):
            raise ValueError(f"task {task_id} must be an object")
        clean = copy.deepcopy(task)
        if not isinstance(clean.get("test"), list):
            raise ValueError(f"task {task_id} has no test list")
        for item in clean["test"]:
            if not isinstance(item, dict):
                raise ValueError(f"task {task_id} has an invalid test item")
            item.pop("output", None)
        public[str(task_id)] = clean
    return public


def solve_challenges(
    challenges: dict[str, Any], task_ids: Iterable[str] | None = None
) -> tuple[dict[str, list[dict[str, Grid]]], dict[str, Any]]:
    """Run label-free inference and return submission-shaped top-2 predictions."""
    selected = sorted(challenges) if task_ids is None else list(task_ids)
    submission: dict[str, list[dict[str, Grid]]] = {}
    reports: dict[str, Any] = {}
    for task_id in selected:
        results = solve_task(challenges[task_id])
        submission[task_id] = [
            {
                f"attempt_{rank}": candidate
                for rank, candidate in enumerate(result.candidates[:2], 1)
            }
            for result in results
        ]
        reports[task_id] = [result.report for result in results]
    return submission, reports


def load_labels(path: str | Path) -> dict[str, list[Grid]]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("label file must contain an object")
    return {
        str(task_id): [
            validate_grid(grid, name=f"labels[{task_id}][{index}]")
            for index, grid in enumerate(outputs)
        ]
        for task_id, outputs in raw.items()
    }


def evaluate_posthoc(
    submission: dict[str, list[Any]],
    labels: dict[str, list[Grid]],
) -> dict[str, Any]:
    """Score already-materialized predictions; this function never invokes solving."""
    tasks: dict[str, Any] = {}
    item_total = item_top1 = item_top2 = 0
    for task_id, items in submission.items():
        truths = labels.get(task_id, [])
        details = []
        for index, candidate_item in enumerate(items):
            if index >= len(truths):
                continue
            candidates = (
                [
                    candidate_item[key]
                    for key in ("attempt_1", "attempt_2")
                    if key in candidate_item
                ]
                if isinstance(candidate_item, dict)
                else candidate_item
            )
            truth = truths[index]
            top1 = bool(candidates) and candidates[0] == truth
            top2 = any(candidate == truth for candidate in candidates[:2])
            details.append({"index": index, "top1_exact": top1, "top2_exact": top2})
            item_total += 1
            item_top1 += int(top1)
            item_top2 += int(top2)
        tasks[task_id] = {
            "items": details,
            "task_top1_exact": bool(details) and all(item["top1_exact"] for item in details),
            "task_top2_exact": bool(details) and all(item["top2_exact"] for item in details),
        }
    return {
        "task_count": len(tasks),
        "item_count": item_total,
        "item_top1_exact": item_top1,
        "item_top2_exact": item_top2,
        "task_top1_exact": sum(item["task_top1_exact"] for item in tasks.values()),
        "task_top2_exact": sum(item["task_top2_exact"] for item in tasks.values()),
        "tasks": tasks,
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
