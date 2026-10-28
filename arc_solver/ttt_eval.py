"""Leakage-safe utilities for ARC test-time-training experiments."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any


Grid = list[list[int]]
PHASE1_SEED = 13013
REVIEWED_TASK_IDS = {
    "03560426",
    "045e512c",
    "05269061",
    "05a7bcf2",
    "05f2a901",
    "0607ce86",
    "06df4c85",
    "09629e4f",
    "0962bcdd",
    "0a1d4ef5",
}


def is_valid_grid(value: Any) -> bool:
    """Return whether value is a rectangular ARC grid."""
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return False
    if not isinstance(value[0], list) or not 1 <= len(value[0]) <= 30:
        return False
    width = len(value[0])
    return all(
        isinstance(row, list)
        and len(row) == width
        and all(type(cell) is int and 0 <= cell <= 9 for cell in row)
        for row in value
    )


def select_task_ids(
    challenges: dict[str, Any],
    count: int = 20,
    seed: int = PHASE1_SEED,
    excluded: set[str] | None = None,
) -> list[str]:
    """Select a stable sample without relying on source dictionary order."""
    excluded_ids = REVIEWED_TASK_IDS if excluded is None else excluded
    candidates = sorted(set(challenges) - set(excluded_ids))
    if count < 1 or count > len(candidates):
        raise ValueError("count must be between 1 and the number of candidates")
    return sorted(random.Random(seed).sample(candidates, count))


def hide_test_outputs(task: dict[str, Any]) -> dict[str, Any]:
    """Copy one task and ensure no test label reaches the solver."""
    public_task = copy.deepcopy(task)
    tests = public_task.get("test")
    if not isinstance(tests, list):
        raise ValueError("task test field must be a list")
    for item in tests:
        if not isinstance(item, dict) or not is_valid_grid(item.get("input")):
            raise ValueError("each test item must contain a valid input grid")
        item.pop("output", None)
    return public_task


def build_phase1_bundle(
    challenges: dict[str, Any],
    solutions: dict[str, Any],
    task_ids: list[str],
) -> tuple[dict[str, Any], dict[str, list[Grid]]]:
    """Build separated inference challenges and evaluator-only labels."""
    public: dict[str, Any] = {}
    labels: dict[str, list[Grid]] = {}
    for task_id in task_ids:
        if task_id not in challenges or task_id not in solutions:
            raise KeyError(f"missing challenge or solution for {task_id}")
        outputs = solutions[task_id]
        if not isinstance(outputs, list) or not outputs or not all(
            is_valid_grid(grid) for grid in outputs
        ):
            raise ValueError(f"invalid solution grids for {task_id}")
        task = hide_test_outputs(challenges[task_id])
        if len(task["test"]) != len(outputs):
            raise ValueError(f"test/solution count mismatch for {task_id}")
        public[task_id] = task
        labels[task_id] = copy.deepcopy(outputs)
    return public, labels


def transform_grid(grid: Grid, transform: str) -> Grid:
    """Apply one of the eight D4 transforms."""
    if not is_valid_grid(grid):
        raise ValueError("invalid grid")
    operations = {
        "identity": lambda value: [row[:] for row in value],
        "rot90": lambda value: [list(row) for row in zip(*value[::-1])],
        "rot180": lambda value: [row[::-1] for row in value[::-1]],
        "rot270": lambda value: [list(row) for row in zip(*value)][::-1],
        "hmirror": lambda value: [row[:] for row in value[::-1]],
        "vmirror": lambda value: [row[::-1] for row in value],
        "dmirror": lambda value: [list(row) for row in zip(*value)],
        "cmirror": lambda value: [
            list(row) for row in zip(*(row[::-1] for row in value[::-1]))
        ],
    }
    try:
        return operations[transform](grid)
    except KeyError as exc:
        raise ValueError(f"unknown transform: {transform}") from exc


def inverse_transform_grid(grid: Grid, transform: str) -> Grid:
    inverse = {
        "identity": "identity",
        "rot90": "rot270",
        "rot180": "rot180",
        "rot270": "rot90",
        "hmirror": "hmirror",
        "vmirror": "vmirror",
        "dmirror": "dmirror",
        "cmirror": "cmirror",
    }
    try:
        return transform_grid(grid, inverse[transform])
    except KeyError as exc:
        raise ValueError(f"unknown transform: {transform}") from exc


def _cell_accuracy(prediction: Any, truth: Grid) -> float:
    if not is_valid_grid(prediction):
        return 0.0
    total = max(
        sum(len(row) for row in prediction),
        sum(len(row) for row in truth),
    )
    matching = sum(
        prediction[row][column] == truth[row][column]
        for row in range(min(len(prediction), len(truth)))
        for column in range(min(len(prediction[row]), len(truth[row])))
    )
    return matching / total


def score_condition(
    submission: dict[str, Any],
    solutions: dict[str, list[Grid]],
) -> dict[str, Any]:
    """Score Kaggle-style top-two attempts without exposing labels upstream."""
    task_records = []
    valid_attempts = total_attempts = exact_items = total_items = 0
    top1_cells = []
    for task_id, expected_outputs in solutions.items():
        predicted_outputs = submission.get(task_id, [])
        item_records = []
        for index, truth in enumerate(expected_outputs):
            item = predicted_outputs[index] if index < len(predicted_outputs) else {}
            attempts = [
                item.get("attempt_1") if isinstance(item, dict) else None,
                item.get("attempt_2") if isinstance(item, dict) else None,
            ]
            valid = [is_valid_grid(attempt) for attempt in attempts]
            exact = any(
                is_valid and attempt == truth
                for attempt, is_valid in zip(attempts, valid)
            )
            valid_attempts += sum(valid)
            total_attempts += len(attempts)
            exact_items += int(exact)
            total_items += 1
            top1_cells.append(_cell_accuracy(attempts[0], truth))
            item_records.append(
                {
                    "test_index": index,
                    "valid_attempts": sum(valid),
                    "top2_exact": exact,
                    "top1_cell_accuracy": top1_cells[-1],
                }
            )
        task_records.append(
            {
                "task_id": task_id,
                "exact": bool(item_records)
                and all(record["top2_exact"] for record in item_records),
                "items": item_records,
            }
        )
    return {
        "task_count": len(task_records),
        "task_exact": sum(record["exact"] for record in task_records),
        "item_count": total_items,
        "item_exact": exact_items,
        "valid_grid_rate": valid_attempts / total_attempts if total_attempts else 0.0,
        "top1_cell_accuracy": sum(top1_cells) / len(top1_cells)
        if top1_cells
        else 0.0,
        "tasks": task_records,
    }


def compare_conditions(
    submissions: dict[str, dict[str, Any]],
    solutions: dict[str, list[Grid]],
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the same labels and metrics to every experimental condition."""
    return {
        "task_ids": sorted(solutions),
        "conditions": {
            name: score_condition(submission, solutions)
            for name, submission in submissions.items()
        },
        "timings": timings or {},
    }


def write_phase1_bundle(
    challenges_path: Path,
    solutions_path: Path,
    task_ids_path: Path,
    output_dir: Path,
) -> None:
    challenges = json.loads(challenges_path.read_text())
    solutions = json.loads(solutions_path.read_text())
    task_ids = json.loads(task_ids_path.read_text())
    public, labels = build_phase1_bundle(challenges, solutions, task_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1_challenges.json").write_text(
        json.dumps(public, separators=(",", ":"))
    )
    (output_dir / "phase1_solutions.json").write_text(
        json.dumps(labels, separators=(",", ":"))
    )
