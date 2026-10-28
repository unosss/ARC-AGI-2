"""Fail-closed grading for the ten structured blind traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
TRACE_DIR = EXP / "traces"
TASK_IDS = [
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
]
ANSWER_KEY = ROOT / "outputs/assistant_blind10_20261014/answer_key.json"


def valid_grid(grid) -> bool:
    return (
        isinstance(grid, list)
        and bool(grid)
        and isinstance(grid[0], list)
        and bool(grid[0])
        and all(
            isinstance(row, list)
            and len(row) == len(grid[0])
            and all(type(cell) is int and 0 <= cell <= 9 for cell in row)
            for row in grid
        )
    )


def main() -> None:
    required = {
        "task_id",
        "observations",
        "correspondences",
        "hypotheses",
        "revision_history",
        "selected_rule",
        "test_construction",
        "test",
    }
    traces = {}
    trace_hashes = {}
    # Complete this boundary check before opening the answer key.
    for task_id in TASK_IDS:
        path = TRACE_DIR / f"{task_id}.json"
        if not path.exists():
            raise RuntimeError(f"grading sealed: missing trace {task_id}")
        payload = json.loads(path.read_text())
        if payload.get("task_id") != task_id:
            raise ValueError(f"trace task mismatch: {task_id}")
        if not required.issubset(payload):
            raise ValueError(f"trace schema incomplete: {task_id}")
        if not payload["hypotheses"] or not payload["selected_rule"]:
            raise ValueError(f"trace has no falsifiable rule artifacts: {task_id}")
        if not isinstance(payload["test"], list) or not all(
            valid_grid(grid) for grid in payload["test"]
        ):
            raise ValueError(f"invalid test prediction: {task_id}")
        traces[task_id] = payload
        trace_hashes[task_id] = hashlib.sha256(path.read_bytes()).hexdigest()

    answers = json.loads(ANSWER_KEY.read_text())
    rows = []
    total_cells = 0
    correct_cells = 0
    exact_items = 0
    total_items = 0
    for task_id in TASK_IDS:
        predicted = traces[task_id]["test"]
        truth = answers[task_id]
        if len(predicted) != len(truth):
            item_rows = [
                {
                    "index": index,
                    "exact": False,
                    "cell_accuracy": 0.0,
                    "shape_match": False,
                }
                for index in range(max(len(predicted), len(truth)))
            ]
        else:
            item_rows = []
            for index, (candidate, expected) in enumerate(zip(predicted, truth)):
                shape_match = (
                    len(candidate) == len(expected)
                    and len(candidate[0]) == len(expected[0])
                )
                exact = candidate == expected
                if shape_match:
                    item_total = len(expected) * len(expected[0])
                    item_correct = sum(
                        candidate[row][column] == expected[row][column]
                        for row in range(len(expected))
                        for column in range(len(expected[0]))
                    )
                    accuracy = item_correct / item_total
                    total_cells += item_total
                    correct_cells += item_correct
                else:
                    accuracy = 0.0
                    total_cells += len(expected) * len(expected[0])
                item_rows.append(
                    {
                        "index": index,
                        "exact": exact,
                        "cell_accuracy": accuracy,
                        "shape_match": shape_match,
                    }
                )
                exact_items += int(exact)
                total_items += 1
        rows.append(
            {
                "task_id": task_id,
                "task_exact": len(predicted) == len(truth)
                and all(item["exact"] for item in item_rows),
                "items": item_rows,
                "selected_rule": traces[task_id]["selected_rule"],
                "hypothesis_count": len(traces[task_id]["hypotheses"]),
            }
        )
    result = {
        "experiment": 30,
        "label_boundary": "all ten trace files validated before answer_key read",
        "trace_sha256": trace_hashes,
        "task_exact": sum(row["task_exact"] for row in rows),
        "task_count": len(rows),
        "item_exact": exact_items,
        "item_count": total_items,
        "cell_accuracy": correct_cells / total_cells,
        "tasks": rows,
    }
    (EXP / "score.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
