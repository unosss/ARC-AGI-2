"""Grade frozen representation-only executions and controlled interventions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
TASKS = ["72207abc", "ad38a9d0", "7039b2d7", "bf32578f", "0d87d2a6"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_execution_grid(data: dict) -> list[list[int]]:
    test = data["test"]
    if test and isinstance(test[0], dict):
        return test[0]["output"]
    return test


def compare(pred: list[list[int]], truth: list[list[int]]) -> dict:
    shape_match = (
        len(pred) == len(truth)
        and all(len(a) == len(b) for a, b in zip(pred, truth))
    )
    if not shape_match:
        return {
            "exact": False,
            "shape_match": False,
            "cell_accuracy": 0.0,
            "wrong_cells": None,
        }
    total = sum(map(len, truth))
    wrong = [
        [r, c, pred[r][c], truth[r][c]]
        for r in range(len(truth))
        for c in range(len(truth[r]))
        if pred[r][c] != truth[r][c]
    ]
    return {
        "exact": not wrong,
        "shape_match": True,
        "cell_accuracy": (total - len(wrong)) / total,
        "wrong_cells": wrong,
    }


def main() -> None:
    rep_paths = sorted((EXP / "representations").glob("*.json"))
    execution_paths = sorted((EXP / "executions").glob("*.json"))
    intervention_paths = sorted((EXP / "interventions").glob("*.json"))
    intervention_execution_paths = sorted(
        (EXP / "intervention_executions").glob("*.json")
    )
    if len(rep_paths) != 10 or len(execution_paths) != 10:
        raise RuntimeError("Expected ten frozen representations and executions")
    if len(intervention_paths) != 2 or len(intervention_execution_paths) != 2:
        raise RuntimeError("Expected two frozen intervention pairs")

    frozen_hashes = {
        str(path.relative_to(EXP)): digest(path)
        for path in (
            rep_paths
            + execution_paths
            + intervention_paths
            + intervention_execution_paths
        )
    }

    answers = json.loads(
        (ROOT / "outputs/assistant_blind10_20261014/answer_key.json").read_text()
    )
    results = []
    for path in execution_paths:
        data = json.loads(path.read_text())
        task_id = data["task_id"]
        result = compare(extract_execution_grid(data), answers[task_id][0])
        results.append({"execution": path.stem, "task_id": task_id, **result})

    interventions = []
    for path in intervention_execution_paths:
        data = json.loads(path.read_text())
        result = compare(extract_execution_grid(data), answers["0d87d2a6"][0])
        interventions.append(
            {"condition": data["condition"], "execution": path.stem, **result}
        )

    by_condition = {
        item["condition"]: extract_execution_grid(
            json.loads(
                (
                    EXP
                    / "intervention_executions"
                    / f"0d87d2a6_{item['condition'].removeprefix('I-')}.json"
                ).read_text()
            )
        )
        for item in interventions
    }
    overlap = by_condition["I-overlap"]
    contact = by_condition["I-contact"]
    intervention_diff = [
        [r, c, overlap[r][c], contact[r][c]]
        for r in range(len(overlap))
        for c in range(len(overlap[r]))
        if overlap[r][c] != contact[r][c]
    ]

    total_cells = 0
    correct_cells = 0
    for item in results:
        truth = answers[item["task_id"]][0]
        cells = sum(map(len, truth))
        total_cells += cells
        correct_cells += round(item["cell_accuracy"] * cells)

    report = {
        "experiment": 31,
        "frozen_before_label_read": frozen_hashes,
        "representation_only": {
            "exact_executions": sum(item["exact"] for item in results),
            "execution_count": len(results),
            "tasks_with_both_runs_exact": sum(
                all(
                    item["exact"]
                    for item in results
                    if item["task_id"] == task_id
                )
                for task_id in TASKS
            ),
            "task_count": len(TASKS),
            "micro_cell_accuracy": correct_cells / total_cells,
            "results": results,
        },
        "interventions": {
            "results": interventions,
            "pairwise_difference_count": len(intervention_diff),
            "pairwise_differences": intervention_diff,
        },
    }
    (EXP / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

