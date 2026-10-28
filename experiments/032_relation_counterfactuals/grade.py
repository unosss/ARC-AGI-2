"""Fail-closed grading for frozen Experiment 032 executions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parent
CONDITIONS = [
    "incidence_overlap",
    "incidence_contact",
    "reflection_center_strict",
    "reflection_endpoint_interval",
    "shape_translation",
    "shape_d4",
]
PAIRS = [
    ("incidence_overlap", "incidence_contact"),
    ("reflection_center_strict", "reflection_endpoint_interval"),
    ("shape_translation", "shape_d4"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_grid(pred: list[list[int]], truth: list[list[int]]) -> dict:
    shape_match = (
        len(pred) == len(truth)
        and all(len(a) == len(b) for a, b in zip(pred, truth))
    )
    if not shape_match:
        return {
            "exact": False,
            "shape_match": False,
            "correct_cells": 0,
            "total_cells": sum(map(len, truth)),
            "wrong_cells": None,
        }
    wrong = [
        [r, c, pred[r][c], truth[r][c]]
        for r in range(len(truth))
        for c in range(len(truth[r]))
        if pred[r][c] != truth[r][c]
    ]
    total = sum(map(len, truth))
    return {
        "exact": not wrong,
        "shape_match": True,
        "correct_cells": total - len(wrong),
        "total_cells": total,
        "wrong_cells": wrong,
    }


def grid_differences(a: list[list[int]], b: list[list[int]]) -> set[tuple[int, int]]:
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("Cannot compare differently shaped paired grids")
    return {
        (r, c)
        for r in range(len(a))
        for c in range(len(a[r]))
        if a[r][c] != b[r][c]
    }


def extract_test_grids(data: dict) -> list[list[list[int]]]:
    grids = data["test"]
    if grids and isinstance(grids[0], dict):
        return [item["output"] for item in grids]
    return grids


def main() -> None:
    artifact_paths = sorted((EXP / "artifacts").glob("*.json"))
    execution_paths = sorted((EXP / "executions").glob("*.json"))
    if [path.stem for path in artifact_paths] != sorted(CONDITIONS):
        raise RuntimeError("Artifact set is incomplete or unexpected")
    if [path.stem for path in execution_paths] != sorted(CONDITIONS):
        raise RuntimeError("Execution set is incomplete or unexpected")

    frozen_hashes = {
        str(path.relative_to(EXP)): digest(path)
        for path in artifact_paths + execution_paths
    }

    answers = json.loads((EXP / "private_answer_key.json").read_text())
    predictions = {
        path.stem: extract_test_grids(json.loads(path.read_text()))
        for path in execution_paths
    }

    results = {}
    total_correct = 0
    total_cells = 0
    for condition in CONDITIONS:
        predicted_cases = predictions[condition]
        truth_cases = answers[condition]
        if len(predicted_cases) != len(truth_cases):
            raise RuntimeError(f"Wrong case count for {condition}")
        cases = [
            compare_grid(pred, truth)
            for pred, truth in zip(predicted_cases, truth_cases)
        ]
        results[condition] = {
            "exact": all(case["exact"] for case in cases),
            "exact_cases": sum(case["exact"] for case in cases),
            "case_count": len(cases),
            "cases": cases,
        }
        total_correct += sum(case["correct_cells"] for case in cases)
        total_cells += sum(case["total_cells"] for case in cases)

    paired_interventions = []
    for left, right in PAIRS:
        predicted_cases_left = predictions[left]
        predicted_cases_right = predictions[right]
        truth_cases_left = answers[left]
        truth_cases_right = answers[right]
        case_results = []
        for index, (pred_l, pred_r, truth_l, truth_r) in enumerate(
            zip(
                predicted_cases_left,
                predicted_cases_right,
                truth_cases_left,
                truth_cases_right,
            )
        ):
            pred_diff = grid_differences(pred_l, pred_r)
            truth_diff = grid_differences(truth_l, truth_r)
            case_results.append(
                {
                    "case": index,
                    "predicted_difference_count": len(pred_diff),
                    "expected_difference_count": len(truth_diff),
                    "difference_support_exact": pred_diff == truth_diff,
                }
            )
        paired_interventions.append(
            {
                "left": left,
                "right": right,
                "all_difference_support_exact": all(
                    case["difference_support_exact"] for case in case_results
                ),
                "cases": case_results,
            }
        )

    report = {
        "experiment": 32,
        "frozen_before_label_read": frozen_hashes,
        "exact_conditions": sum(item["exact"] for item in results.values()),
        "condition_count": len(CONDITIONS),
        "exact_cases": sum(item["exact_cases"] for item in results.values()),
        "case_count": sum(item["case_count"] for item in results.values()),
        "micro_cell_accuracy": total_correct / total_cells,
        "results": results,
        "paired_interventions": paired_interventions,
    }
    (EXP / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

