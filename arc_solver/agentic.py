"""Verifier feedback helpers for iterative ARC program synthesis."""

from __future__ import annotations

import json
from typing import Any


Grid = list[list[int]]


def grid_diagnostics(
    prediction: Grid | None,
    truth: Grid,
    error: str | None = None,
    coordinate_limit: int = 40,
) -> dict[str, Any]:
    if prediction is None:
        return {"error": error or "no output"}
    predicted_shape = [
        len(prediction),
        len(prediction[0]) if prediction and prediction[0] else 0,
    ]
    truth_shape = [len(truth), len(truth[0]) if truth and truth[0] else 0]
    mismatches = []
    missing = extra = wrong_color = 0
    max_rows = max(len(prediction), len(truth))
    for row in range(max_rows):
        predicted_row = prediction[row] if row < len(prediction) else []
        truth_row = truth[row] if row < len(truth) else []
        for column in range(max(len(predicted_row), len(truth_row))):
            predicted = predicted_row[column] if column < len(predicted_row) else None
            expected = truth_row[column] if column < len(truth_row) else None
            if predicted == expected:
                continue
            if expected not in (None, 0) and predicted in (None, 0):
                kind = "missing"
                missing += 1
            elif predicted not in (None, 0) and expected in (None, 0):
                kind = "extra"
                extra += 1
            else:
                kind = "wrong_color"
                wrong_color += 1
            if len(mismatches) < coordinate_limit:
                mismatches.append(
                    {
                        "row": row,
                        "col": column,
                        "got": predicted,
                        "expected": expected,
                        "kind": kind,
                    }
                )
    mismatch_count = missing + extra + wrong_color
    return {
        "exact": mismatch_count == 0,
        "predicted_shape": predicted_shape,
        "expected_shape": truth_shape,
        "mismatch_count": mismatch_count,
        "missing": missing,
        "extra": extra,
        "wrong_color": wrong_color,
        "mismatches": mismatches,
        "coordinates_truncated": mismatch_count > len(mismatches),
    }


def feedback_payload(evaluations: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    """Keep feedback compact while preserving the best and distinct failures."""
    ranked = sorted(
        evaluations,
        key=lambda item: (
            not item.get("valid", False),
            -item.get("train_matches", 0),
            -item.get("cell_accuracy", 0.0),
            item.get("complexity")
            if item.get("complexity") is not None
            else 10**9,
        ),
    )
    payload = []
    for item in ranked[:limit]:
        payload.append(
            {
                "program": item.get("source"),
                "valid": item.get("valid", False),
                "error": item.get("error"),
                "train_matches": item.get("train_matches", 0),
                "train_total": item.get("train_total", 0),
                "cell_accuracy": round(item.get("cell_accuracy", 0.0), 6),
                "train_diagnostics": item.get("train_diagnostics", []),
            }
        )
    return payload


def build_repair_instruction(
    evaluations: list[dict[str, Any]],
    round_index: int,
    force_reparse: bool,
) -> str:
    feedback = json.dumps(
        feedback_payload(evaluations), ensure_ascii=False, separators=(",", ":")
    )
    strategy = (
        "The previous interpretation stopped improving. Discard its central "
        "assumption and infer a materially different scene decomposition and rule."
        if force_reparse
        else "Diagnose the failed assumption from the counterexamples and repair it."
    )
    return f"""This is repair round {round_index}. {strategy}
The verifier executed prior programs on every training pair. Its exact reports are:
{feedback}
All training outputs were already provided, so using these differences is allowed.
Return a revised general solve(grid) program, not a patch tied to coordinates.
Obey the sandbox restrictions. Return only the requested JSON object."""
