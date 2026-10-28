"""CPU-only helpers for leakage-safe NVARC evaluation and finalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from arc_solver.ttt_eval import build_phase1_bundle, hide_test_outputs, is_valid_grid


def write_solver_challenges(
    challenges_path: Path,
    task_ids: list[str],
    solver_data_dir: Path,
) -> Path:
    """Write only label-free challenges into the solver-readable directory."""
    challenges = json.loads(challenges_path.read_text())
    public = {}
    for task_id in task_ids:
        if task_id not in challenges:
            raise KeyError(f"missing challenge for {task_id}")
        public[task_id] = hide_test_outputs(challenges[task_id])
    solver_data_dir.mkdir(parents=True, exist_ok=True)
    output = solver_data_dir / "challenges.json"
    output.write_text(json.dumps(public, separators=(",", ":")))
    return output


def load_finalizer_labels(
    challenges_path: Path,
    solutions_path: Path,
    task_ids: list[str],
    finalizer_dir: Path,
) -> dict[str, Any]:
    """Materialize and load labels after the solver has exited."""
    challenges = json.loads(challenges_path.read_text())
    solutions = json.loads(solutions_path.read_text())
    _, labels = build_phase1_bundle(challenges, solutions, task_ids)
    finalizer_dir.mkdir(parents=True, exist_ok=True)
    label_path = finalizer_dir / "labels.json"
    label_path.write_text(json.dumps(labels, separators=(",", ":")))
    return json.loads(label_path.read_text())


def _plain_grid(value: Any) -> list[list[int]] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not is_valid_grid(value):
        return None
    return [[int(cell) for cell in row] for row in value]


def finalize_top2(
    challenges: dict[str, Any],
    selected: dict[str, list[Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Finalize score_kgmon selections with valid, deterministic fallbacks."""
    submission = {}
    counts = {
        "candidate_output_items": 0,
        "identity_fallbacks": 0,
        "zero_fallbacks": 0,
    }
    for task_id in sorted(challenges):
        items = []
        for index, test_item in enumerate(challenges[task_id]["test"]):
            candidates = selected.get(f"{task_id}_{index}", [])
            valid = []
            for candidate in candidates:
                grid = _plain_grid(candidate)
                if grid is not None and grid not in valid:
                    valid.append(grid)
                if len(valid) == 2:
                    break
            counts["candidate_output_items"] += int(bool(valid))
            if valid:
                attempt_1 = valid[0]
            else:
                attempt_1 = _plain_grid(test_item["input"])
                if attempt_1 is None:
                    raise ValueError(f"invalid test input for {task_id}[{index}]")
                counts["identity_fallbacks"] += 1
            if len(valid) > 1:
                attempt_2 = valid[1]
            else:
                attempt_2 = [[0]]
                counts["zero_fallbacks"] += 1
            items.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
        submission[task_id] = items
    return submission, counts


def _cell_accuracy(candidate: list[list[int]], truth: list[list[int]]) -> float:
    if len(candidate) != len(truth) or any(
        len(candidate_row) != len(truth_row)
        for candidate_row, truth_row in zip(candidate, truth)
    ):
        return 0.0
    cells = sum(len(row) for row in truth)
    if not cells:
        return 0.0
    correct = sum(
        candidate_cell == truth_cell
        for candidate_row, truth_row in zip(candidate, truth)
        for candidate_cell, truth_cell in zip(candidate_row, truth_row)
    )
    return correct / cells


def analyze_candidate_oracles(
    decoded_results: dict[str, dict[str, dict[str, Any]]],
    labels: dict[str, list[Any]],
    scorers: dict[str, Callable[[dict[str, dict[str, Any]]], list[Any]]],
) -> dict[str, Any]:
    """Analyze rankings after labels load without modifying candidate selection.

    ``decoded_results`` is the validated sample pool produced by ``ArcDecoder``.
    Each scorer receives all valid samples, including repeated samples that carry
    voting information, and must rank every unique candidate grid.
    """
    if set(scorers) != {"score_kgmon", "score_full_probmul_3"}:
        raise ValueError("both NVARC scorers are required")

    item_reports = []
    task_reports = []
    for task_id in sorted(labels):
        task_items = []
        for test_index, raw_truth in enumerate(labels[task_id]):
            truth = _plain_grid(raw_truth)
            if truth is None:
                raise ValueError(f"invalid truth grid for {task_id}[{test_index}]")
            key = f"{task_id}_{test_index}"
            samples = decoded_results.get(key, {})
            valid_samples = {
                sample_id: sample
                for sample_id, sample in samples.items()
                if _plain_grid(sample.get("solution")) is not None
            }
            unique = []
            for sample in valid_samples.values():
                grid = _plain_grid(sample["solution"])
                assert grid is not None
                if grid not in unique:
                    unique.append(grid)

            rankings: dict[str, list[list[list[int]]]] = {}
            ranks: dict[str, int | None] = {}
            for scorer_name, scorer in scorers.items():
                ranked = []
                for candidate in scorer(valid_samples) if valid_samples else []:
                    grid = _plain_grid(candidate)
                    if grid is not None and grid not in ranked:
                        ranked.append(grid)
                # A scorer must expose every valid unique candidate.
                if len(ranked) != len(unique) or any(grid not in ranked for grid in unique):
                    raise ValueError(
                        f"{scorer_name} omitted candidates for {key}: "
                        f"ranked={len(ranked)}, unique={len(unique)}"
                    )
                rankings[scorer_name] = ranked
                ranks[scorer_name] = (
                    ranked.index(truth) + 1 if truth in ranked else None
                )

            matching_shape = [
                grid
                for grid in unique
                if len(grid) == len(truth)
                and all(
                    len(candidate_row) == len(truth_row)
                    for candidate_row, truth_row in zip(grid, truth)
                )
            ]
            accuracies = [_cell_accuracy(grid, truth) for grid in unique]
            best_index = max(
                range(len(unique)), key=lambda index: accuracies[index], default=None
            )
            best_candidate = None
            if best_index is not None:
                best_grid = unique[best_index]
                best_candidate = {
                    "cell_accuracy": accuracies[best_index],
                    "grid": best_grid,
                    "score_kgmon_rank": rankings["score_kgmon"].index(best_grid) + 1,
                    "score_full_probmul_3_rank": (
                        rankings["score_full_probmul_3"].index(best_grid) + 1
                    ),
                }
            truth_present = truth in unique
            item = {
                "task_id": task_id,
                "test_index": test_index,
                "unique_candidate_count": len(unique),
                "matching_shape_candidate_count": len(matching_shape),
                "truth_present": truth_present,
                "score_kgmon_truth_rank": ranks["score_kgmon"],
                "score_full_probmul_3_truth_rank": ranks[
                    "score_full_probmul_3"
                ],
                "score_kgmon_top1": ranks["score_kgmon"] == 1,
                "score_kgmon_top2": (
                    ranks["score_kgmon"] is not None
                    and ranks["score_kgmon"] <= 2
                ),
                "score_full_probmul_3_top1": (
                    ranks["score_full_probmul_3"] == 1
                ),
                "score_full_probmul_3_top2": (
                    ranks["score_full_probmul_3"] is not None
                    and ranks["score_full_probmul_3"] <= 2
                ),
                "best_candidate_cell_accuracy": (
                    accuracies[best_index] if best_index is not None else 0.0
                ),
                "best_candidate": best_candidate,
            }
            task_items.append(item)
            item_reports.append(item)

        task_reports.append(
            {
                "task_id": task_id,
                "item_count": len(task_items),
                "oracle_exact": all(item["truth_present"] for item in task_items),
                "score_kgmon_top2_exact": all(
                    item["score_kgmon_top2"] for item in task_items
                ),
                "score_full_probmul_3_top2_exact": all(
                    item["score_full_probmul_3_top2"] for item in task_items
                ),
                "items": task_items,
            }
        )

    item_count = len(item_reports)
    truth_present = sum(item["truth_present"] for item in item_reports)
    aggregate = {
        "item_count": item_count,
        "task_count": len(task_reports),
        "truth_present_items": truth_present,
        "generation_miss_items": item_count - truth_present,
        "score_kgmon_top1_items": sum(
            item["score_kgmon_top1"] for item in item_reports
        ),
        "score_kgmon_top2_items": sum(
            item["score_kgmon_top2"] for item in item_reports
        ),
        "score_kgmon_selection_miss_items": sum(
            item["truth_present"] and not item["score_kgmon_top2"]
            for item in item_reports
        ),
        "score_full_probmul_3_top1_items": sum(
            item["score_full_probmul_3_top1"] for item in item_reports
        ),
        "score_full_probmul_3_top2_items": sum(
            item["score_full_probmul_3_top2"] for item in item_reports
        ),
        "score_full_probmul_3_selection_miss_items": sum(
            item["truth_present"] and not item["score_full_probmul_3_top2"]
            for item in item_reports
        ),
        "oracle_exact_tasks": sum(task["oracle_exact"] for task in task_reports),
        "score_kgmon_top2_exact_tasks": sum(
            task["score_kgmon_top2_exact"] for task in task_reports
        ),
        "score_full_probmul_3_top2_exact_tasks": sum(
            task["score_full_probmul_3_top2_exact"] for task in task_reports
        ),
    }
    return {"aggregate": aggregate, "tasks": task_reports}


_TRAINING_RE = re.compile(
    r"training stats for puzzle (?P<task>[0-9a-f]+):.*?"
    r"global_step=(?P<steps>\d+).*?"
    r"training_loss=(?P<loss>[-+0-9.eE]+)"
)


def parse_training_stats(log_text: str) -> dict[str, dict[str, Any]]:
    """Extract per-task TTT steps and final loss from upstream console logs."""
    return {
        match.group("task"): {
            "ttt_steps": int(match.group("steps")),
            "ttt_loss": float(match.group("loss")),
        }
        for match in _TRAINING_RE.finditer(log_text)
    }
