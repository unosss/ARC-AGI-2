"""Grade frozen candidate rankings after hashing all blind artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parent
KS = [1, 2, 3, 4, 5, 8, 10, 16]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_grid(grid: object) -> bool:
    return (
        isinstance(grid, list)
        and bool(grid)
        and all(isinstance(row, list) and bool(row) for row in grid)
        and len({len(row) for row in grid}) == 1
        and all(
            isinstance(value, int) and 0 <= value <= 9
            for row in grid
            for value in row
        )
    )


def normalize_test(value: object) -> list[list[list[int]]] | None:
    if not isinstance(value, list) or not value:
        return None
    if valid_grid(value):
        return [value]
    if all(valid_grid(item) for item in value):
        return value
    if all(isinstance(item, dict) and valid_grid(item.get("output")) for item in value):
        return [item["output"] for item in value]
    return None


def test_exact(prediction: object, truth: list[list[list[int]]]) -> bool:
    normalized = normalize_test(prediction)
    return normalized == truth


def cell_stats(prediction: object, truth: list[list[list[int]]]) -> tuple[int, int]:
    normalized = normalize_test(prediction)
    total = sum(len(row) for grid in truth for row in grid)
    if normalized is None or len(normalized) != len(truth):
        return 0, total
    correct = 0
    for pred_grid, truth_grid in zip(normalized, truth):
        if (
            len(pred_grid) != len(truth_grid)
            or any(len(a) != len(b) for a, b in zip(pred_grid, truth_grid))
        ):
            continue
        correct += sum(
            pred == target
            for pred_row, truth_row in zip(pred_grid, truth_grid)
            for pred, target in zip(pred_row, truth_row)
        )
    return correct, total


def train_exact(candidate: dict, challenge: dict) -> bool:
    predicted = candidate.get("predicted_train_outputs")
    expected = [pair["output"] for pair in challenge["train"]]
    return predicted == expected


def main() -> None:
    task_ids = json.loads((EXP / "task_ids.json").read_text())
    ranked_paths = [EXP / "ranked" / f"{task_id}.json" for task_id in task_ids]
    generator_paths = [
        EXP / "generators" / f"{task_id}_{run}.json"
        for task_id in task_ids
        for run in ("r1", "r2")
    ]
    if not all(path.exists() for path in ranked_paths):
        raise RuntimeError("All twenty ranked artifacts must exist before grading")
    if not all(path.exists() for path in generator_paths):
        raise RuntimeError("All forty generator artifacts must exist before grading")

    frozen_hashes = {
        str(path.relative_to(EXP)): digest(path)
        for path in generator_paths + ranked_paths
    }

    answers = json.loads((EXP / "private_answer_key.json").read_text())
    task_results = []
    top1_correct_cells = 0
    top1_total_cells = 0
    run_oracle = {"r1": 0, "r2": 0}

    for task_id in task_ids:
        truth = answers[task_id]
        challenge = json.loads((EXP / "challenges" / f"{task_id}.json").read_text())
        ranked = json.loads((EXP / "ranked" / f"{task_id}.json").read_text())
        ranking = sorted(ranked["ranking"], key=lambda item: item["rank"])

        exact_flags = [test_exact(item.get("test"), truth) for item in ranking]
        exact_ranks = [index + 1 for index, exact in enumerate(exact_flags) if exact]
        if ranking:
            correct, total = cell_stats(ranking[0].get("test"), truth)
        else:
            correct = 0
            total = sum(len(row) for grid in truth for row in grid)
        top1_correct_cells += correct
        top1_total_cells += total

        raw_run_results = {}
        for run in ("r1", "r2"):
            data = json.loads(
                (EXP / "generators" / f"{task_id}_{run}.json").read_text()
            )
            eligible = [
                candidate
                for candidate in data["candidates"]
                if train_exact(candidate, challenge)
            ]
            exact_count = sum(
                test_exact(candidate.get("test"), truth) for candidate in eligible
            )
            raw_run_results[run] = {
                "verified_train_exact_count": len(eligible),
                "test_exact_count": exact_count,
                "oracle": exact_count > 0,
            }
            run_oracle[run] += exact_count > 0

        if exact_ranks:
            classification = (
                "solved_top1" if exact_ranks[0] == 1 else "selection_error"
            )
        elif ranking:
            classification = "generation_error"
        else:
            classification = "invalid_generation"

        task_results.append(
            {
                "task_id": task_id,
                "raw_run_results": raw_run_results,
                "raw_verified_train_exact_count": sum(
                    item["verified_train_exact_count"]
                    for item in raw_run_results.values()
                ),
                "unique_candidate_count": len(ranking),
                "top1_exact": bool(exact_flags and exact_flags[0]),
                "top1_cell_accuracy": correct / total if total else 0.0,
                "oracle_exact": bool(exact_ranks),
                "exact_candidate_count": len(exact_ranks),
                "first_exact_rank": exact_ranks[0] if exact_ranks else None,
                "exact_ranks": exact_ranks,
                "classification": classification,
            }
        )

    oracle_at_k = {
        str(k): sum(
            item["first_exact_rank"] is not None and item["first_exact_rank"] <= k
            for item in task_results
        )
        for k in KS
    }
    classifications = {
        name: sum(item["classification"] == name for item in task_results)
        for name in (
            "solved_top1",
            "selection_error",
            "generation_error",
            "invalid_generation",
        )
    }
    unique_counts = [item["unique_candidate_count"] for item in task_results]
    report = {
        "experiment": 33,
        "task_count": len(task_ids),
        "frozen_before_label_read": frozen_hashes,
        "top1_exact": sum(item["top1_exact"] for item in task_results),
        "top1_cell_accuracy": top1_correct_cells / top1_total_cells,
        "oracle_full": sum(item["oracle_exact"] for item in task_results),
        "oracle_at_k": oracle_at_k,
        "single_run_oracle": run_oracle,
        "classifications": classifications,
        "candidate_counts": {
            "total_unique": sum(unique_counts),
            "min": min(unique_counts),
            "max": max(unique_counts),
            "mean": sum(unique_counts) / len(unique_counts),
        },
        "tasks": task_results,
    }
    (EXP / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

