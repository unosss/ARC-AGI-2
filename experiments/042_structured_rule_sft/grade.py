"""Open sealed labels only after both Experiment 042 arms are complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_by_id(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{source} contains an invalid ID")
        if item_id in result:
            raise ValueError(f"{source} contains duplicate ID {item_id}")
        result[item_id] = row
    return result


def verify_candidates(
    directory: Path, expected_arm: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completion = json.loads((directory / "inference_completed.json").read_text())
    path = directory / "candidates.jsonl"
    if completion["arm"] != expected_arm or not completion["complete"]:
        raise ValueError(f"{expected_arm} inference is not complete")
    if completion["candidate_sha256"] != file_sha256(path):
        raise ValueError(f"{expected_arm} candidate hash mismatch")
    rows = load_jsonl(path)
    if len(rows) != completion["candidate_count"]:
        raise ValueError(f"{expected_arm} candidate count mismatch")
    unique_by_id(rows, expected_arm)
    if any(row.get("arm") != expected_arm for row in rows):
        raise ValueError(f"{expected_arm} candidate payload has the wrong arm")
    return rows, completion


def cell_accuracy(prediction: Any, truth: list[list[int]]) -> float:
    if not isinstance(prediction, list) or len(prediction) != len(truth):
        return 0.0
    if any(
        not isinstance(row, list) or len(row) != len(truth[0])
        for row in prediction
    ):
        return 0.0
    total = len(truth) * len(truth[0])
    correct = sum(
        predicted == expected
        for predicted_row, expected_row in zip(prediction, truth)
        for predicted, expected in zip(predicted_row, expected_row)
    )
    return correct / total


def cell_counts(prediction: Any, truth: list[list[int]]) -> tuple[int, int]:
    total = len(truth) * len(truth[0])
    if not isinstance(prediction, list) or len(prediction) != len(truth):
        return 0, total
    if any(
        not isinstance(row, list) or len(row) != len(truth[0])
        for row in prediction
    ):
        return 0, total
    correct = sum(
        predicted == expected
        for predicted_row, expected_row in zip(prediction, truth)
        for predicted, expected in zip(predicted_row, expected_row)
    )
    return correct, total


def is_valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    return (
        1 <= width <= 30
        and all(len(row) == width for row in value)
        and all(
            isinstance(cell, int)
            and not isinstance(cell, bool)
            and 0 <= cell <= 9
            for row in value
            for cell in row
        )
    )


def paired_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    low = min(wins, losses)
    tail = sum(math.comb(discordant, index) for index in range(low + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_interval(differences: list[float], seed: int = 420044) -> list[float]:
    generator = random.Random(seed)
    samples = []
    for _ in range(10_000):
        samples.append(
            mean(generator.choice(differences) for _ in range(len(differences)))
        )
    samples.sort()
    return [samples[249], samples[9749]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing grade report")

    # Candidate barriers are verified before the labels file is opened.
    direct_rows, direct_completion = verify_candidates(args.direct_dir, "direct")
    program_rows, program_completion = verify_candidates(args.program_dir, "program")
    for key in ("budget", "config_sha256", "challenges_sha256", "resolved_model"):
        if direct_completion.get(key) != program_completion.get(key):
            raise ValueError(f"paired arm provenance mismatch: {key}")
    selection = None
    if args.phase == "test":
        if args.selection_manifest is None:
            raise ValueError("test grading requires a frozen validation selection")
        selection = json.loads(args.selection_manifest.read_text())
        if selection.get("selected_budget") != direct_completion["budget"]:
            raise ValueError("test candidate budget differs from validation selection")
        if selection.get("config_sha256") != direct_completion["config_sha256"]:
            raise ValueError("selection and test config differ")
    elif args.selection_manifest is not None:
        raise ValueError("selection manifest is only valid for test grading")
    labels = load_jsonl(args.labels)
    direct = unique_by_id(direct_rows, "direct")
    program = unique_by_id(program_rows, "program")
    truth = unique_by_id(labels, "labels")
    if set(direct) != set(program) or set(direct) != set(truth):
        raise ValueError("arm and label IDs differ")

    items = []
    for item_id in sorted(truth):
        expected_challenge_hash = truth[item_id].get("challenge_sha256")
        if (
            not expected_challenge_hash
            or direct[item_id].get("challenge_sha256") != expected_challenge_hash
            or program[item_id].get("challenge_sha256") != expected_challenge_hash
        ):
            raise ValueError(f"challenge/label binding mismatch for {item_id}")
        expected = truth[item_id]["query_output"]
        direct_exact = direct[item_id]["prediction"] == expected
        program_exact = program[item_id]["prediction"] == expected
        items.append(
            {
                "id": item_id,
                "direct_exact": direct_exact,
                "program_exact": program_exact,
                "direct_cell_accuracy": cell_accuracy(
                    direct[item_id]["prediction"], expected
                ),
                "program_cell_accuracy": cell_accuracy(
                    program[item_id]["prediction"], expected
                ),
                "program_demo_replay_exact": bool(
                    program[item_id]["demo_replay_exact"]
                ),
                "direct_valid_grid": is_valid_grid(direct[item_id]["prediction"]),
                "program_valid_grid": is_valid_grid(program[item_id]["prediction"]),
                "program_parsed": program[item_id].get("program") is not None,
                "program_validated": bool(program[item_id].get("program_validated")),
                "direct_elapsed_seconds": float(direct[item_id]["elapsed_seconds"]),
                "program_elapsed_seconds": float(program[item_id]["elapsed_seconds"]),
                "direct_generated_tokens": int(direct[item_id]["generated_tokens"]),
                "program_generated_tokens": int(program[item_id]["generated_tokens"]),
            }
        )
    wins = sum(row["program_exact"] and not row["direct_exact"] for row in items)
    losses = sum(row["direct_exact"] and not row["program_exact"] for row in items)
    differences = [
        float(row["program_exact"]) - float(row["direct_exact"]) for row in items
    ]
    replay_items = [row for row in items if row["program_demo_replay_exact"]]
    direct_cells = [
        cell_counts(direct[item_id]["prediction"], truth[item_id]["query_output"])
        for item_id in sorted(truth)
    ]
    program_cells = [
        cell_counts(program[item_id]["prediction"], truth[item_id]["query_output"])
        for item_id in sorted(truth)
    ]
    interval = bootstrap_interval(differences)
    p_value = paired_p_value(wins, losses)
    report = {
        "schema_version": 1,
        "experiment": 42,
        "phase": args.phase,
        "budget": direct_completion["budget"],
        "config_sha256": direct_completion["config_sha256"],
        "challenges_sha256": direct_completion["challenges_sha256"],
        "selection_manifest_sha256": (
            file_sha256(args.selection_manifest) if args.selection_manifest else None
        ),
        "count": len(items),
        "direct_exact": sum(row["direct_exact"] for row in items),
        "program_exact": sum(row["program_exact"] for row in items),
        "program_minus_direct": mean(differences),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": len(items) - wins - losses,
        "paired_exact_p_value": p_value,
        "bootstrap_95_interval": interval,
        "confirmatory_gate_passed": (
            mean(differences) > 0 and p_value <= 0.05 and interval[0] > 0
        ),
        "direct_macro_cell_accuracy": mean(
            row["direct_cell_accuracy"] for row in items
        ),
        "program_macro_cell_accuracy": mean(
            row["program_cell_accuracy"] for row in items
        ),
        "direct_micro_cell_accuracy": (
            sum(value[0] for value in direct_cells)
            / sum(value[1] for value in direct_cells)
        ),
        "program_micro_cell_accuracy": (
            sum(value[0] for value in program_cells)
            / sum(value[1] for value in program_cells)
        ),
        "direct_valid_grid_rate": mean(row["direct_valid_grid"] for row in items),
        "program_valid_grid_rate": mean(row["program_valid_grid"] for row in items),
        "program_parse_rate": mean(row["program_parsed"] for row in items),
        "program_validation_rate": mean(
            row["program_validated"] for row in items
        ),
        "program_demo_replay_exact": len(replay_items),
        "program_query_exact_given_replay": (
            sum(row["program_exact"] for row in replay_items) / len(replay_items)
            if replay_items
            else None
        ),
        "direct_mean_elapsed_seconds": mean(
            row["direct_elapsed_seconds"] for row in items
        ),
        "program_mean_elapsed_seconds": mean(
            row["program_elapsed_seconds"] for row in items
        ),
        "direct_mean_generated_tokens": mean(
            row["direct_generated_tokens"] for row in items
        ),
        "program_mean_generated_tokens": mean(
            row["program_generated_tokens"] for row in items
        ),
        "direct_peak_memory_bytes": direct_completion["peak_memory_bytes"],
        "program_peak_memory_bytes": program_completion["peak_memory_bytes"],
        "direct_candidates_sha256": file_sha256(
            args.direct_dir / "candidates.jsonl"
        ),
        "program_candidates_sha256": file_sha256(
            args.program_dir / "candidates.jsonl"
        ),
        "labels_sha256": file_sha256(args.labels),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
