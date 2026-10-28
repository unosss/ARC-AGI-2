"""CPU-only construction and audit helpers for Experiment 018."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from arc_solver.ttt_eval import is_valid_grid


TASK_ID = "0934a4d8"
TEST_INDEX = 0
SOURCE_BBOX = (0, 3, 14, 23)  # half-open: rows 0..2, columns 14..22
PREFIX_ROWS = (0, 1, 2, 3, 4, 5, 8)
PREFIX_8_ROWS_PLUS_1_CELL = "prefix_8_rows_plus_1_cell"
DISTRACTOR_BBOXES = (
    (0, 3, 0, 9),
    (0, 3, 7, 16),
    SOURCE_BBOX,
    (0, 3, 21, 30),
    (9, 12, 14, 23),
)


def _copy_grid(grid: Any) -> list[list[int]]:
    if not is_valid_grid(grid):
        raise ValueError("expected a valid ARC grid")
    return copy.deepcopy(grid)


def build_training_bundle(
    challenges: dict[str, Any], task_id: str = TASK_ID
) -> dict[str, Any]:
    """Return the one-task solver bundle without reading or retaining test labels."""
    if task_id not in challenges:
        raise KeyError(f"missing task {task_id}")
    task = challenges[task_id]
    train = task.get("train")
    test = task.get("test")
    if not isinstance(train, list) or not train:
        raise ValueError("training demonstrations are required")
    if not isinstance(test, list) or TEST_INDEX >= len(test):
        raise ValueError("test input is required")
    clean_train = []
    for pair in train:
        clean_train.append(
            {
                "input": _copy_grid(pair.get("input")),
                "output": _copy_grid(pair.get("output")),
            }
        )
    clean_test = [{"input": _copy_grid(item.get("input"))} for item in test]
    return {task_id: {"train": clean_train, "test": clean_test}}


def assert_oracle_source(
    test_input: list[list[int]],
    truth: list[list[int]],
    bbox: tuple[int, int, int, int] = SOURCE_BBOX,
) -> list[list[int]]:
    """Extract the pinned source and abort unless its transpose is exactly truth."""
    grid = _copy_grid(test_input)
    expected = _copy_grid(truth)
    r0, r1, c0, c1 = bbox
    crop = [row[c0:c1] for row in grid[r0:r1]]
    if (len(crop), len(crop[0]) if crop else 0) != (3, 9):
        raise AssertionError("pinned source bbox is not 3x9")
    transposed = [list(row) for row in zip(*crop)]
    if transposed != expected:
        raise AssertionError("pinned source transpose does not equal canonical truth")
    return crop


def _mask_to_bbox(
    grid: list[list[int]], bbox: tuple[int, int, int, int]
) -> list[list[int]]:
    r0, r1, c0, c1 = bbox
    return [
        [value if r0 <= row < r1 and c0 <= col < c1 else 0 for col, value in enumerate(values)]
        for row, values in enumerate(grid)
    ]


def transpose_probe_demos() -> list[dict[str, list[list[int]]]]:
    """Grid-only synthetic operation hints; never part of the LoRA train set."""
    inputs = [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 0, 2], [3, 4, 5]],
    ]
    return [
        {"input": grid, "output": [list(row) for row in zip(*grid)]}
        for grid in inputs
    ]


def build_oracle_conditions(
    solver_task: dict[str, Any],
    truth: list[list[int]],
) -> dict[str, Any]:
    """Build label-tagged interventions after ordinary generation has completed."""
    test_input = _copy_grid(solver_task["test"][TEST_INDEX]["input"])
    source = assert_oracle_source(test_input, truth)
    base_train = copy.deepcopy(solver_task["train"])
    transpose_train = base_train + transpose_probe_demos()
    conditions: dict[str, Any] = {
        "oracle_source_crop": {
            "causal_question": "localization",
            "oracle": True,
            "ood": True,
            "train": base_train,
            "test_input": source,
        },
        "oracle_source_mask": {
            "causal_question": "localization",
            "oracle": True,
            "ood": False,
            "train": base_train,
            "test_input": _mask_to_bbox(test_input, SOURCE_BBOX),
        },
        "transpose_demonstration_probe": {
            "causal_question": "execution",
            "oracle": True,
            "oracle_operation_hint": True,
            "fair_solver_condition": False,
            "ood": True,
            "train": copy.deepcopy(transpose_train),
            "test_input": source,
        },
        "transpose_hint_full": {
            "causal_question": "operation_given_localization_unknown",
            "oracle": True,
            "oracle_operation_hint": True,
            "oracle_localization_hint": False,
            "fair_solver_condition": False,
            "ood": False,
            "train": copy.deepcopy(transpose_train),
            "test_input": test_input,
        },
        "transpose_hint_mask": {
            "causal_question": "operation_and_localized_source",
            "oracle": True,
            "oracle_operation_hint": True,
            "oracle_localization_hint": True,
            "fair_solver_condition": False,
            "ood": False,
            "train": copy.deepcopy(transpose_train),
            "test_input": _mask_to_bbox(test_input, SOURCE_BBOX),
        },
    }
    correspondence = []
    for index, bbox in enumerate(DISTRACTOR_BBOXES):
        r0, r1, c0, c1 = bbox
        crop = [row[c0:c1] for row in test_input[r0:r1]]
        if (len(crop), len(crop[0]) if crop else 0) != (3, 9):
            raise AssertionError(f"invalid distractor bbox {bbox}")
        correspondence.append(
            {
                "condition_id": f"crop_{index}",
                "bbox": list(bbox),
                "is_true_source": bbox == SOURCE_BBOX,
                "test_input": crop,
            }
        )
        conditions[f"transpose_hint_crop_{index}"] = {
            "causal_question": "hinted_correspondence",
            "oracle": True,
            "oracle_operation_hint": True,
            "oracle_localization_hint": True,
            "fair_solver_condition": False,
            "ood": True,
            "correspondence_crop_index": index,
            "bbox": list(bbox),
            "is_true_source": bbox == SOURCE_BBOX,
            "train": copy.deepcopy(transpose_train),
            "test_input": crop,
        }
    if sum(item["is_true_source"] for item in correspondence) != 1:
        raise AssertionError("correspondence set must contain exactly one true source")
    return {
        "schema_version": 2,
        "task_id": TASK_ID,
        "test_index": TEST_INDEX,
        "source_bbox": list(SOURCE_BBOX),
        "truth": _copy_grid(truth),
        "conditions": conditions,
        "correspondence_crops": correspondence,
        "prefix_rows": list(PREFIX_ROWS),
    }


def canonical_reply_tokens(
    grid: list[list[int]], *, eos_token_id: int = 15, newline_token_id: int = 10
) -> tuple[list[int], list[dict[str, Any]]]:
    """Map the checkpoint's one-token ARC reply vocabulary to cells and separators."""
    checked = _copy_grid(grid)
    tokens: list[int] = []
    mapping: list[dict[str, Any]] = []
    for row_index, row in enumerate(checked):
        for column_index, value in enumerate(row):
            if not 0 <= value <= 9:
                raise ValueError("ARC cell must be a single digit")
            tokens.append(value)
            mapping.append(
                {"kind": "cell", "row": row_index, "column": column_index}
            )
        if row_index + 1 < len(checked):
            tokens.append(newline_token_id)
            mapping.append({"kind": "newline", "after_row": row_index})
    tokens.append(eos_token_id)
    mapping.append({"kind": "eos"})
    return tokens, mapping


def forced_prefix_tokens(
    truth: list[list[int]],
    prefix_rows: int,
    *,
    newline_token_id: int = 10,
) -> list[int]:
    """Return exact row-prefix tokens, including the boundary newline."""
    checked = _copy_grid(truth)
    if prefix_rows < 0 or prefix_rows > len(checked):
        raise ValueError("prefix_rows outside truth")
    prefix: list[int] = []
    for row_index in range(prefix_rows):
        prefix.extend(checked[row_index])
        if row_index + 1 < len(checked):
            prefix.append(newline_token_id)
    return prefix


def forced_prefix_8_rows_plus_1_cell(
    truth: list[list[int]], *, newline_token_id: int = 10
) -> list[int]:
    """Force eight complete rows and exactly the first cell of truth row eight."""
    checked = _copy_grid(truth)
    if len(checked) <= 8 or not checked[8]:
        raise ValueError("truth must contain row 8 with at least one cell")
    return forced_prefix_tokens(
        checked, 8, newline_token_id=newline_token_id
    ) + [checked[8][0]]


def summarize_token_nll(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-token teacher-forcing records without discarding cell detail."""
    cells = [record for record in records if record["kind"] == "cell"]
    rows = []
    for row_index in sorted({record["row"] for record in cells}):
        selected = [record for record in cells if record["row"] == row_index]
        rows.append(
            {
                "row": row_index,
                "cell_count": len(selected),
                "sum_nll": sum(record["nll"] for record in selected),
                "mean_nll": sum(record["nll"] for record in selected) / len(selected),
                "mean_correct_probability": sum(
                    record["correct_probability"] for record in selected
                )
                / len(selected),
                "mean_correct_rank": sum(
                    record["correct_rank"] for record in selected
                )
                / len(selected),
            }
        )
    return {
        "token_count": len(records),
        "cell_count": len(cells),
        "sum_nll": sum(record["nll"] for record in records),
        "per_token_nll": sum(record["nll"] for record in records) / len(records),
        "cell_mean_nll": sum(record["nll"] for record in cells) / len(cells),
        "rows": rows,
        "tokens": records,
    }


def validate_lifecycle(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    names = [event["event"] for event in events]
    required = [
        "ordinary_training_completed",
        "ordinary_generation_completed",
        "ordinary_worker_exited",
        "labels_opened",
        "oracle_payload_written",
        "oracle_training_completed",
        "oracle_payload_opened",
        "oracle_diagnostics_completed",
        "oracle_worker_exited",
    ]
    positions = []
    for name in required:
        if names.count(name) != 1:
            raise AssertionError(f"lifecycle event {name!r} must occur exactly once")
        positions.append(names.index(name))
    if positions != sorted(positions):
        raise AssertionError(f"invalid lifecycle order: {names}")
    return {
        "valid": True,
        "events": names,
        "ordinary_labels_available": False,
        "oracle_payload_opened_after_training": True,
        "post_label_optimizer_steps": 0,
    }


def aggregate_report(
    config: dict[str, Any],
    ordinary: dict[str, Any],
    oracle: dict[str, Any],
    lifecycle: list[dict[str, Any]],
) -> dict[str, Any]:
    audit = validate_lifecycle(lifecycle)
    crops = oracle["correspondence"]["crops"]
    true_results = [item for item in crops if item["is_true_source"]]
    distractors = [item for item in crops if not item["is_true_source"]]
    prefix = oracle["forced_prefix"]["conditions"]
    exact_prefixes = [
        item["prefix_rows"]
        for item in prefix
        if item.get("prefix_rows") is not None and item["exact_completion"]
    ]
    hinted_crops = oracle["hinted_correspondence"]["crops"]
    true_hinted = [item for item in hinted_crops if item["is_true_source"]]
    distractor_hinted = [item for item in hinted_crops if not item["is_true_source"]]
    return {
        "schema_version": 2,
        "run_version": 2,
        "experiment": "018_qwen_stage_diagnosis",
        "status": "gpu_run_complete",
        "task_id": TASK_ID,
        "test_index": TEST_INDEX,
        "config": config,
        "label_lifecycle": audit,
        "training_runs": {
            "ordinary": ordinary["training"],
            "oracle_replay_before_payload_open": oracle["training"],
        },
        "causal_questions": {
            "execution": oracle["execution"],
            "localization": {
                "baseline": ordinary["baseline"],
                "baseline_exp015_comparable": ordinary["baseline_exp015_comparable"],
                "oracle_crop": oracle["conditions"]["oracle_source_crop"],
                "oracle_mask": oracle["conditions"]["oracle_source_mask"],
                "transpose_hint_full": oracle["conditions"]["transpose_hint_full"],
                "transpose_hint_mask": oracle["conditions"]["transpose_hint_mask"],
            },
            "correspondence": {
                "crops": crops,
                "only_true_source_yields_truth": bool(
                    true_results[0]["truth_present"]
                    and not any(item["truth_present"] for item in distractors)
                ),
            },
            "hinted_correspondence": {
                "crops": hinted_crops,
                "only_true_source_yields_truth": bool(
                    true_hinted[0]["truth_present"]
                    and not any(item["truth_present"] for item in distractor_hinted)
                ),
            },
            "model_likelihood": oracle["nll"],
            "decoding": {
                **oracle["forced_prefix"],
                "smallest_exact_prefix_rows": min(exact_prefixes)
                if exact_prefixes
                else None,
            },
        },
        "limitations": {
            "oracle_crop_is_ood": True,
            "transpose_probe_is_capability_test_not_fair_solver": True,
            "results_require_gpu_execution": False,
            "forced_prefix_scores_comparable_across_prefixes": False,
        },
        "v1_findings": {
            "report_preserved_as": "stage_diagnosis_report.json",
            "crop_without_hint_failed": True,
            "crop_with_two_transpose_demonstrations_succeeded": True,
            "full_baseline_failed": True,
            "eight_forced_rows_only_error": {"row": 8, "column": 0},
            "eight_forced_rows_truth_probability_at_error": 0.010953,
            "min_probability": 0.2,
        },
        "v2_hypotheses": {
            "transpose_hint_full": (
                "tests whether supplying the operation is sufficient when localization "
                "remains unknown"
            ),
            "transpose_hint_mask": (
                "tests whether supplying both operation and localized source is sufficient "
                "while preserving canvas shape"
            ),
            "transpose_hint_crops": (
                "tests source correspondence with an identical operation hint and budget"
            ),
            PREFIX_8_ROWS_PLUS_1_CELL: (
                "tests whether forcing the sole low-probability truth cell closes DFS"
            ),
        },
    }
