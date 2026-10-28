"""Deterministic NVARC-compatible D4 and color augmentations."""

from __future__ import annotations

import copy
import random
from typing import Any, Sequence


STRETCH_MODES = ("identity", "rows", "cols", "both")


def _repeat_positions(length: int, *, max_dim: int, phase: int) -> set[int]:
    """Choose evenly spaced cells to duplicate once, up to ``max_dim``.

    A literal 2x stretch is used when it fits.  Near ARC's 30-cell limit the
    same operation becomes a capped stretch: e.g. a length-25 axis duplicates
    five evenly spaced positions and becomes length 30.
    """
    if not 1 <= length <= max_dim:
        raise ValueError("axis length must be in 1..max_dim")
    extra = min(length, max_dim - length)
    if extra == 0:
        return set()
    offset = phase % length
    return {
        (offset + ((2 * index + 1) * length) // (2 * extra)) % length
        for index in range(extra)
    }


def stretch_grid(
    grid: Sequence[Sequence[int]],
    *,
    repeat_rows: bool,
    repeat_cols: bool,
    max_dim: int = 30,
    row_phase: int = 0,
    col_phase: int = 0,
) -> list[list[int]]:
    """Duplicate rows and/or columns without exceeding ARC's size limit."""
    value = [list(row) for row in grid]
    if not value or not value[0] or any(len(row) != len(value[0]) for row in value):
        raise ValueError("grid must be a non-empty rectangle")
    if len(value) > max_dim or len(value[0]) > max_dim:
        raise ValueError("grid exceeds max_dim")
    row_positions = (
        _repeat_positions(len(value), max_dim=max_dim, phase=row_phase)
        if repeat_rows
        else set()
    )
    col_positions = (
        _repeat_positions(len(value[0]), max_dim=max_dim, phase=col_phase)
        if repeat_cols
        else set()
    )
    stretched = []
    for row_index, row in enumerate(value):
        expanded = [
            cell
            for col_index, cell in enumerate(row)
            for _ in range(2 if col_index in col_positions else 1)
        ]
        stretched.extend(
            [expanded.copy() for _ in range(2 if row_index in row_positions else 1)]
        )
    return stretched


def stretch_task(
    task: dict[str, Any],
    *,
    mode: str,
    phase: int = 0,
    max_dim: int = 30,
    keep_test_outputs: bool = False,
) -> dict[str, Any]:
    """Apply one capped anisotropic stretch consistently to every visible grid."""
    if mode not in STRETCH_MODES:
        raise ValueError(f"unknown stretch mode: {mode}")
    repeat_rows = mode in {"rows", "both"}
    repeat_cols = mode in {"cols", "both"}
    result = copy.deepcopy(task)
    for pair in result["train"]:
        pair["input"] = stretch_grid(
            pair["input"],
            repeat_rows=repeat_rows,
            repeat_cols=repeat_cols,
            max_dim=max_dim,
            row_phase=phase,
            col_phase=phase,
        )
        pair["output"] = stretch_grid(
            pair["output"],
            repeat_rows=repeat_rows,
            repeat_cols=repeat_cols,
            max_dim=max_dim,
            row_phase=phase,
            col_phase=phase,
        )
    for item in result.get("test", []):
        item["input"] = stretch_grid(
            item["input"],
            repeat_rows=repeat_rows,
            repeat_cols=repeat_cols,
            max_dim=max_dim,
            row_phase=phase,
            col_phase=phase,
        )
        if keep_test_outputs and "output" in item:
            item["output"] = stretch_grid(
                item["output"],
                repeat_rows=repeat_rows,
                repeat_cols=repeat_cols,
                max_dim=max_dim,
                row_phase=phase,
                col_phase=phase,
            )
        else:
            item.pop("output", None)
    return result


def stretch_grid_exact(
    grid: Sequence[Sequence[int]],
    *,
    repeat_rows: bool,
    repeat_cols: bool,
    max_dim: int = 30,
) -> list[list[int]]:
    """Repeat every selected axis position exactly twice or reject the grid."""
    value = [list(row) for row in grid]
    if not value or not value[0] or any(len(row) != len(value[0]) for row in value):
        raise ValueError("grid must be a non-empty rectangle")
    target_height = len(value) * (2 if repeat_rows else 1)
    target_width = len(value[0]) * (2 if repeat_cols else 1)
    if target_height > max_dim or target_width > max_dim:
        raise ValueError(
            f"exact stretch would exceed {max_dim}: "
            f"{len(value)}x{len(value[0])} -> {target_height}x{target_width}"
        )
    return [
        expanded.copy()
        for row in value
        for expanded in [
            [
                cell
                for cell in row
                for _ in range(2 if repeat_cols else 1)
            ]
        ]
        for _ in range(2 if repeat_rows else 1)
    ]


def stretch_task_exact(
    task: dict[str, Any],
    *,
    mode: str,
    max_dim: int = 30,
    keep_test_outputs: bool = False,
) -> dict[str, Any]:
    """Apply exact uniform 2× row/column repetition to all visible grids."""
    if mode not in STRETCH_MODES:
        raise ValueError(f"unknown stretch mode: {mode}")
    repeat_rows = mode in {"rows", "both"}
    repeat_cols = mode in {"cols", "both"}
    result = copy.deepcopy(task)

    def transform(grid: Sequence[Sequence[int]]) -> list[list[int]]:
        return stretch_grid_exact(
            grid,
            repeat_rows=repeat_rows,
            repeat_cols=repeat_cols,
            max_dim=max_dim,
        )

    for pair in result["train"]:
        pair["input"] = transform(pair["input"])
        pair["output"] = transform(pair["output"])
    for item in result.get("test", []):
        item["input"] = transform(item["input"])
        if keep_test_outputs and "output" in item:
            item["output"] = transform(item["output"])
        else:
            item.pop("output", None)
    return result


def d4(grid: Sequence[Sequence[int]], index: int) -> list[list[int]]:
    value = [list(row) for row in grid]
    if not 0 <= index < 8:
        raise ValueError("D4 index must be in 0..7")
    if index >= 4:
        value = [row[::-1] for row in value]
        index -= 4
    for _ in range(index):
        value = [list(row) for row in zip(*value[::-1])]
    return value


def inverse_d4(grid: Sequence[Sequence[int]], index: int) -> list[list[int]]:
    if index < 4:
        return d4(grid, (-index) % 4)
    # Our convention applies horizontal reflection, then rotation.
    value = d4(grid, (-index + 4) % 4)
    return [row[::-1] for row in value]


def color_permutation(
    seed: int, *, preserve_background: bool = True
) -> tuple[int, ...]:
    """Return a deterministic color permutation.

    Component metadata defines ARC color 0 as background.  Moving 0 would make
    the component arm observe a different foreground decomposition from the
    grid shown to the language model.  Experiment 021 therefore permutes the
    nine foreground colors while keeping the background fixed.
    """
    foreground = list(range(1, 10))
    random.Random(seed).shuffle(foreground)
    if preserve_background:
        return (0, *foreground)
    values = [0, *foreground]
    random.Random(seed).shuffle(values)
    return tuple(values)


def apply_colors(
    grid: Sequence[Sequence[int]], permutation: Sequence[int]
) -> list[list[int]]:
    if sorted(permutation) != list(range(10)):
        raise ValueError("color mapping must be a permutation of 0..9")
    return [[int(permutation[cell]) for cell in row] for row in grid]


def invert_colors(
    grid: Sequence[Sequence[int]], permutation: Sequence[int]
) -> list[list[int]]:
    inverse = [0] * 10
    for source, target in enumerate(permutation):
        inverse[target] = source
    return apply_colors(grid, inverse)


def augment_task(
    task: dict[str, Any],
    *,
    geometry: int,
    permutation: Sequence[int],
    order_seed: int,
    keep_test_outputs: bool = False,
) -> dict[str, Any]:
    result = copy.deepcopy(task)
    rng = random.Random(order_seed)
    rng.shuffle(result["train"])
    for pair in result["train"]:
        pair["input"] = apply_colors(d4(pair["input"], geometry), permutation)
        pair["output"] = apply_colors(d4(pair["output"], geometry), permutation)
    for item in result.get("test", []):
        item["input"] = apply_colors(d4(item["input"], geometry), permutation)
        if keep_test_outputs and "output" in item:
            item["output"] = apply_colors(d4(item["output"], geometry), permutation)
        else:
            item.pop("output", None)
    return result


def invert_prediction(
    grid: Sequence[Sequence[int]], *, geometry: int, permutation: Sequence[int]
) -> list[list[int]]:
    return inverse_d4(invert_colors(grid, permutation), geometry)
