"""CPU-only, deterministic structural novelty measures for ARC grids.

Exact orbit equivalence treats every integer color, including 0, as a normal
permutable symbol. Foreground-only measures explicitly use 0 as background.
"""

from __future__ import annotations

from collections import Counter, deque
from fractions import Fraction
import math
from typing import Any, Iterable, Sequence


Grid = Sequence[Sequence[int]]
Coord = tuple[int, int]
D4_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "reflect",
    "reflect_rot90",
    "reflect_rot180",
    "reflect_rot270",
)


def _as_tuple(grid: Grid) -> tuple[tuple[int, ...], ...]:
    result = tuple(tuple(int(value) for value in row) for row in grid)
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise ValueError("grid must be a non-empty rectangle")
    return result


def d4_transform(grid: Grid, transform: str) -> tuple[tuple[int, ...], ...]:
    """Apply one named D4 transform; 90-degree rotations are clockwise."""
    current = _as_tuple(grid)
    reflected = transform.startswith("reflect")
    if reflected:
        current = tuple(tuple(reversed(row)) for row in current)
        suffix = transform.removeprefix("reflect")
        turns = 0 if not suffix else int(suffix.removeprefix("_rot")) // 90
    else:
        turns = 0 if transform == "identity" else int(transform.removeprefix("rot")) // 90
    if transform not in D4_NAMES:
        raise ValueError(f"unknown D4 transform: {transform}")
    for _ in range(turns):
        current = tuple(tuple(row) for row in zip(*current[::-1]))
    return current


def _extend_injective_mapping(
    source: Grid,
    target: Grid,
    mapping: dict[int, int],
    reverse: dict[int, int],
) -> bool:
    source_grid, target_grid = _as_tuple(source), _as_tuple(target)
    if (len(source_grid), len(source_grid[0])) != (
        len(target_grid),
        len(target_grid[0]),
    ):
        return False
    for source_row, target_row in zip(source_grid, target_grid):
        for source_color, target_color in zip(source_row, target_row):
            if source_color in mapping and mapping[source_color] != target_color:
                return False
            if target_color in reverse and reverse[target_color] != source_color:
                return False
            mapping[source_color] = target_color
            reverse[target_color] = source_color
    return True


def color_equivalent(source: Grid, target: Grid) -> bool:
    """Return whether one globally injective color relabeling maps source to target."""
    return _extend_injective_mapping(source, target, {}, {})


def input_orbit_match(reference_input: Grid, target_input: Grid) -> dict[str, Any] | None:
    """Find a shared D4/color witness for one input grid, if one exists."""
    for transform in D4_NAMES:
        mapping: dict[int, int] = {}
        reverse: dict[int, int] = {}
        transformed = d4_transform(reference_input, transform)
        if _extend_injective_mapping(transformed, target_input, mapping, reverse):
            return {"transform": transform, "color_mapping": dict(sorted(mapping.items()))}
    return None


def pair_orbit_match(
    reference_pair: dict[str, Grid], target_pair: dict[str, Grid]
) -> dict[str, Any] | None:
    """Find one D4 transform and one injective color map shared by input/output."""
    for transform in D4_NAMES:
        mapping: dict[int, int] = {}
        reverse: dict[int, int] = {}
        if not _extend_injective_mapping(
            d4_transform(reference_pair["input"], transform),
            target_pair["input"],
            mapping,
            reverse,
        ):
            continue
        if _extend_injective_mapping(
            d4_transform(reference_pair["output"], transform),
            target_pair["output"],
            mapping,
            reverse,
        ):
            return {"transform": transform, "color_mapping": dict(sorted(mapping.items()))}
    return None


def _components(grid: Grid, *, binary_nonzero: bool) -> list[tuple[int, list[Coord]]]:
    values = _as_tuple(grid)
    height, width = len(values), len(values[0])
    seen: set[Coord] = set()
    result: list[tuple[int, list[Coord]]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or (binary_nonzero and values[row][col] == 0):
                continue
            color = 1 if binary_nonzero else values[row][col]
            queue = deque([(row, col)])
            seen.add((row, col))
            cells: list[Coord] = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in seen:
                        continue
                    next_color = 1 if binary_nonzero and values[next_row][next_col] != 0 else values[next_row][next_col]
                    if next_color == color:
                        seen.add((next_row, next_col))
                        queue.append((next_row, next_col))
            result.append((color, cells))
    return result


def same_color_component_signature(grid: Grid) -> tuple[int, tuple[int, ...]]:
    """Count and sorted sizes of all same-color 4-connected components, including 0."""
    sizes = tuple(sorted(len(cells) for _, cells in _components(grid, binary_nonzero=False)))
    return len(sizes), sizes


def _canonical_shape(cells: Iterable[Coord]) -> tuple[Coord, ...]:
    coords = tuple(cells)

    def normalize(points: Iterable[Coord]) -> tuple[Coord, ...]:
        points_tuple = tuple(points)
        min_row = min(row for row, _ in points_tuple)
        min_col = min(col for _, col in points_tuple)
        return tuple(sorted((row - min_row, col - min_col) for row, col in points_tuple))

    variants = []
    current = coords
    for _ in range(4):
        variants.append(normalize(current))
        variants.append(normalize((row, -col) for row, col in current))
        current = tuple((col, -row) for row, col in current)
    return min(variants)


def nonzero_binary_shape_signature(grid: Grid) -> tuple[tuple[Coord, ...], ...]:
    """D4/translation-normalized shapes of nonzero binary 4-components."""
    shapes = [_canonical_shape(cells) for _, cells in _components(grid, binary_nonzero=True)]
    return tuple(sorted(shapes))


def orientation_normalized_dimensions(grid: Grid) -> tuple[int, int]:
    values = _as_tuple(grid)
    return tuple(sorted((len(values), len(values[0]))))  # type: ignore[return-value]


def _fraction_pair(numerator: int, denominator: int) -> tuple[int, int]:
    value = Fraction(numerator, denominator)
    return value.numerator, value.denominator


def dimension_relation(input_grid: Grid, output_grid: Grid) -> tuple[tuple[int, int], tuple[int, int]]:
    """Output/input ratios for short and long orientation-normalized dimensions."""
    input_dims = orientation_normalized_dimensions(input_grid)
    output_dims = orientation_normalized_dimensions(output_grid)
    return (
        _fraction_pair(output_dims[0], input_dims[0]),
        _fraction_pair(output_dims[1], input_dims[1]),
    )


def grid_signature(grid: Grid) -> dict[str, Any]:
    """Return explicit orientation/color/component signatures for one grid."""
    values = _as_tuple(grid)
    colors = {value for row in values for value in row}
    component_count, component_sizes = same_color_component_signature(values)
    return {
        "orientation_normalized_dimensions": orientation_normalized_dimensions(values),
        "color_count": len(colors),
        "foreground_cell_count": sum(value != 0 for row in values for value in row),
        "same_color_component_count": component_count,
        "same_color_component_size_multiset": component_sizes,
        "nonzero_binary_shape_multiset": nonzero_binary_shape_signature(values),
    }


SIGNATURE_FLAGS = (
    "input_dimensions_novel",
    "output_dimensions_novel",
    "input_color_count_novel",
    "output_color_count_novel",
    "input_foreground_cell_count_novel",
    "output_foreground_cell_count_novel",
    "input_same_color_components_novel",
    "output_same_color_components_novel",
    "input_nonzero_binary_shapes_novel",
    "output_nonzero_binary_shapes_novel",
    "dimension_relation_novel",
)
ALL_FLAGS = ("pair_orbit_novel", "input_orbit_novel", *SIGNATURE_FLAGS)


def analyze_target(
    target_pair: dict[str, Grid], reference_pairs: Sequence[dict[str, Grid]]
) -> dict[str, Any]:
    """Measure one target pair relative to its remaining training references."""
    if not reference_pairs:
        raise ValueError("at least one reference pair is required")
    target_input = grid_signature(target_pair["input"])
    target_output = grid_signature(target_pair["output"])
    reference_signatures = [
        {
            "input": grid_signature(pair["input"]),
            "output": grid_signature(pair["output"]),
            "dimension_relation": dimension_relation(pair["input"], pair["output"]),
        }
        for pair in reference_pairs
    ]
    pair_matches = [
        {"reference_index": index, **witness}
        for index, pair in enumerate(reference_pairs)
        if (witness := pair_orbit_match(pair, target_pair)) is not None
    ]
    input_matches = [
        {"reference_index": index, **witness}
        for index, pair in enumerate(reference_pairs)
        if (witness := input_orbit_match(pair["input"], target_pair["input"])) is not None
    ]

    def novel(side: str, key: str) -> bool:
        target = target_input if side == "input" else target_output
        return all(target[key] != ref[side][key] for ref in reference_signatures)

    flags = {
        "pair_orbit_novel": not pair_matches,
        "input_orbit_novel": not input_matches,
        "input_dimensions_novel": novel("input", "orientation_normalized_dimensions"),
        "output_dimensions_novel": novel("output", "orientation_normalized_dimensions"),
        "input_color_count_novel": novel("input", "color_count"),
        "output_color_count_novel": novel("output", "color_count"),
        "input_foreground_cell_count_novel": novel("input", "foreground_cell_count"),
        "output_foreground_cell_count_novel": novel("output", "foreground_cell_count"),
        "input_same_color_components_novel": novel("input", "same_color_component_size_multiset"),
        "output_same_color_components_novel": novel("output", "same_color_component_size_multiset"),
        "input_nonzero_binary_shapes_novel": novel("input", "nonzero_binary_shape_multiset"),
        "output_nonzero_binary_shapes_novel": novel("output", "nonzero_binary_shape_multiset"),
        "dimension_relation_novel": all(
            dimension_relation(target_pair["input"], target_pair["output"])
            != ref["dimension_relation"]
            for ref in reference_signatures
        ),
    }
    return {
        "reference_count": len(reference_pairs),
        "pair_orbit_matches": pair_matches,
        "input_orbit_matches": input_matches,
        "target_signatures": {
            "input": target_input,
            "output": target_output,
            "dimension_relation": dimension_relation(
                target_pair["input"], target_pair["output"]
            ),
        },
        "novelty_flags": flags,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _phi(flags: list[bool], outcomes: list[bool]) -> dict[str, Any]:
    counts = {
        "flag_false_outcome_false": 0,
        "flag_false_outcome_true": 0,
        "flag_true_outcome_false": 0,
        "flag_true_outcome_true": 0,
    }
    for flag, outcome in zip(flags, outcomes):
        counts[f"flag_{str(flag).lower()}_outcome_{str(outcome).lower()}"] += 1
    a = counts["flag_true_outcome_true"]
    b = counts["flag_true_outcome_false"]
    c = counts["flag_false_outcome_true"]
    d = counts["flag_false_outcome_false"]
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {**counts, "phi": (a * d - b * c) / denominator if denominator else None}


def _point_biserial(flags: list[bool], values: list[float]) -> float | None:
    if len(set(flags)) < 2 or len(set(values)) < 2:
        return None
    mean_flag = sum(flags) / len(flags)
    mean_value = sum(values) / len(values)
    numerator = sum(
        (float(flag) - mean_flag) * (value - mean_value)
        for flag, value in zip(flags, values)
    )
    flag_ss = sum((float(flag) - mean_flag) ** 2 for flag in flags)
    value_ss = sum((value - mean_value) ** 2 for value in values)
    denominator = math.sqrt(flag_ss * value_ss)
    return numerator / denominator if denominator else None


def summarize_records(records: list[dict[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    """Aggregate flag rates and performance conditionals with exact denominators."""
    if not records:
        raise ValueError("records must not be empty")
    summary: dict[str, Any] = {
        "item_count": len(records),
        "flag_rates": {},
        "performance_by_flag": {},
        "associations": {},
    }
    for flag_name in ALL_FLAGS:
        flags = [bool(record["novelty_flags"][flag_name]) for record in records]
        true_count = sum(flags)
        summary["flag_rates"][flag_name] = {
            "count": true_count,
            "denominator": len(records),
            "rate": true_count / len(records),
            "examples": [
                record["item_key"]
                for record, flag in zip(records, flags)
                if flag
            ][:5],
            "counterexamples": [
                record["item_key"]
                for record, flag in zip(records, flags)
                if not flag
            ][:5],
        }
        summary["performance_by_flag"][flag_name] = {}
        summary["associations"][flag_name] = {}
        for arm in arms:
            arm_rows = [record["performance"][arm] for record in records]
            conditioned: dict[str, Any] = {}
            for flag_value in (False, True):
                selected = [
                    performance
                    for performance, current_flag in zip(arm_rows, flags)
                    if current_flag == flag_value
                ]
                denominator = len(selected)
                conditioned[str(flag_value).lower()] = {
                    "count": denominator,
                    "top2_count": sum(bool(row["top2"]) for row in selected),
                    "top2_rate": _mean([float(bool(row["top2"])) for row in selected]),
                    "truth_coverage_count": sum(
                        bool(row["truth_coverage"]) for row in selected
                    ),
                    "truth_coverage_rate": _mean(
                        [float(bool(row["truth_coverage"])) for row in selected]
                    ),
                    "mean_best_cell_accuracy": _mean(
                        [float(row["best_cell_accuracy"]) for row in selected]
                    ),
                }
            summary["performance_by_flag"][flag_name][arm] = conditioned
            summary["associations"][flag_name][arm] = {
                "top2_contingency_phi": _phi(
                    flags, [bool(row["top2"]) for row in arm_rows]
                ),
                "truth_coverage_contingency_phi": _phi(
                    flags, [bool(row["truth_coverage"]) for row in arm_rows]
                ),
                "best_cell_accuracy_point_biserial_r": _point_biserial(
                    flags, [float(row["best_cell_accuracy"]) for row in arm_rows]
                ),
            }
    return summary


def jsonable(value: Any) -> Any:
    """Recursively convert tuples and integer-key mappings for JSON output."""
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value
