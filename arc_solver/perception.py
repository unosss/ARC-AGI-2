"""Objective single-grid perception probes for ARC models."""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from typing import Any, Iterable


Grid = list[list[int]]
ARC_COLORS = (
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
)


def validate_grid(grid: Grid) -> None:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid must be rectangular")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in grid for cell in row):
        raise ValueError("cells must be integers from 0 through 9")


def connected_components(grid: Grid, connectivity: int = 4) -> list[dict[str, Any]]:
    """Return same-color nonzero components in deterministic order."""
    validate_grid(grid)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = len(grid), len(grid[0])
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for row in range(height):
        for column in range(width):
            color = grid[row][column]
            if color == 0 or (row, column) in seen:
                continue
            queue = deque([(row, column)])
            seen.add((row, column))
            cells = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column))
                for delta_row, delta_column in directions:
                    next_row = current_row + delta_row
                    next_column = current_column + delta_column
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and (next_row, next_column) not in seen
                        and grid[next_row][next_column] == color
                    ):
                        seen.add((next_row, next_column))
                        queue.append((next_row, next_column))
            min_row = min(cell[0] for cell in cells)
            min_column = min(cell[1] for cell in cells)
            max_row = max(cell[0] for cell in cells)
            max_column = max(cell[1] for cell in cells)
            shape = sorted([[r - min_row, c - min_column] for r, c in cells])
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": [min_row, min_column, max_row, max_column],
                    "shape": shape,
                }
            )
    return sorted(
        components,
        key=lambda item: (
            -item["size"],
            item["color"],
            item["bbox"][0],
            item["bbox"][1],
            item["shape"],
        ),
    )


def sample_coordinates(grid: Grid, limit: int = 6) -> list[tuple[int, int]]:
    """Choose deterministic cells that test boundaries, center, and foreground."""
    validate_grid(grid)
    height, width = len(grid), len(grid[0])
    foreground = [
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != 0
    ]
    candidates = [
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
        (height // 2, width // 2),
    ]
    if foreground:
        candidates.extend([foreground[0], foreground[-1]])
        candidates.append(foreground[len(foreground) // 2])
    unique = []
    for coordinate in candidates:
        if coordinate not in unique:
            unique.append(coordinate)
    return unique[:limit]


def sample_row_indices(grid: Grid, limit: int = 2) -> list[int]:
    validate_grid(grid)
    counts = [sum(cell != 0 for cell in row) for row in grid]
    ranked = sorted(range(len(grid)), key=lambda row: (-counts[row], row))
    result = []
    for row in [ranked[0], len(grid) // 2, 0, len(grid) - 1]:
        if row not in result:
            result.append(row)
        if len(result) == limit:
            break
    return result


def _same_shape_pairs(components: list[dict[str, Any]]) -> list[list[int]]:
    pairs = []
    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            if components[left]["shape"] == components[right]["shape"]:
                pairs.append([left, right])
    return pairs


def describe_grid(grid: Grid, max_components: int = 6) -> dict[str, Any]:
    """Build the exact answer expected from a model for one isolated grid."""
    validate_grid(grid)
    height, width = len(grid), len(grid[0])
    counts = Counter(cell for row in grid for cell in row)
    coordinates = sample_coordinates(grid)
    row_indices = sample_row_indices(grid)
    foreground = [
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != 0
    ]
    bbox = None
    if foreground:
        bbox = [
            min(row for row, _ in foreground),
            min(column for _, column in foreground),
            max(row for row, _ in foreground),
            max(column for _, column in foreground),
        ]
    components_4 = connected_components(grid, 4)
    components_8 = connected_components(grid, 8)
    largest = components_4[:max_components]
    return {
        "dimensions": [height, width],
        "color_counts": {str(color): counts[color] for color in sorted(counts)},
        "sampled_cells": {
            f"{row},{column}": grid[row][column] for row, column in coordinates
        },
        "sampled_rows": {str(row): grid[row] for row in row_indices},
        "nonzero_bbox": bbox,
        "zero_rows": [
            row for row in range(height) if all(cell == 0 for cell in grid[row])
        ],
        "zero_cols": [
            column
            for column in range(width)
            if all(grid[row][column] == 0 for row in range(height))
        ],
        "component_count_4": len(components_4),
        "component_count_8": len(components_8),
        "largest_components_4": largest,
        "same_shape_pairs_4": _same_shape_pairs(largest),
        "symmetry": {
            "horizontal": grid == list(reversed(grid)),
            "vertical": all(row == list(reversed(row)) for row in grid),
            "rotation_180": grid
            == [list(reversed(row)) for row in reversed(grid)],
        },
    }


def build_probe_prompt(grid: Grid, modality: str) -> str:
    """Create one bundled objective probe; the image is attached separately."""
    if modality not in {"text", "image", "both"}:
        raise ValueError("modality must be text, image, or both")
    expected = describe_grid(grid)
    cell_keys = list(expected["sampled_cells"])
    row_keys = list(expected["sampled_rows"])
    source = ""
    if modality in {"text", "both"}:
        source = "\nGrid as JSON:\n" + json.dumps(grid, separators=(",", ":"))
    image_note = (
        "\nA color-grid image is attached. Rows run top-to-bottom and columns "
        "left-to-right, both zero-indexed."
        if modality in {"image", "both"}
        else ""
    )
    return f"""Analyze this single ARC grid independently. Do not infer any input-output rule.
Use same-color nonzero connected components. For 4-connectivity use only orthogonal
neighbors; for 8-connectivity include diagonals. Bounding boxes are inclusive
[min_row,min_col,max_row,max_col]. Component shapes are cell coordinates normalized
to the bounding-box top-left. Sort components by decreasing size, then color,
min_row, min_col; return at most six. same_shape_pairs_4 contains index pairs into
largest_components_4 whose normalized shapes are exactly equal, regardless of color.
Horizontal symmetry means top-bottom reflection; vertical means left-right reflection.
Report sampled cells {cell_keys} and complete sampled rows {row_keys}.
Return only one JSON object with exactly these keys:
dimensions, color_counts, sampled_cells, sampled_rows, nonzero_bbox, zero_rows,
zero_cols, component_count_4, component_count_8, largest_components_4,
same_shape_pairs_4, symmetry.{image_note}{source}"""


def fixed_probe_description(grid: Grid) -> dict[str, Any]:
    """Describe a grid using content-independent cell and row questions."""
    expected = describe_grid(grid)
    height, width = len(grid), len(grid[0])
    fixed_cells = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    expected["sampled_cells"] = {
        f"{row},{column}": (
            grid[row][column] if row < height and column < width else None
        )
        for row, column in fixed_cells
    }
    expected["sampled_rows"] = {
        str(row): grid[row] if row < height else None for row in (0, 1)
    }
    return expected


def fixed_probe_prompt(grid: Grid, modality: str) -> str:
    """Build the fixed Experiment 027/028 prompt without content-derived hints."""
    validate_grid(grid)
    if modality not in {"text", "image", "both"}:
        raise ValueError("modality must be text, image, or both")
    source = ""
    if modality in {"text", "both"}:
        source = "\nGrid as JSON:\n" + json.dumps(grid, separators=(",", ":"))
    image_note = (
        "\nA color-grid image is attached. Rows run top-to-bottom and columns "
        "left-to-right, both zero-indexed."
        if modality in {"image", "both"}
        else ""
    )
    return f"""Analyze this single ARC grid independently. Do not infer any input-output rule.
Use same-color nonzero connected components. For 4-connectivity use only orthogonal
neighbors; for 8-connectivity include diagonals. Bounding boxes are inclusive
[min_row,min_col,max_row,max_col]. Component shapes are cell coordinates normalized
to the bounding-box top-left. Sort components by decreasing size, then color,
min_row, min_col; return at most six. same_shape_pairs_4 contains index pairs into
largest_components_4 whose normalized shapes are exactly equal, regardless of color.
Horizontal symmetry means top-bottom reflection; vertical means left-right reflection.
Report sampled cells ["0,0","0,1","1,0","1,1","2,2"] and complete rows ["0","1"].
Return only one JSON object with exactly these keys:
dimensions, color_counts, sampled_cells, sampled_rows, nonzero_bbox, zero_rows,
zero_cols, component_count_4, component_count_8, largest_components_4,
same_shape_pairs_4, symmetry.{image_note}{source}"""


def render_grid_image(grid: Grid, cell_size: int = 24):
    """Render an ARC grid as a PIL image without labels or interpolation."""
    validate_grid(grid)
    if cell_size < 4:
        raise ValueError("cell_size must be at least 4")
    try:
        from PIL import Image, ImageColor, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image probes") from exc
    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width * cell_size, height * cell_size))
    draw = ImageDraw.Draw(image)
    for row in range(height):
        for column in range(width):
            left = column * cell_size
            top = row * cell_size
            draw.rectangle(
                [left, top, left + cell_size - 1, top + cell_size - 1],
                fill=ImageColor.getrgb(ARC_COLORS[grid[row][column]]),
            )
    line_color = ImageColor.getrgb("#303030")
    for row in range(height + 1):
        y = min(row * cell_size, height * cell_size - 1)
        draw.line((0, y, width * cell_size - 1, y), fill=line_color)
    for column in range(width + 1):
        x = min(column * cell_size, width * cell_size - 1)
        draw.line((x, 0, x, height * cell_size - 1), fill=line_color)
    return image


def parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _fraction(correct: Iterable[bool]) -> float:
    values = list(correct)
    return sum(values) / len(values) if values else 1.0


def score_description(
    prediction: dict[str, Any] | None, expected: dict[str, Any]
) -> dict[str, Any]:
    """Score reading, object parsing, and relation fields independently."""
    if not isinstance(prediction, dict):
        return {
            "structured": False,
            "reading": 0.0,
            "objects": 0.0,
            "relations": 0.0,
            "overall": 0.0,
            "exact": False,
        }
    predicted_counts = prediction.get("color_counts", {})
    predicted_cells = prediction.get("sampled_cells", {})
    predicted_rows = prediction.get("sampled_rows", {})
    if not isinstance(predicted_counts, dict):
        predicted_counts = {}
    if not isinstance(predicted_cells, dict):
        predicted_cells = {}
    if not isinstance(predicted_rows, dict):
        predicted_rows = {}
    reading_parts = [
        prediction.get("dimensions") == expected["dimensions"],
        predicted_counts == expected["color_counts"],
        _fraction(
            predicted_cells.get(key) == value
            for key, value in expected["sampled_cells"].items()
        ),
        _fraction(
            predicted_rows.get(key) == value
            for key, value in expected["sampled_rows"].items()
        ),
    ]
    object_parts = [
        prediction.get("component_count_4") == expected["component_count_4"],
        prediction.get("component_count_8") == expected["component_count_8"],
        prediction.get("largest_components_4")
        == expected["largest_components_4"],
    ]
    predicted_symmetry = prediction.get("symmetry", {})
    if not isinstance(predicted_symmetry, dict):
        predicted_symmetry = {}
    relation_parts = [
        prediction.get("nonzero_bbox") == expected["nonzero_bbox"],
        prediction.get("zero_rows") == expected["zero_rows"],
        prediction.get("zero_cols") == expected["zero_cols"],
        prediction.get("same_shape_pairs_4") == expected["same_shape_pairs_4"],
        _fraction(
            predicted_symmetry.get(key) == value
            for key, value in expected["symmetry"].items()
        ),
    ]
    reading = sum(float(value) for value in reading_parts) / len(reading_parts)
    objects = sum(float(value) for value in object_parts) / len(object_parts)
    relations = sum(float(value) for value in relation_parts) / len(relation_parts)
    overall = (reading + objects + relations) / 3
    return {
        "structured": True,
        "reading": reading,
        "objects": objects,
        "relations": relations,
        "overall": overall,
        "exact": prediction == expected,
    }
