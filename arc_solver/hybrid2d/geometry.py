"""D4 and coordinate utilities."""

from __future__ import annotations

from .types import AffineD4, BBox, Coord, Grid

D4: tuple[AffineD4, ...] = (
    AffineD4("identity", ((1, 0), (0, 1))),
    AffineD4("rotate_90", ((0, 1), (-1, 0))),
    AffineD4("rotate_180", ((-1, 0), (0, -1))),
    AffineD4("rotate_270", ((0, -1), (1, 0))),
    AffineD4("reflect_rows", ((-1, 0), (0, 1))),
    AffineD4("reflect_columns", ((1, 0), (0, -1))),
    AffineD4("transpose", ((0, 1), (1, 0))),
    AffineD4("anti_transpose", ((0, -1), (-1, 0))),
)


def in_grid(grid: Grid, coordinate: Coord) -> bool:
    row, column = coordinate
    return 0 <= row < len(grid) and 0 <= column < len(grid[0])


def bbox_of(coordinates: list[Coord]) -> BBox:
    if not coordinates:
        raise ValueError("cannot bound an empty coordinate set")
    return BBox(
        min(row for row, _ in coordinates),
        max(row for row, _ in coordinates) + 1,
        min(column for _, column in coordinates),
        max(column for _, column in coordinates) + 1,
    )


def crop(grid: Grid, bbox: BBox) -> Grid:
    return [row[bbox.c0 : bbox.c1] for row in grid[bbox.r0 : bbox.r1]]


def frame_coordinates(bbox: BBox, height: int, width: int, radius: int = 1) -> list[Coord]:
    return [
        (row, column)
        for row in range(max(0, bbox.r0 - radius), min(height, bbox.r1 + radius))
        for column in range(max(0, bbox.c0 - radius), min(width, bbox.c1 + radius))
        if not bbox.contains((row, column))
    ]


def offset_bounds(matrix: tuple[tuple[int, int], tuple[int, int]], height: int, width: int) -> tuple[range, range]:
    """Return offsets which can map at least one grid coordinate into the grid."""
    corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
    mapped = [
        (
            matrix[0][0] * row + matrix[0][1] * column,
            matrix[1][0] * row + matrix[1][1] * column,
        )
        for row, column in corners
    ]
    min_row, max_row = min(x[0] for x in mapped), max(x[0] for x in mapped)
    min_col, max_col = min(x[1] for x in mapped), max(x[1] for x in mapped)
    return (
        range(-max_row, height - min_row),
        range(-max_col, width - min_col),
    )
