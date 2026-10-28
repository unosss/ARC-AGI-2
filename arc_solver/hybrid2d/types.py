"""Typed value objects used by the deterministic hybrid 2D solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

Grid: TypeAlias = list[list[int]]
Coord: TypeAlias = tuple[int, int]
Matrix2: TypeAlias = tuple[tuple[int, int], tuple[int, int]]


def validate_grid(value: Any, *, name: str = "grid") -> Grid:
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        raise ValueError(f"{name} must be a non-empty rectangular grid")
    width = len(value[0])
    if width == 0 or any(not isinstance(row, list) or len(row) != width for row in value):
        raise ValueError(f"{name} must be a non-empty rectangular grid")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in value for cell in row):
        raise ValueError(f"{name} cells must be integers from 0 through 9")
    return [row[:] for row in value]


@dataclass(frozen=True, order=True)
class BBox:
    """Half-open rectangle."""

    r0: int
    r1: int
    c0: int
    c1: int

    def __post_init__(self) -> None:
        if self.r0 < 0 or self.c0 < 0 or self.r1 <= self.r0 or self.c1 <= self.c0:
            raise ValueError("invalid half-open bounding box")

    @property
    def height(self) -> int:
        return self.r1 - self.r0

    @property
    def width(self) -> int:
        return self.c1 - self.c0

    @property
    def area(self) -> int:
        return self.height * self.width

    def contains(self, coordinate: Coord) -> bool:
        row, column = coordinate
        return self.r0 <= row < self.r1 and self.c0 <= column < self.c1

    def coordinates(self) -> list[Coord]:
        return [
            (row, column)
            for row in range(self.r0, self.r1)
            for column in range(self.c0, self.c1)
        ]

    def as_list(self) -> list[int]:
        return [self.r0, self.r1, self.c0, self.c1]


@dataclass(frozen=True)
class AffineD4:
    """A destination-to-source coordinate map ``x -> A*x + b``."""

    name: str
    matrix: Matrix2
    offset: Coord = (0, 0)

    @property
    def swaps_axes(self) -> bool:
        return self.matrix[0][0] == 0 and self.matrix[1][1] == 0

    def apply(self, coordinate: Coord) -> Coord:
        row, column = coordinate
        (a, b), (c, d) = self.matrix
        return (
            a * row + b * column + self.offset[0],
            c * row + d * column + self.offset[1],
        )

    def with_offset(self, offset: Coord) -> "AffineD4":
        return AffineD4(self.name, self.matrix, offset)

    def description(self) -> str:
        return f"{self.name}+({self.offset[0]},{self.offset[1]})"


@dataclass(frozen=True)
class RelationWitness:
    relation: AffineD4
    kind: str
    support: int
    mismatch: int
    frame_support: int = 0
    target_bbox: BBox | None = None
    source_bbox: BBox | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation.description(),
            "kind": self.kind,
            "support": self.support,
            "mismatch": self.mismatch,
            "frame_support": self.frame_support,
            "target_bbox": self.target_bbox.as_list() if self.target_bbox else None,
            "source_bbox": self.source_bbox.as_list() if self.source_bbox else None,
        }


@dataclass(frozen=True)
class Candidate:
    grid: Grid
    score: tuple[int, ...]
    program: str
    witnesses: tuple[RelationWitness, ...]

    def as_dict(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "grid": self.grid,
            "score": list(self.score),
            "program": self.program,
            "witnesses": [witness.as_dict() for witness in self.witnesses],
        }
