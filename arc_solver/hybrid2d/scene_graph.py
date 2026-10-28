"""Leakage-safe component views over one grid or a same-shape grid pair."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable

from .types import BBox, Coord, Grid, validate_grid


Mask = tuple[Coord, ...]


@dataclass(frozen=True, order=True)
class ViewSpec:
    background: int
    connectivity: int
    multicolor: bool = False
    background_rule: str = "literal"

    @property
    def description_length(self) -> int:
        return 1 + int(self.connectivity == 8) + 2 * int(self.multicolor)

    def as_dict(self) -> dict[str, Any]:
        return {
            "background": self.background,
            "background_rule": self.background_rule,
            "connectivity": self.connectivity,
            "multicolor": self.multicolor,
        }


@dataclass(frozen=True)
class Component:
    cells: frozenset[Coord]
    color: int | None
    colors: tuple[int, ...]
    area: int
    bbox: BBox
    mask: Mask
    d4_mask: Mask
    hole_count: int
    topology: str
    border_touch: tuple[str, ...]
    relative_position: tuple[str, str]
    equivalence_class: int = 0
    repetition: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "colors": list(self.colors),
            "area": self.area,
            "bbox": self.bbox.as_list(),
            "mask": [list(cell) for cell in self.mask],
            "d4_normalized_mask": [list(cell) for cell in self.d4_mask],
            "hole_count": self.hole_count,
            "topology": self.topology,
            "border_touch": list(self.border_touch),
            "relative_position": list(self.relative_position),
            "equivalence_class": self.equivalence_class,
            "repetition": self.repetition,
        }


def _directions(connectivity: int) -> tuple[Coord, ...]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    cardinal = ((-1, 0), (0, -1), (0, 1), (1, 0))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return cardinal + (diagonal if connectivity == 8 else ())


def normalize_mask(cells: Iterable[Coord]) -> Mask:
    materialized = tuple(cells)
    r0 = min(row for row, _ in materialized)
    c0 = min(column for _, column in materialized)
    return tuple(sorted((row - r0, column - c0) for row, column in materialized))


def d4_normalize_mask(mask: Mask) -> Mask:
    variants: list[Mask] = []
    for swap in (False, True):
        for sr, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            transformed = [
                (sr * (column if swap else row), sc * (row if swap else column))
                for row, column in mask
            ]
            variants.append(normalize_mask(transformed))
    return min(variants)


def _holes(cells: frozenset[Coord], bbox: BBox) -> int:
    """Count cardinally-connected enclosed regions inside the component bbox."""
    empty = set(bbox.coordinates()) - set(cells)
    exterior: set[Coord] = set()
    queue: deque[Coord] = deque()
    for cell in empty:
        row, column = cell
        if row in (bbox.r0, bbox.r1 - 1) or column in (bbox.c0, bbox.c1 - 1):
            exterior.add(cell)
            queue.append(cell)
    while queue:
        row, column = queue.popleft()
        for dr, dc in _directions(4):
            adjacent = row + dr, column + dc
            if adjacent in empty and adjacent not in exterior:
                exterior.add(adjacent)
                queue.append(adjacent)
    unseen = empty - exterior
    count = 0
    while unseen:
        count += 1
        queue.append(unseen.pop())
        while queue:
            row, column = queue.popleft()
            for dr, dc in _directions(4):
                adjacent = row + dr, column + dc
                if adjacent in unseen:
                    unseen.remove(adjacent)
                    queue.append(adjacent)
    return count


class ComponentScene:
    """A canonical segmentation view; it receives only one grid."""

    def __init__(self, grid: Grid, view: ViewSpec):
        self.grid = validate_grid(grid)
        self.view = view
        self.height = len(grid)
        self.width = len(grid[0])
        raw = self._segment()
        classes: dict[tuple[Any, ...], int] = {}
        counts = Counter((component.color, component.d4_mask) for component in raw)
        components: list[Component] = []
        for component in raw:
            key = component.color, component.d4_mask
            if key not in classes:
                classes[key] = len(classes)
            components.append(
                Component(
                    **{
                        **component.__dict__,
                        "equivalence_class": classes[key],
                        "repetition": counts[key],
                    }
                )
            )
        self.components = tuple(components)

    def _segment(self) -> list[Component]:
        foreground = {
            (row, column)
            for row in range(self.height)
            for column in range(self.width)
            if self.grid[row][column] != self.view.background
        }
        unseen = set(foreground)
        result: list[Component] = []
        directions = _directions(self.view.connectivity)
        while unseen:
            seed = min(unseen)
            seed_color = self.grid[seed[0]][seed[1]]
            cells = {seed}
            unseen.remove(seed)
            queue = deque([seed])
            while queue:
                row, column = queue.popleft()
                for dr, dc in directions:
                    adjacent = row + dr, column + dc
                    if adjacent not in unseen:
                        continue
                    if not self.view.multicolor and (
                        self.grid[adjacent[0]][adjacent[1]] != seed_color
                    ):
                        continue
                    unseen.remove(adjacent)
                    cells.add(adjacent)
                    queue.append(adjacent)
            frozen = frozenset(cells)
            rows = [cell[0] for cell in cells]
            columns = [cell[1] for cell in cells]
            bbox = BBox(min(rows), max(rows) + 1, min(columns), max(columns) + 1)
            mask = normalize_mask(cells)
            colors = tuple(sorted({self.grid[row][column] for row, column in cells}))
            holes = _holes(frozen, bbox)
            touches = tuple(
                name
                for name, hit in (
                    ("top", bbox.r0 == 0),
                    ("bottom", bbox.r1 == self.height),
                    ("left", bbox.c0 == 0),
                    ("right", bbox.c1 == self.width),
                )
                if hit
            )
            row_position = (
                "top"
                if 2 * (bbox.r0 + bbox.r1) < self.height
                else "bottom"
                if 2 * (bbox.r0 + bbox.r1) > 3 * self.height
                else "middle"
            )
            column_position = (
                "left"
                if 2 * (bbox.c0 + bbox.c1) < self.width
                else "right"
                if 2 * (bbox.c0 + bbox.c1) > 3 * self.width
                else "middle"
            )
            result.append(
                Component(
                    cells=frozen,
                    color=colors[0] if len(colors) == 1 else None,
                    colors=colors,
                    area=len(cells),
                    bbox=bbox,
                    mask=mask,
                    d4_mask=d4_normalize_mask(mask),
                    hole_count=holes,
                    topology="holed" if holes else "solid",
                    border_touch=touches,
                    relative_position=(row_position, column_position),
                )
            )
        return sorted(result, key=lambda item: (item.bbox, item.color or -1, item.mask))

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": [self.height, self.width],
            "view": self.view.as_dict(),
            "component_count": len(self.components),
            "components": [component.as_dict() for component in self.components],
        }


def background_hypotheses(grid: Grid) -> tuple[tuple[str, int], ...]:
    clean = validate_grid(grid)
    counts = Counter(cell for row in clean for cell in row)
    border = (
        clean[0]
        + clean[-1]
        + [row[0] for row in clean[1:-1]]
        + [row[-1] for row in clean[1:-1]]
    )
    candidates = (
        ("zero", 0),
        ("global_majority", counts.most_common(1)[0][0]),
        ("border_majority", Counter(border).most_common(1)[0][0]),
    )
    seen: set[int] = set()
    unique = []
    for rule, color in candidates:
        if color not in seen:
            seen.add(color)
            unique.append((rule, color))
    return tuple(unique)


def segmentation_views(grid: Grid, *, include_multicolor: bool = True) -> tuple[ViewSpec, ...]:
    return tuple(
        ViewSpec(color, connectivity, multicolor, rule)
        for rule, color in background_hypotheses(grid)
        for connectivity in (4, 8)
        for multicolor in ((False, True) if include_multicolor else (False,))
    )


@dataclass(frozen=True)
class PairedSceneGraph:
    """Input/output component graphs for a same-shape demonstration."""

    input: ComponentScene
    output: ComponentScene
    changed: frozenset[Coord]
    transitions: tuple[tuple[int, int, int], ...]

    @classmethod
    def build(cls, input_grid: Grid, output_grid: Grid, view: ViewSpec) -> "PairedSceneGraph":
        before = validate_grid(input_grid)
        after = validate_grid(output_grid)
        if (len(before), len(before[0])) != (len(after), len(after[0])):
            raise ValueError("paired scene graphs require equal shapes")
        changed = frozenset(
            (row, column)
            for row in range(len(before))
            for column in range(len(before[0]))
            if before[row][column] != after[row][column]
        )
        transitions = Counter(
            (before[row][column], after[row][column]) for row, column in changed
        )
        return cls(
            ComponentScene(before, view),
            ComponentScene(after, view),
            changed,
            tuple(sorted((source, target, count) for (source, target), count in transitions.items())),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "view": self.input.view.as_dict(),
            "changed_cell_count": len(self.changed),
            "transitions": [list(item) for item in self.transitions],
            "input_component_count": len(self.input.components),
            "output_component_count": len(self.output.components),
        }
