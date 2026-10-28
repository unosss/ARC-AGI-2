"""Compact, typed component/delta program families.

Every search routine sees training pairs only.  Returned programs are shared
executors: they contain no demonstration index, task identifier, or coordinate
list.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .scene_graph import ComponentScene, ViewSpec, segmentation_views
from .types import BBox, Coord, Grid


TrainPair = tuple[Grid, Grid]
CARDINAL: tuple[Coord, ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))
EIGHT: tuple[Coord, ...] = CARDINAL + ((-1, -1), (-1, 1), (1, -1), (1, 1))


class SceneProgram(Protocol):
    family: str
    name: str
    description_length: int

    def execute(self, grid: Grid) -> Grid | None: ...

    def as_dict(self) -> dict[str, Any]: ...


def _same_shape(pair: TrainPair) -> bool:
    before, after = pair
    return len(before) == len(after) and len(before[0]) == len(after[0])


def _copy(grid: Grid) -> Grid:
    return [row[:] for row in grid]


@dataclass(frozen=True)
class LocalCrossRecolor:
    source: int
    target: int
    family: str = "local_motif"
    name: str = "recolor_solid_cardinal_crosses"
    description_length: int = 6

    def execute(self, grid: Grid) -> Grid:
        height, width = len(grid), len(grid[0])
        selected: set[Coord] = set()
        for row in range(1, height - 1):
            for column in range(1, width - 1):
                motif = ((row, column),) + tuple(
                    (row + dr, column + dc) for dr, dc in CARDINAL
                )
                if all(grid[r][c] == self.source for r, c in motif):
                    selected.update(motif)
        result = _copy(grid)
        for row, column in selected:
            result[row][column] = self.target
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "selector": {
                "type": "neighborhood_template",
                "template": "center_and_four_cardinal_neighbors_same_color",
                "source_color": self.source,
            },
            "operation": {"type": "recolor", "target_color": self.target},
            "description_length": self.description_length,
        }


def search_local_motif(train: Sequence[TrainPair]) -> tuple[list[SceneProgram], dict[str, Any]]:
    if not all(_same_shape(pair) for pair in train):
        return [], {"family": "local_motif", "reason": "requires_same_shape"}
    transitions = {
        (before[row][column], after[row][column])
        for before, after in train
        for row in range(len(before))
        for column in range(len(before[0]))
        if before[row][column] != after[row][column]
    }
    programs = [
        LocalCrossRecolor(source, target)
        for source, target in sorted(transitions)
        if source != target
    ]
    exact = [program for program in programs if all(program.execute(a) == b for a, b in train)]
    return exact, {
        "family": "local_motif",
        "generated": len(programs),
        "train_exact": len(exact),
        "reason": None if exact else "no_shared_neighborhood_template",
    }


def _hole_cells(cells: frozenset[Coord], bbox: BBox) -> set[Coord]:
    empty = set(bbox.coordinates()) - set(cells)
    exterior: set[Coord] = set()
    queue: deque[Coord] = deque()
    for cell in empty:
        if cell[0] in (bbox.r0, bbox.r1 - 1) or cell[1] in (bbox.c0, bbox.c1 - 1):
            exterior.add(cell)
            queue.append(cell)
    while queue:
        row, column = queue.popleft()
        for dr, dc in CARDINAL:
            adjacent = row + dr, column + dc
            if adjacent in empty and adjacent not in exterior:
                exterior.add(adjacent)
                queue.append(adjacent)
    return empty - exterior


@dataclass(frozen=True)
class TopologyRecolor:
    view: ViewSpec
    foreground: int
    outline_color: int
    holed_component_color: int
    hole_color: int
    family: str = "component_recolor"
    name: str = "outline_and_recolor_holed_components"

    @property
    def description_length(self) -> int:
        return 8 + self.view.description_length

    def execute(self, grid: Grid) -> Grid:
        scene = ComponentScene(grid, self.view)
        result = _copy(grid)
        for component in scene.components:
            if component.color != self.foreground:
                continue
            holes = _hole_cells(component.cells, component.bbox)
            if holes:
                for row, column in component.cells:
                    result[row][column] = self.holed_component_color
                for row, column in holes:
                    result[row][column] = self.hole_color
            # The relation is exterior adjacency, not a coordinate list.
            for row, column in component.cells:
                for dr, dc in EIGHT:
                    adjacent = row + dr, column + dc
                    if (
                        0 <= adjacent[0] < scene.height
                        and 0 <= adjacent[1] < scene.width
                        and grid[adjacent[0]][adjacent[1]] == self.view.background
                        and adjacent not in holes
                    ):
                        result[adjacent[0]][adjacent[1]] = self.outline_color
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "view": self.view.as_dict(),
            "selector": {
                "type": "component_topology",
                "foreground_color": self.foreground,
                "predicate": "hole_count>0",
            },
            "relations": ["eight_neighbor_exterior", "enclosed_background_region"],
            "operation": {
                "outline_color": self.outline_color,
                "holed_component_color": self.holed_component_color,
                "hole_color": self.hole_color,
            },
            "description_length": self.description_length,
        }


def _derive_topology_program(before: Grid, after: Grid, view: ViewSpec) -> TopologyRecolor | None:
    scene = ComponentScene(before, view)
    monochrome = [component for component in scene.components if component.color is not None]
    foregrounds = sorted({component.color for component in monochrome})
    for foreground in foregrounds:
        components = [component for component in monochrome if component.color == foreground]
        if not components or not any(component.hole_count for component in components):
            continue
        exterior_targets: list[int] = []
        holed_targets: list[int] = []
        hole_targets: list[int] = []
        for component in components:
            holes = _hole_cells(component.cells, component.bbox)
            if holes:
                holed_targets.extend(after[row][column] for row, column in component.cells)
                hole_targets.extend(after[row][column] for row, column in holes)
            for row, column in component.cells:
                for dr, dc in EIGHT:
                    adjacent = row + dr, column + dc
                    if (
                        0 <= adjacent[0] < scene.height
                        and 0 <= adjacent[1] < scene.width
                        and before[adjacent[0]][adjacent[1]] == view.background
                        and adjacent not in holes
                    ):
                        exterior_targets.append(after[adjacent[0]][adjacent[1]])
        if not (exterior_targets and holed_targets and hole_targets):
            continue
        if any(len(set(values)) != 1 for values in (exterior_targets, holed_targets, hole_targets)):
            continue
        return TopologyRecolor(
            view,
            foreground,
            exterior_targets[0],
            holed_targets[0],
            hole_targets[0],
        )
    return None


def search_component_recolor(
    train: Sequence[TrainPair],
) -> tuple[list[SceneProgram], dict[str, Any]]:
    if not all(_same_shape(pair) for pair in train):
        return [], {"family": "component_recolor", "reason": "requires_same_shape"}
    candidates: list[TopologyRecolor] = []
    for view in segmentation_views(train[0][0], include_multicolor=False):
        program = _derive_topology_program(*train[0], view)
        if program is not None:
            candidates.append(program)
    exact = [program for program in candidates if all(program.execute(a) == b for a, b in train)]
    dedup: dict[tuple[Any, ...], TopologyRecolor] = {}
    for program in exact:
        key = (
            program.foreground,
            program.outline_color,
            program.holed_component_color,
            program.hole_color,
            program.view.background,
            program.view.connectivity,
        )
        dedup.setdefault(key, program)
    return list(dedup.values()), {
        "family": "component_recolor",
        "generated": len(candidates),
        "train_exact": len(dedup),
        "reason": None if dedup else "no_shared_topology_selector",
    }


@dataclass(frozen=True)
class _Frame:
    bbox: BBox
    background: int
    frame_color: int


def _components_of_color(grid: Grid, color: int, connectivity: int = 8) -> list[set[Coord]]:
    height, width = len(grid), len(grid[0])
    unseen = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] == color
    }
    components: list[set[Coord]] = []
    directions = EIGHT if connectivity == 8 else CARDINAL
    while unseen:
        component = {unseen.pop()}
        queue = deque(component)
        while queue:
            row, column = queue.popleft()
            for dr, dc in directions:
                adjacent = row + dr, column + dc
                if adjacent in unseen:
                    unseen.remove(adjacent)
                    component.add(adjacent)
                    queue.append(adjacent)
        components.append(component)
    return components


def _detect_frame(grid: Grid) -> _Frame | None:
    height, width = len(grid), len(grid[0])
    candidates: list[tuple[int, _Frame]] = []
    for background, count in Counter(cell for row in grid for cell in row).most_common():
        for component in _components_of_color(grid, background, connectivity=4):
            rows = [cell[0] for cell in component]
            columns = [cell[1] for cell in component]
            interior = BBox(min(rows), max(rows) + 1, min(columns), max(columns) + 1)
            if len(component) != interior.area:
                continue
            r0, r1 = interior.r0 - 1, interior.r1
            c0, c1 = interior.c0 - 1, interior.c1
            if r0 < 0 or c0 < 0 or r1 >= height or c1 >= width:
                continue
            bbox = BBox(r0, r1 + 1, c0, c1 + 1)
            perimeter = [
                (row, column)
                for row, column in bbox.coordinates()
                if row in (bbox.r0, bbox.r1 - 1) or column in (bbox.c0, bbox.c1 - 1)
            ]
            values = [grid[row][column] for row, column in perimeter]
            if background in values:
                continue
            frame_color, support = Counter(values).most_common(1)[0]
            if support < len(values) // 2:
                continue
            candidates.append((interior.area, _Frame(bbox, background, frame_color)))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _stamp_transform(grid: Grid) -> Grid | None:
    frame = _detect_frame(grid)
    if frame is None:
        return None
    bbox = frame.bbox
    top_bottom: dict[int, set[int]] = {}
    left_right: dict[int, set[int]] = {}
    for column in range(bbox.c0 + 1, bbox.c1 - 1):
        for row in (bbox.r0, bbox.r1 - 1):
            color = grid[row][column]
            if color != frame.frame_color:
                top_bottom.setdefault(color, set()).add(column)
    for row in range(bbox.r0 + 1, bbox.r1 - 1):
        for column in (bbox.c0, bbox.c1 - 1):
            color = grid[row][column]
            if color != frame.frame_color:
                left_right.setdefault(color, set()).add(row)
    result = _copy(grid)
    stamped = 0
    marker_colors = sorted(set(top_bottom) & set(left_right))
    for color in marker_colors:
        source_components = [
            component
            for component in _components_of_color(grid, color, connectivity=8)
            if component
            and all(not bbox.contains(cell) for cell in component)
            and len(component) > 1
        ]
        if len(source_components) != 1:
            continue
        source = source_components[0]
        rows = [cell[0] for cell in source]
        columns = [cell[1] for cell in source]
        source_anchor = ((min(rows) + max(rows)) // 2, (min(columns) + max(columns)) // 2)
        for row in left_right[color]:
            for column in top_bottom[color]:
                translated = {
                    (row + sr - source_anchor[0], column + sc - source_anchor[1])
                    for sr, sc in source
                }
                if not all(
                    bbox.r0 < r < bbox.r1 - 1 and bbox.c0 < c < bbox.c1 - 1
                    for r, c in translated
                ):
                    return None
                for r, c in translated:
                    # Marker colors are traversed canonically.  A previous stamp
                    # wins at overlap, yielding one shared collision relation.
                    if result[r][c] == frame.background:
                        result[r][c] = color
                stamped += 1
        for row, column in source:
            result[row][column] = frame.background
    return result if stamped else None


@dataclass(frozen=True)
class MarkerStamp:
    family: str = "marker_indexed_stamp"
    name: str = "copy_external_shape_to_frame_marker_intersections"
    description_length: int = 10

    def execute(self, grid: Grid) -> Grid | None:
        return _stamp_transform(grid)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "view": {"connectivity": 8, "background": "enclosed_rectangle"},
            "selector": {
                "type": "relational",
                "predicate": "source_color_matches_row_and_column_frame_markers",
            },
            "relations": ["marker_row", "marker_column", "bbox_center_relative_offset"],
            "operation": "erase_source_then_translate_copy_to_intersection",
            "description_length": self.description_length,
        }


def search_marker_stamp(train: Sequence[TrainPair]) -> tuple[list[SceneProgram], dict[str, Any]]:
    if not all(_same_shape(pair) for pair in train):
        return [], {"family": "marker_indexed_stamp", "reason": "requires_same_shape"}
    program = MarkerStamp()
    exact = all(program.execute(before) == after for before, after in train)
    return ([program] if exact else []), {
        "family": program.family,
        "generated": 1,
        "train_exact": int(exact),
        "reason": None if exact else "frame_or_relative_offset_not_shared",
    }


def _periodic_repair(grid: Grid, axis: str, period: int) -> Grid | None:
    result = _copy(grid)
    line_count = len(grid) if axis == "rows" else len(grid[0])
    line_length = len(grid[0]) if axis == "rows" else len(grid)
    for line_index in range(line_count):
        line = (
            grid[line_index][:]
            if axis == "rows"
            else [grid[row][line_index] for row in range(len(grid))]
        )
        repaired = line[:]
        changed = False
        for phase in range(period):
            indexes = list(range(phase, line_length, period))
            counts = Counter(line[index] for index in indexes)
            if not counts:
                continue
            ordered = counts.most_common()
            if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
                return None
            value = ordered[0][0]
            for index in indexes:
                changed |= repaired[index] != value
                repaired[index] = value
        if changed:
            if axis == "rows":
                result[line_index] = repaired
            else:
                for row, value in enumerate(repaired):
                    result[row][line_index] = value
    return result


@dataclass(frozen=True)
class PeriodicRepair:
    axis: str
    period: int
    family: str = "periodic_repair"
    name: str = "repair_residue_classes_by_unique_majority"

    @property
    def description_length(self) -> int:
        return 7 + self.period

    def execute(self, grid: Grid) -> Grid | None:
        return _periodic_repair(grid, self.axis, self.period)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "selector": {"type": "periodic_residue", "axis": self.axis},
            "relation": {"period": self.period, "phase": "all_residue_classes"},
            "operation": "replace_outliers_with_unique_residue_majority",
            "loo_survival": True,
            "description_length": self.description_length,
        }


def search_periodic(train: Sequence[TrainPair]) -> tuple[list[SceneProgram], dict[str, Any]]:
    if not all(_same_shape(pair) for pair in train):
        return [], {"family": "periodic_repair", "reason": "requires_same_shape"}
    if len(train) < 3:
        return [], {
            "family": "periodic_repair",
            "generated": 0,
            "train_exact": 0,
            "ambiguous": True,
            "loo_survival": False,
            "reason": "requires_at_least_three_demonstrations_for_loo",
        }
    generated = [
        PeriodicRepair(axis, period)
        for axis in ("rows", "columns")
        for period in range(
            1,
            min(
                len(grid[0]) if axis == "rows" else len(grid)
                for grid, _ in train
            ),
        )
    ]
    exact = [
        program
        for program in generated
        if all(program.execute(before) == after for before, after in train)
    ]
    # Exact replay on every proper leave-one-out split is explicit even though
    # the shared parameter was obtained without an example index.
    survivors = [
        program
        for program in exact
        if all(
            all(program.execute(a) == b for index, (a, b) in enumerate(train) if index != held)
            for held in range(len(train))
        )
    ]
    if not survivors:
        return [], {
            "family": "periodic_repair",
            "generated": len(generated),
            "train_exact": len(exact),
            "ambiguous": False,
            "loo_survival": False,
            "reason": "no_train_exact_loo_stable_period",
        }
    best_mdl = min(program.description_length for program in survivors)
    best = [program for program in survivors if program.description_length == best_mdl]
    ambiguous = len(best) != 1
    return ([] if ambiguous else best), {
        "family": "periodic_repair",
        "generated": len(generated),
        "train_exact": len(exact),
        "loo_survival": True,
        "ambiguous": ambiguous,
        "reason": "multiple_equal_mdl_periods" if ambiguous else None,
        "surviving_periods": [
            {"axis": program.axis, "period": program.period, "mdl": program.description_length}
            for program in survivors
        ],
    }


PROGRAM_SEARCHERS = (
    search_component_recolor,
    search_local_motif,
    search_marker_stamp,
    search_periodic,
)
