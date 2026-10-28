"""Deterministic scene perception and relation-witness discovery."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from .geometry import D4, bbox_of, frame_coordinates, in_grid, offset_bounds
from .types import AffineD4, BBox, Grid, RelationWitness, validate_grid


@dataclass(frozen=True)
class RectangleProposal:
    bbox: BBox
    color: int
    area: int
    score: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.as_list(),
            "color": self.color,
            "area": self.area,
            "score": list(self.score),
        }


class Scene2D:
    """Perception view over one input grid; it never receives an output grid."""

    def __init__(self, grid: Grid):
        self.grid = validate_grid(grid)
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self._global_cache: dict[tuple[BBox, float], list[RelationWitness]] = {}
        self._local_cache: dict[tuple[BBox, int, int], list[RelationWitness]] = {}

    def solid_rectangles(self) -> list[RectangleProposal]:
        """Return maximal same-color connected components that are solid rectangles."""
        seen: set[tuple[int, int]] = set()
        proposals: list[RectangleProposal] = []
        color_counts = Counter(cell for row in self.grid for cell in row)
        for row in range(self.height):
            for column in range(self.width):
                if (row, column) in seen:
                    continue
                color = self.grid[row][column]
                queue = deque([(row, column)])
                seen.add((row, column))
                cells: list[tuple[int, int]] = []
                while queue:
                    current = queue.popleft()
                    cells.append(current)
                    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                        adjacent = current[0] + dr, current[1] + dc
                        if (
                            in_grid(self.grid, adjacent)
                            and adjacent not in seen
                            and self.grid[adjacent[0]][adjacent[1]] == color
                        ):
                            seen.add(adjacent)
                            queue.append(adjacent)
                bbox = bbox_of(cells)
                if len(cells) != bbox.area or min(bbox.height, bbox.width) < 2:
                    continue
                # Large, two-dimensional, globally unusual solids are favored.  No
                # color value or location is special.
                rarity = self.height * self.width - color_counts[color]
                score = (bbox.area, min(bbox.height, bbox.width), rarity, -color)
                proposals.append(RectangleProposal(bbox, color, bbox.area, score))
        return sorted(proposals, key=lambda proposal: proposal.score, reverse=True)

    def detect_occluder(self) -> RectangleProposal | None:
        proposals = self.solid_rectangles()
        if not proposals:
            return None
        minimum_area = max(4, (self.height * self.width) // 100)
        return next((proposal for proposal in proposals if proposal.area >= minimum_area), None)

    def global_relations(
        self,
        occluder: BBox,
        *,
        minimum_support_fraction: float = 0.35,
    ) -> list[RelationWitness]:
        """Enumerate broad, zero-error affine-D4 self-relations."""
        cache_key = occluder, minimum_support_fraction
        if cache_key in self._global_cache:
            return self._global_cache[cache_key][:]
        visible = self.height * self.width - occluder.area
        minimum_support = max(8, int(visible * minimum_support_fraction))
        witnesses: list[RelationWitness] = []
        for base in D4:
            row_offsets, column_offsets = offset_bounds(base.matrix, self.height, self.width)
            for row_offset in row_offsets:
                for column_offset in column_offsets:
                    # D4 matrices are signed permutations, so valid geometric
                    # overlap is separable by source axis.  Reject offsets that
                    # cannot possibly reach broad support before inspecting cells.
                    overlap = 1
                    for source_axis, (limit, offset) in enumerate(
                        ((self.height, row_offset), (self.width, column_offset))
                    ):
                        coefficients = base.matrix[source_axis]
                        target_axis = 0 if coefficients[0] else 1
                        sign = coefficients[target_axis]
                        target_limit = self.height if target_axis == 0 else self.width
                        count = sum(
                            0 <= sign * value + offset < limit
                            for value in range(target_limit)
                        )
                        overlap *= count
                    if overlap < minimum_support:
                        continue
                    if base.name == "identity" and row_offset == column_offset == 0:
                        continue
                    relation = base.with_offset((row_offset, column_offset))
                    support = 0
                    mismatch = 0
                    for row in range(self.height):
                        for column in range(self.width):
                            target = row, column
                            source = relation.apply(target)
                            if (
                                occluder.contains(target)
                                or not in_grid(self.grid, source)
                                or occluder.contains(source)
                            ):
                                continue
                            support += 1
                            if self.grid[row][column] != self.grid[source[0]][source[1]]:
                                mismatch += 1
                                break
                        if mismatch:
                            break
                    if mismatch == 0 and support >= minimum_support:
                        witnesses.append(
                            RelationWitness(relation, "global", support, 0)
                        )
        result = sorted(
            witnesses,
            key=lambda witness: (-witness.support, witness.relation.description()),
        )
        self._global_cache[cache_key] = result
        return result[:]

    def local_axis_swap_relations(
        self,
        occluder: BBox,
        *,
        frame_radius: int = 1,
        minimum_frame_support: int = 4,
    ) -> list[RelationWitness]:
        """Find swapped-coordinate source windows with an exact visible frame."""
        cache_key = occluder, frame_radius, minimum_frame_support
        if cache_key in self._local_cache:
            return self._local_cache[cache_key][:]
        frame = frame_coordinates(
            occluder, self.height, self.width, radius=frame_radius
        )
        witnesses: list[RelationWitness] = []
        for base in D4:
            if not base.swaps_axes:
                continue
            row_offsets, column_offsets = offset_bounds(base.matrix, self.height, self.width)
            for row_offset in row_offsets:
                for column_offset in column_offsets:
                    relation = base.with_offset((row_offset, column_offset))
                    source_coordinates = [
                        relation.apply(coordinate) for coordinate in occluder.coordinates()
                    ]
                    if not all(
                        in_grid(self.grid, source) and not occluder.contains(source)
                        for source in source_coordinates
                    ):
                        continue
                    support = 0
                    mismatch = 0
                    for target in frame:
                        source = relation.apply(target)
                        if not in_grid(self.grid, source) or occluder.contains(source):
                            continue
                        support += 1
                        if self.grid[target[0]][target[1]] != self.grid[source[0]][source[1]]:
                            mismatch += 1
                            break
                    if mismatch or support < minimum_frame_support:
                        continue
                    witnesses.append(
                        RelationWitness(
                            relation=relation,
                            kind="local_axis_swap",
                            support=occluder.area,
                            mismatch=0,
                            frame_support=support,
                            target_bbox=occluder,
                            source_bbox=bbox_of(source_coordinates),
                        )
                    )
        result = sorted(
            witnesses,
            key=lambda witness: (
                -witness.frame_support,
                witness.relation.description(),
            ),
        )
        self._local_cache[cache_key] = result
        return result[:]

    def diagnostics(self, occluder: RectangleProposal | None) -> dict[str, Any]:
        return {
            "shape": [self.height, self.width],
            "occluder": occluder.as_dict() if occluder else None,
            "rectangle_proposals": [
                proposal.as_dict() for proposal in self.solid_rectangles()[:10]
            ],
        }
