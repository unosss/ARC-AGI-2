"""Explicit operation and program objects for occlusion reconstruction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .geometry import crop, in_grid
from .scene import Scene2D
from .types import BBox, Candidate, Grid, RelationWitness


@dataclass(frozen=True)
class DetectOccluder:
    def apply(self, scene: Scene2D) -> BBox | None:
        proposal = scene.detect_occluder()
        return proposal.bbox if proposal else None


@dataclass(frozen=True)
class ExtractTargetBBox:
    def apply(self, reconstructed: Grid, bbox: BBox) -> Grid:
        return crop(reconstructed, bbox)


@dataclass(frozen=True)
class ReconstructionPolicy:
    name: str
    minimum_global_support_fraction: float = 0.35
    use_local_axis_swap: bool = True
    minimum_local_witnesses: int = 2
    frame_radius: int = 1

    @property
    def description_length(self) -> int:
        return 1 + int(self.use_local_axis_swap) + self.frame_radius


@dataclass
class ReconstructViaWitnesses:
    policy: ReconstructionPolicy

    def _values(
        self, scene: Scene2D, bbox: BBox, witnesses: list[RelationWitness]
    ) -> dict[tuple[int, int], set[int]]:
        values: dict[tuple[int, int], set[int]] = defaultdict(set)
        for target in bbox.coordinates():
            for witness in witnesses:
                source = witness.relation.apply(target)
                if in_grid(scene.grid, source) and not bbox.contains(source):
                    values[target].add(scene.grid[source[0]][source[1]])
        return values

    def apply(self, scene: Scene2D, bbox: BBox) -> tuple[list[Candidate], dict[str, Any]]:
        global_witnesses = scene.global_relations(
            bbox,
            minimum_support_fraction=self.policy.minimum_global_support_fraction,
        )
        global_values = self._values(scene, bbox, global_witnesses)
        conflicts = {
            f"{row},{column}": sorted(values)
            for (row, column), values in global_values.items()
            if len(values) > 1
        }
        if conflicts:
            return [], {
                "status": "abstained_global_conflict",
                "global_witnesses": [w.as_dict() for w in global_witnesses],
                "conflicts": conflicts,
            }

        known = {
            coordinate: next(iter(values))
            for coordinate, values in global_values.items()
            if values
        }
        missing = [coordinate for coordinate in bbox.coordinates() if coordinate not in known]
        candidates: list[Candidate] = []
        local_witnesses: list[RelationWitness] = []
        grouped: dict[tuple[tuple[int, ...], ...], list[RelationWitness]] = defaultdict(list)

        if not missing:
            output = [
                [known[(row, column)] for column in range(bbox.c0, bbox.c1)]
                for row in range(bbox.r0, bbox.r1)
            ]
            candidates.append(
                Candidate(
                    output,
                    (
                        len(known),
                        len(global_witnesses),
                        sum(w.support for w in global_witnesses),
                        -self.policy.description_length,
                    ),
                    self.policy.name,
                    tuple(global_witnesses),
                )
            )
        elif self.policy.use_local_axis_swap:
            local_witnesses = scene.local_axis_swap_relations(
                bbox, frame_radius=self.policy.frame_radius
            )
            for witness in local_witnesses:
                values = dict(known)
                compatible = True
                for target in bbox.coordinates():
                    source = witness.relation.apply(target)
                    value = scene.grid[source[0]][source[1]]
                    if target in values and values[target] != value:
                        compatible = False
                        break
                    values[target] = value
                if not compatible:
                    continue
                prediction = tuple(
                    tuple(values[(row, column)] for column in range(bbox.c0, bbox.c1))
                    for row in range(bbox.r0, bbox.r1)
                )
                grouped[prediction].append(witness)

            for prediction, agreeing in grouped.items():
                if len(agreeing) < self.policy.minimum_local_witnesses:
                    continue
                output = [list(row) for row in prediction]
                candidates.append(
                    Candidate(
                        output,
                        (
                            len(known),
                            len(agreeing),
                            sum(w.frame_support for w in agreeing),
                            sum(w.support for w in global_witnesses),
                            -self.policy.description_length,
                        ),
                        self.policy.name,
                        tuple(global_witnesses + agreeing),
                    )
                )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates, {
            "status": "predicted" if candidates else "abstained_incomplete",
            "global_coverage": len(known),
            "target_area": bbox.area,
            "missing_before_local": len(missing),
            "global_witnesses": [w.as_dict() for w in global_witnesses],
            "local_witnesses": [w.as_dict() for w in local_witnesses],
            "local_prediction_groups": [
                {"agreement": len(group), "grid": [list(row) for row in prediction]}
                for prediction, group in grouped.items()
            ],
        }


@dataclass
class OcclusionProgram:
    policy: ReconstructionPolicy

    def run(self, grid: Grid | Scene2D) -> tuple[list[Candidate], dict[str, Any]]:
        scene = grid if isinstance(grid, Scene2D) else Scene2D(grid)
        proposal = scene.detect_occluder()
        if proposal is None:
            return [], {**scene.diagnostics(None), "status": "abstained_no_occluder"}
        candidates, reconstruction = ReconstructViaWitnesses(self.policy).apply(
            scene, proposal.bbox
        )
        return candidates, {
            **scene.diagnostics(proposal),
            "status": reconstruction["status"],
            "reconstruction": reconstruction,
        }
