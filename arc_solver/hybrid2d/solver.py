"""Train-validated deterministic hypothesis search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .component_programs import PROGRAM_SEARCHERS, SceneProgram
from .operations import OcclusionProgram, ReconstructionPolicy
from .scene import Scene2D
from .types import Candidate, Grid, validate_grid


DEFAULT_POLICIES: tuple[ReconstructionPolicy, ...] = (
    ReconstructionPolicy(
        "global_reflections_then_two_local_axis_witnesses",
        use_local_axis_swap=True,
        minimum_local_witnesses=2,
    ),
    ReconstructionPolicy(
        "global_reflections_then_local_axis_witness",
        use_local_axis_swap=True,
        minimum_local_witnesses=1,
    ),
    ReconstructionPolicy(
        "global_reflections_only",
        use_local_axis_swap=False,
        minimum_local_witnesses=0,
    ),
)


@dataclass(frozen=True)
class SolveResult:
    candidates: list[Grid]
    report: dict[str, Any]


def _validated_train_pairs(train_pairs: Sequence[dict[str, Any]]) -> list[tuple[Grid, Grid]]:
    if not train_pairs:
        raise ValueError("at least one training pair is required")
    validated = []
    for index, pair in enumerate(train_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"train[{index}] must be an object")
        validated.append(
            (
                validate_grid(pair.get("input"), name=f"train[{index}].input"),
                validate_grid(pair.get("output"), name=f"train[{index}].output"),
            )
        )
    return validated


def solve(
    train_pairs: Sequence[dict[str, Any]],
    test_input: Grid,
    *,
    policies: Sequence[ReconstructionPolicy] = DEFAULT_POLICIES,
    max_candidates: int = 2,
) -> SolveResult:
    """Solve from demonstrations and one test input.

    The API deliberately has no test-output parameter.  Every surviving policy
    must exactly reproduce every training output before it is run on the test input.
    """
    started = time.perf_counter()
    train = _validated_train_pairs(train_pairs)
    clean_test = validate_grid(test_input, name="test_input")
    train_scenes = [Scene2D(input_grid) for input_grid, _ in train]
    test_scene = Scene2D(clean_test)
    exact_programs: list[tuple[ReconstructionPolicy, list[dict[str, Any]]]] = []
    rejected_programs: list[dict[str, Any]] = []
    family_diagnostics: list[dict[str, Any]] = []

    scene_programs: list[SceneProgram] = []
    for searcher in PROGRAM_SEARCHERS:
        programs, diagnostics = searcher(train)
        family_diagnostics.append(diagnostics)
        scene_programs.extend(programs)

    for policy in policies:
        program = OcclusionProgram(policy)
        demonstrations: list[dict[str, Any]] = []
        all_exact = True
        for index, ((_, expected), scene) in enumerate(zip(train, train_scenes)):
            candidates, diagnostics = program.run(scene)
            exact = any(candidate.grid == expected for candidate in candidates)
            demonstrations.append(
                {
                    "index": index,
                    "exact": exact,
                    "candidate_count": len(candidates),
                    "diagnostics": diagnostics,
                }
            )
            all_exact &= exact
        if all_exact:
            exact_programs.append((policy, demonstrations))
        else:
            rejected_programs.append(
                {
                    "family": "affine_d4_occlusion_reconstruction",
                    "program": policy.name,
                    "reason": "failed_train_all_exact_gate",
                    "demonstrations": demonstrations,
                }
            )

    ranked: list[tuple[tuple[int, int, str], Grid, dict[str, Any]]] = []
    test_programs: list[dict[str, Any]] = []
    for policy, demonstrations in exact_programs:
        candidates, diagnostics = OcclusionProgram(policy).run(test_scene)
        test_programs.append(
            {
                "program": policy.name,
                "train": demonstrations,
                "candidate_count": len(candidates),
                "diagnostics": diagnostics,
            }
        )
        for candidate in candidates:
            detail = candidate.as_dict(0)
            detail.update(
                {
                    "family": "affine_d4_occlusion_reconstruction",
                    "description_length": policy.description_length,
                    "train_support": len(train),
                    "operations": [
                        "detect_solid_occluder",
                        "enumerate_affine_d4_witnesses",
                        "reconstruct_by_agreement",
                    ],
                }
            )
            ranked.append(
                (
                    (policy.description_length, -len(train), policy.name),
                    candidate.grid,
                    detail,
                )
            )

    for program in scene_programs:
        prediction = program.execute(clean_test)
        if prediction is None:
            family_diagnostics.append(
                {
                    "family": program.family,
                    "program": program.name,
                    "reason": "test_scene_preconditions_not_met",
                }
            )
            continue
        detail = {
            **program.as_dict(),
            "train_support": len(train),
            "operations": [program.name],
        }
        ranked.append(
            (
                (program.description_length, -len(train), program.name),
                prediction,
                detail,
            )
        )

    ranked.sort(key=lambda item: item[0])
    unique: list[tuple[Grid, dict[str, Any]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    equivalent_programs: dict[tuple[tuple[int, ...], ...], list[str]] = {}
    for _, grid, detail in ranked:
        key = tuple(tuple(row) for row in grid)
        equivalent_programs.setdefault(key, []).append(str(detail["name"] if "name" in detail else detail["program"]))
        if key not in seen:
            seen.add(key)
            unique.append((grid, detail))
        if len(unique) == max_candidates:
            break

    elapsed = time.perf_counter() - started
    return SolveResult(
        candidates=[grid for grid, _ in unique],
        report={
            "solver": "hybrid2d-v2-component-scene-programs",
            "train_all_exact_gate": True,
            "train_pair_count": len(train),
            "train_exact_program_count": len(exact_programs) + len(scene_programs),
            "train_exact_programs": [
                *[policy.name for policy, _ in exact_programs],
                *[program.name for program in scene_programs],
            ],
            "rejected_programs": rejected_programs,
            "family_diagnostics": family_diagnostics,
            "candidate_count": len(unique),
            "candidates": [
                {
                    "rank": rank,
                    "grid": grid,
                    **detail,
                    "equivalent_programs": equivalent_programs[
                        tuple(tuple(row) for row in grid)
                    ],
                }
                for rank, (grid, detail) in enumerate(unique, 1)
            ],
            "test_programs": test_programs,
            "abstained": not unique,
            "runtime_seconds": elapsed,
        },
    )


def solve_task(task: dict[str, Any], *, max_candidates: int = 2) -> list[SolveResult]:
    """Convenience entry point that strips any accidental test labels."""
    train = task.get("train")
    tests = task.get("test")
    if not isinstance(train, list) or not isinstance(tests, list):
        raise ValueError("task must contain train and test lists")
    results = []
    for index, item in enumerate(tests):
        if not isinstance(item, dict):
            raise ValueError(f"test[{index}] must be an object")
        # Only the input value crosses the solver boundary.
        results.append(solve(train, item.get("input"), max_candidates=max_candidates))
    return results
