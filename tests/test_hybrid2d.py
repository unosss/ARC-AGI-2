import inspect
import json
from pathlib import Path

import pytest

import arc_solver.hybrid2d as hybrid2d
from arc_solver.hybrid2d.geometry import D4, crop
from arc_solver.hybrid2d.io import evaluate_posthoc, load_challenges
from arc_solver.hybrid2d.operations import ReconstructViaWitnesses, ReconstructionPolicy
from arc_solver.hybrid2d.scene import Scene2D
from arc_solver.hybrid2d.solver import solve, solve_task
from arc_solver.hybrid2d.types import AffineD4, BBox, RelationWitness


ROOT = Path(__file__).resolve().parents[1]
CHALLENGES = (
    ROOT
    / "kaggle_runs/nvarc_single_pass_l4/output_v007/experiment_015"
    / "solver_input/challenges.json"
)
LABELS = (
    ROOT
    / "kaggle_runs/nvarc_single_pass_l4/output_v007/experiment_015"
    / "parent_finalizer_only/labels.json"
)


def _real_task():
    return json.loads(CHALLENGES.read_text())["0934a4d8"]


def test_bbox_crop_and_all_d4_matrices_are_well_typed():
    bbox = BBox(1, 3, 2, 5)
    assert (bbox.height, bbox.width, bbox.area) == (2, 3, 6)
    assert bbox.contains((2, 4))
    assert crop([[0, 1, 2, 3, 4]] * 4, bbox) == [[2, 3, 4], [2, 3, 4]]
    assert len(D4) == 8
    assert len({transform.matrix for transform in D4}) == 8
    assert next(transform for transform in D4 if transform.name == "transpose").apply(
        (2, 5)
    ) == (5, 2)


def test_synthetic_reflection_occlusion_is_detected_and_reconstructed():
    complete = [
        [1, 2, 3, 3, 2, 1],
        [4, 5, 6, 6, 5, 4],
        [7, 1, 2, 2, 1, 7],
        [7, 1, 2, 2, 1, 7],
        [4, 5, 6, 6, 5, 4],
        [1, 2, 3, 3, 2, 1],
    ]
    occluded = [row[:] for row in complete]
    for row in range(1, 3):
        for column in range(2, 4):
            occluded[row][column] = 8
    expected = [row[2:4] for row in complete[1:3]]
    result = solve([{"input": occluded, "output": expected}], occluded)
    assert result.candidates[0] == expected
    assert result.report["train_exact_program_count"] >= 1
    assert result.report["test_programs"][0]["diagnostics"]["occluder"]["color"] == 8


def test_synthetic_transpose_source_has_zero_error_context_frame():
    grid = [[(row * 6 + column) % 7 + 1 for column in range(6)] for row in range(6)]
    bbox = BBox(2, 4, 0, 2)
    transpose = next(transform for transform in D4 if transform.name == "transpose")
    for target in ((1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (4, 0), (4, 1), (4, 2)):
        source = transpose.apply(target)
        if not bbox.contains(source):
            grid[target[0]][target[1]] = grid[source[0]][source[1]]
    for row, column in bbox.coordinates():
        grid[row][column] = 8
    witnesses = Scene2D(grid).local_axis_swap_relations(bbox)
    matching = [
        witness
        for witness in witnesses
        if witness.relation.name == "transpose" and witness.relation.offset == (0, 0)
    ]
    assert matching
    assert matching[0].source_bbox == BBox(0, 2, 2, 4)
    assert matching[0].mismatch == 0


def test_conflicting_global_witnesses_force_abstention(monkeypatch):
    grid = [
        [1, 2, 3, 4],
        [5, 8, 8, 6],
        [7, 8, 8, 9],
        [1, 2, 3, 4],
    ]
    scene = Scene2D(grid)
    bbox = BBox(1, 3, 1, 3)
    witnesses = [
        RelationWitness(AffineD4("left", ((1, 0), (0, 1)), (0, -1)), "global", 8, 0),
        RelationWitness(AffineD4("up", ((1, 0), (0, 1)), (-1, 0)), "global", 8, 0),
    ]
    monkeypatch.setattr(scene, "global_relations", lambda *args, **kwargs: witnesses)
    candidates, diagnostics = ReconstructViaWitnesses(
        ReconstructionPolicy("conflict", use_local_axis_swap=False)
    ).apply(scene, bbox)
    assert candidates == []
    assert diagnostics["status"] == "abstained_global_conflict"
    assert diagnostics["conflicts"]


def test_train_all_exact_gate_rejects_nonmatching_programs():
    task = _real_task()
    bad_train = [
        {"input": pair["input"], "output": [[0 for _ in row] for row in pair["output"]]}
        for pair in task["train"]
    ]
    result = solve(bad_train, task["test"][0]["input"])
    assert result.candidates == []
    assert result.report["train_exact_program_count"] == 0


def test_0934_train_only_search_discovers_geometry_and_programs():
    task = _real_task()
    result = solve(task["train"], task["test"][0]["input"])
    assert result.report["train_exact_program_count"] == 3
    assert result.report["candidate_count"] == 1
    diagnostic = result.report["test_programs"][0]["diagnostics"]
    assert diagnostic["occluder"]["bbox"] == [14, 23, 0, 3]
    reconstruction = diagnostic["reconstruction"]
    assert reconstruction["global_coverage"] == 19
    assert len(reconstruction["local_witnesses"]) == 2
    assert {
        tuple(witness["source_bbox"]) for witness in reconstruction["local_witnesses"]
    } == {(0, 3, 9, 18), (0, 3, 14, 23)}


def test_0934_posthoc_oracle_score_is_top1_exact():
    task = _real_task()
    truth = json.loads(LABELS.read_text())["0934a4d8"][0]
    result = solve(task["train"], task["test"][0]["input"])
    evaluation = evaluate_posthoc(
        {"0934a4d8": [[*result.candidates]]}, {"0934a4d8": [truth]}
    )
    assert evaluation["item_top1_exact"] == 1
    assert evaluation["item_top2_exact"] == 1


def test_solver_never_reads_a_test_output():
    class Spy(dict):
        output_reads = 0

        def get(self, key, default=None):
            if key == "output":
                type(self).output_reads += 1
                raise AssertionError("test output was read")
            return super().get(key, default)

        def __getitem__(self, key):
            if key == "output":
                type(self).output_reads += 1
                raise AssertionError("test output was read")
            return super().__getitem__(key)

    task = _real_task()
    spy = Spy(input=task["test"][0]["input"], output=[[9]])
    results = solve_task({"train": task["train"], "test": [spy]})
    assert results[0].candidates
    assert Spy.output_reads == 0


def test_loader_strips_test_outputs_before_solver_boundary(tmp_path):
    task = _real_task()
    task["test"][0]["output"] = [[9]]
    path = tmp_path / "challenge.json"
    path.write_text(json.dumps({"x": task}))
    clean = load_challenges(path)
    assert "output" not in clean["x"]["test"][0]


def test_production_package_has_no_task_specific_or_oracle_imports():
    package = Path(inspect.getfile(hybrid2d)).parent
    source = "\n".join(path.read_text() for path in package.glob("*.py"))
    assert "0934a4d8" not in source
    assert "nvarc_stage_diagnosis" not in source
    assert "SOURCE_BBOX" not in source
