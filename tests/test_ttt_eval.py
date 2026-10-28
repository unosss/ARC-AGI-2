import copy

import pytest

from arc_solver.ttt_eval import (
    build_phase1_bundle,
    compare_conditions,
    hide_test_outputs,
    inverse_transform_grid,
    is_valid_grid,
    score_condition,
    select_task_ids,
    transform_grid,
)


GRID = [[0, 1, 2], [3, 4, 5]]


def test_valid_grid_rejects_invalid_arc_outputs():
    assert is_valid_grid(GRID)
    assert not is_valid_grid([])
    assert not is_valid_grid([[0], [0, 1]])
    assert not is_valid_grid([[10]])
    assert not is_valid_grid([[True]])


def test_hide_test_outputs_does_not_mutate_source():
    task = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[3]]}],
    }
    public = hide_test_outputs(task)

    assert public["test"] == [{"input": [[2]]}]
    assert task["test"][0]["output"] == [[3]]


def test_bundle_separates_labels_from_solver_input():
    task = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]]}],
    }
    public, labels = build_phase1_bundle(
        {"abc": task}, {"abc": [[[3]]]}, ["abc"]
    )

    assert "output" not in public["abc"]["test"][0]
    assert labels == {"abc": [[[3]]]}


def test_selection_is_stable_and_excludes_requested_ids():
    challenges = {str(index): {} for index in range(30)}
    first = select_task_ids(challenges, count=5, seed=7, excluded={"0", "1"})
    second = select_task_ids(
        copy.deepcopy(challenges), count=5, seed=7, excluded={"0", "1"}
    )

    assert first == second
    assert not set(first) & {"0", "1"}


@pytest.mark.parametrize(
    "name",
    [
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "hmirror",
        "vmirror",
        "dmirror",
        "cmirror",
    ],
)
def test_d4_transforms_round_trip(name):
    transformed = transform_grid(GRID, name)
    assert inverse_transform_grid(transformed, name) == GRID


def test_score_condition_uses_top_two_and_requires_all_task_items():
    solutions = {"abc": [[[1]], [[2]]], "def": [[[3]]]}
    submission = {
        "abc": [
            {"attempt_1": [[0]], "attempt_2": [[1]]},
            {"attempt_1": [[2]], "attempt_2": [[0]]},
        ],
        "def": [{"attempt_1": [[0]], "attempt_2": "invalid"}],
    }
    score = score_condition(submission, solutions)

    assert score["task_exact"] == 1
    assert score["item_exact"] == 2
    assert score["item_count"] == 3
    assert score["valid_grid_rate"] == pytest.approx(5 / 6)


def test_compare_conditions_uses_identical_task_set():
    solutions = {"abc": [[[1]]]}
    baseline = {"abc": [{"attempt_1": [[0]], "attempt_2": [[0]]}]}
    ttt = {"abc": [{"attempt_1": [[1]], "attempt_2": [[0]]}]}

    report = compare_conditions({"baseline": baseline, "ttt": ttt}, solutions)

    assert report["task_ids"] == ["abc"]
    assert report["conditions"]["baseline"]["task_exact"] == 0
    assert report["conditions"]["ttt"]["task_exact"] == 1
