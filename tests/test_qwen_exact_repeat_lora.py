from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc_solver.qwen2d_augment import stretch_grid_exact, stretch_task_exact
from kaggle_runs.qwen_exact_repeat_lora_runpod import protocol, worker


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "kaggle_runs" / "qwen_exact_repeat_lora_runpod" / "config.json"


def config():
    return json.loads(CONFIG_PATH.read_text())


def tiny_task():
    return {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
            {"input": [[2, 3], [4, 5]], "output": [[6, 7], [8, 9]]},
        ],
        "test": [{"input": [[0, 1], [1, 0]]}],
    }


def test_exact_repeat_duplicates_every_row_and_column():
    grid = [[1, 2], [3, 4]]
    rows = stretch_grid_exact(grid, repeat_rows=True, repeat_cols=False)
    cols = stretch_grid_exact(grid, repeat_rows=False, repeat_cols=True)
    assert rows == [[1, 2], [1, 2], [3, 4], [3, 4]]
    assert cols == [[1, 1, 2, 2], [3, 3, 4, 4]]
    both = stretch_task_exact(tiny_task(), mode="both")
    assert len(both["train"][0]["input"]) == 4
    assert len(both["train"][0]["input"][0]) == 4
    assert "output" not in both["test"][0]


def test_exact_repeat_rejects_instead_of_partially_warping():
    grid = [[0] * 16 for _ in range(2)]
    with pytest.raises(ValueError, match="would exceed 30"):
        stretch_grid_exact(grid, repeat_rows=False, repeat_cols=True)
    incompatible = {
        "train": [{"input": grid, "output": grid}],
        "test": [{"input": [[0]]}],
    }
    with pytest.raises(ValueError, match="incompatible"):
        worker.assert_exact_repeat_compatible(incompatible, 30)


def test_matched_schedule_keeps_original_cycle_and_changes_only_added_cycle():
    value = protocol.mode_config(config(), "calibrate")
    control = list(worker.training_rows(tiny_task(), "tiny", value, "repeat_control"))
    exact = list(worker.training_rows(tiny_task(), "tiny", value, "exact_repeat"))
    assert len(control) == len(exact) == 8
    assert [row["input_ids"] for row in control[:4]] == [
        row["input_ids"] for row in exact[:4]
    ]
    assert {row["repeat_mode"] for row in exact[:4]} == {"identity"}
    assert [row["repeat_mode"] for row in exact[4:]] == [
        "rows",
        "cols",
        "rows",
        "cols",
    ]
    assert {row["repeat_mode"] for row in control} == {"identity"}
    assert all(
        len(exact[index]["input_ids"]) > len(control[index]["input_ids"])
        for index in range(4, 8)
    )


def test_smoke_pair_seals_and_grades(tmp_path):
    task_id = "20270e3b"
    challenges = {task_id: tiny_task()}
    for arm in protocol.ARMS:
        worker.run_arm(arm, "smoke", config(), challenges, [task_id], tmp_path)
    report = protocol.grade(
        tmp_path,
        [task_id],
        {task_id: [tiny_task()["test"][0]["input"]]},
    )
    assert report["arms"]["repeat_control"]["aggregate"]["item_top2_exact"] == 1
    assert report["arms"]["exact_repeat"]["aggregate"]["item_top2_exact"] == 1
