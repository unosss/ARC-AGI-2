from __future__ import annotations

import copy
import json
from pathlib import Path

from arc_solver.qwen2d_augment import stretch_grid, stretch_task
from kaggle_runs.qwen_stretch_lora_runpod import protocol, run_experiment, worker


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "kaggle_runs" / "qwen_stretch_lora_runpod" / "config.json"


def config():
    return json.loads(CONFIG_PATH.read_text())


def test_exact_and_capped_axis_repetition_preserve_cell_order():
    grid = [[1, 2], [3, 4]]
    assert stretch_grid(
        grid, repeat_rows=True, repeat_cols=False, max_dim=4
    ) == [[1, 2], [1, 2], [3, 4], [3, 4]]
    both = stretch_grid(grid, repeat_rows=True, repeat_cols=True, max_dim=3)
    assert len(both) == 3
    assert len(both[0]) == 3
    # Every added cell is a repetition, never a new color or interpolation.
    assert {cell for row in both for cell in row} == {1, 2, 3, 4}


def test_stretch_task_transforms_inputs_and_outputs_but_keeps_labels_blind():
    task = {
        "train": [{"input": [[1, 2]], "output": [[3, 4]]}],
        "test": [{"input": [[5, 6]], "output": [[7, 8]]}],
    }
    stretched = stretch_task(task, mode="cols", max_dim=4)
    assert stretched["train"][0]["input"] == [[1, 1, 2, 2]]
    assert stretched["train"][0]["output"] == [[3, 3, 4, 4]]
    assert stretched["test"][0]["input"] == [[5, 5, 6, 6]]
    assert "output" not in stretched["test"][0]


def test_paired_training_keeps_step_count_and_cycles_four_stretch_modes():
    value = protocol.mode_config(config(), "calibrate")
    task = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[0, 1], [2, 3]]},
            {"input": [[4, 5], [6, 7]], "output": [[4, 5], [6, 7]]},
        ],
        "test": [{"input": [[8, 9], [0, 1]]}],
    }
    control = list(worker.training_rows(task, "tiny", value, "control"))
    stretch = list(worker.training_rows(task, "tiny", value, "stretch"))
    assert len(control) == len(stretch) == value["task_local"]["steps"] == 4
    assert {row["stretch_mode"] for row in control} == {"identity"}
    assert [row["stretch_mode"] for row in stretch] == list(
        value["task_local"]["stretch_modes"]
    )
    assert all(len(row["input_ids"]) > 0 for row in stretch)


def test_config_is_frozen_and_worker_command_never_receives_solutions():
    value = config()
    protocol.validate_config(value)
    changed = copy.deepcopy(value)
    changed["task_local"]["stretch_max_dim"] = 29
    try:
        protocol.validate_config(changed)
    except ValueError as error:
        assert "stretch_max_dim" in str(error)
    else:
        raise AssertionError("changed stretch limit was accepted")
    command = run_experiment.worker_command(
        arm="stretch",
        mode="smoke",
        config_path=CONFIG_PATH,
        challenges_path=Path("/tmp/challenges.json"),
        task_ids=["195c6913"],
        work=Path("/tmp/work"),
        model_dir=None,
        adapter=None,
    )
    assert all("solution" not in argument.lower() for argument in command)


def test_smoke_arms_seal_then_grade(tmp_path):
    task_id = "195c6913"
    challenges = {
        task_id: {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
    }
    for arm in protocol.ARMS:
        worker.run_arm(arm, "smoke", config(), challenges, [task_id], tmp_path)
    report = protocol.grade(tmp_path, [task_id], {task_id: [[[2]]]})
    assert report["arms"]["control"]["aggregate"]["item_top2_exact"] == 1
    assert report["arms"]["stretch"]["aggregate"]["item_top2_exact"] == 1
    assert report["labels_opened_after_hash_barrier"] is True
