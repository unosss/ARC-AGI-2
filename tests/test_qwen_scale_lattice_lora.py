from __future__ import annotations

import json
from pathlib import Path

from kaggle_runs.qwen_scale_lattice_lora_runpod import protocol, worker


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "kaggle_runs" / "qwen_scale_lattice_lora_runpod" / "config.json"


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


def test_scale_lattice_has_four_exact_cycles_and_matched_control():
    value = protocol.mode_config(config(), "calibrate")
    control = list(worker.training_rows(tiny_task(), "tiny", value, "repeat_control"))
    lattice = list(worker.training_rows(tiny_task(), "tiny", value, "scale_lattice"))
    assert len(control) == len(lattice) == 16
    modes = [row["scale_mode"] for row in lattice]
    assert {mode: modes.count(mode) for mode in set(modes)} == {
        "identity": 4,
        "rows": 4,
        "cols": 4,
        "both": 4,
    }
    assert modes != ["identity"] * 4 + ["rows"] * 4 + ["cols"] * 4 + ["both"] * 4
    assert all(len(set(modes[offset : offset + 4])) >= 2 for offset in range(0, 16, 4))
    assert {row["scale_mode"] for row in control} == {"identity"}
    assert [
        (row["schedule_mode"], row["permutation_index"], row["geometry"])
        for row in control
    ] == [
        (row["schedule_mode"], row["permutation_index"], row["geometry"])
        for row in lattice
    ]
    for control_row, lattice_row in zip(control, lattice):
        if lattice_row["scale_mode"] == "identity":
            assert lattice_row["input_ids"] == control_row["input_ids"]
        else:
            assert len(lattice_row["input_ids"]) > len(control_row["input_ids"])
    both_shapes = {
        tuple(row["held_shape"])
        for row in lattice
        if row["scale_mode"] == "both"
    }
    assert both_shapes == {(4, 4)}


def test_scale_lattice_config_and_smoke_pair(tmp_path):
    value = config()
    protocol.validate_config(value)
    task_id = "20270e3b"
    challenges = {task_id: tiny_task()}
    for arm in protocol.ARMS:
        worker.run_arm(arm, "smoke", value, challenges, [task_id], tmp_path)
    report = protocol.grade(
        tmp_path,
        [task_id],
        {task_id: [tiny_task()["test"][0]["input"]]},
    )
    assert report["arms"]["repeat_control"]["aggregate"]["item_top2_exact"] == 1
    assert report["arms"]["scale_lattice"]["aggregate"]["item_top2_exact"] == 1
