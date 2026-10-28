from __future__ import annotations

import json
from pathlib import Path

from kaggle_runs.qwen_augmentation_outer_loo_runpod.protocol import (
    outer_fold,
    training_rows,
)


ROOT = Path(__file__).parents[1]
CONFIG = json.loads(
    (ROOT / "kaggle_runs/qwen_augmentation_outer_loo_runpod/config.json").read_text()
)


def task():
    return {
        "train": [
            {"input": [[index]], "output": [[index + 4]]}
            for index in range(4)
        ],
        "test": [],
    }


def test_outer_fold_removes_held_input_and_output_completely():
    adaptation, held = outer_fold(task(), 2)
    assert held == {"input": [[2]], "output": [[6]]}
    assert held not in adaptation["train"]
    assert all(pair["input"] != held["input"] for pair in adaptation["train"])
    assert adaptation["test"] == [{"input": [[2]]}]
    assert "output" not in adaptation["test"][0]


def test_each_mode_has_matched_shuffled_rows_without_outer_label():
    adaptation, held = outer_fold(task(), 2)
    config = json.loads(json.dumps(CONFIG))
    config["task_local"]["train_color_permutations"] = 2
    config["task_local"]["train_geometries"] = 2
    config["task_local"]["steps"] = 8
    control = training_rows(adaptation, "tiny", 2, "control", config)
    both = training_rows(adaptation, "tiny", 2, "both", config)
    assert len(control) == len(both) == 8
    assert all(row["source_pair_count"] == 3 for row in both)
    assert all(row["outer_held_index"] == 2 for row in both)
    assert sum(row["applied_mode"] == "both" for row in both) == 4
    assert all(row["applied_mode"] == "identity" for row in control)
    held_tokens = {cell for row in held["output"] for cell in row}
    assert held_tokens == {6}
