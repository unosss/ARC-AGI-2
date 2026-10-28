from __future__ import annotations

import json
from pathlib import Path

from kaggle_runs.qwen_axis_panel_runpod.run import mixed_axis_rows


ROOT = Path(__file__).parents[1]
CONFIG = json.loads(
    (ROOT / "kaggle_runs/qwen_exact_repeat_lora_runpod/config.json").read_text()
)


def test_panel_schedule_is_matched_mixed_and_full_demo():
    config = json.loads(json.dumps(CONFIG))
    config["task_local"]["train_color_permutations"] = 2
    config["task_local"]["train_geometries"] = 2
    task = {
        "train": [
            {"input": [[index]], "output": [[index + 1]]}
            for index in range(4)
        ],
        "test": [{"input": [[9]]}],
    }
    control = mixed_axis_rows(task, "tiny", config, "repeat_control")
    exact = mixed_axis_rows(task, "tiny", config, "exact_repeat")
    assert len(control) == len(exact) == 8
    spec = lambda row: (
        row["cycle"],
        row["latent_mode"],
        row["permutation_index"],
        row["geometry"],
    )
    assert [spec(row) for row in control] == [spec(row) for row in exact]
    assert all(row["repeat_mode"] == "identity" for row in control)
    assert sum(row["repeat_mode"] == "rows" for row in exact) == 2
    assert sum(row["repeat_mode"] == "cols" for row in exact) == 2
    assert [row["cycle"] for row in exact] != [0] * 4 + [1] * 4
