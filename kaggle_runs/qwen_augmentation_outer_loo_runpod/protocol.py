"""True outer-LOO data construction for Experiment 040."""

from __future__ import annotations

import copy
import random
from typing import Any

from arc_solver.qwen2d_augment import (
    augment_task,
    color_permutation,
    stretch_task_exact,
)
from arc_solver.qwen2d_runtime import serialize_chatml_bounded
from kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker import stable_seed
from kaggle_runs.qwen_exact_repeat_lora_runpod.worker import (
    assert_exact_repeat_compatible,
)


MODES = ("control", "rows", "cols", "both")


def outer_fold(task: dict[str, Any], held_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = task["train"]
    if not 0 <= held_index < len(pairs):
        raise ValueError("held_index outside train pairs")
    adaptation = {
        "train": copy.deepcopy(
            [pair for index, pair in enumerate(pairs) if index != held_index]
        ),
        "test": [{"input": copy.deepcopy(pairs[held_index]["input"])}],
    }
    return adaptation, copy.deepcopy(pairs[held_index])


def training_rows(
    adaptation: dict[str, Any],
    task_id: str,
    held_index: int,
    mode: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    local = config["task_local"]
    assert_exact_repeat_compatible(adaptation, local["stretch_max_dim"])
    specs = [
        (cycle, permutation_index, geometry)
        for cycle in range(2)
        for permutation_index in range(local["train_color_permutations"])
        for geometry in range(local["train_geometries"])
    ]
    random.Random(
        stable_seed(f"{task_id}:outer:{held_index}:schedule", config["seed"])
    ).shuffle(specs)
    rows = []
    for cycle, permutation_index, geometry in specs:
        permutation = color_permutation(
            stable_seed(f"{task_id}:train:{permutation_index}", config["seed"])
        )
        augmented = augment_task(
            adaptation,
            geometry=geometry,
            permutation=permutation,
            order_seed=stable_seed(
                f"{task_id}:outer:{held_index}:order:{permutation_index}:{geometry}",
                config["seed"],
            ),
        )
        applied = "identity"
        if cycle == 1 and mode != "control":
            applied = mode
            train_only = {"train": augmented["train"], "test": []}
            augmented["train"] = stretch_task_exact(
                train_only,
                mode=mode,
                max_dim=local["stretch_max_dim"],
            )["train"]
        inner_held = augmented["train"][-1]
        tokens, _, _ = serialize_chatml_bounded(
            augmented["train"][:-1],
            inner_held["input"],
            inner_held["output"],
            max_length=config["global_max_sequence_length"],
        )
        rows.append(
            {
                "input_ids": tokens,
                "cycle": cycle,
                "applied_mode": applied,
                "source_pair_count": len(adaptation["train"]),
                "outer_held_index": held_index,
            }
        )
    if len(rows) != local["steps"]:
        raise AssertionError("outer-LOO training row count differs from steps")
    return rows
