"""Experiment 041: full-demo paired panel for exact axis repetition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arc_solver.qwen2d_augment import augment_task, color_permutation, stretch_task_exact
from arc_solver.qwen2d_runtime import serialize_chatml_bounded
from kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker import stable_seed
from kaggle_runs.qwen_diarc_paired_runpod.protocol import atomic_json
from kaggle_runs.qwen_exact_repeat_lora_runpod import worker as base_worker
from kaggle_runs.qwen_exact_repeat_lora_runpod.protocol import (
    ARMS,
    candidate_path,
    grade,
    seal_candidate,
    verify_hash_barrier,
)
from kaggle_runs.qwen_exact_repeat_lora_runpod.worker import (
    assert_exact_repeat_compatible,
)


TASK_IDS = ("136b0064", "20270e3b")


def mixed_axis_rows(task, task_id, config, arm):
    local = config["task_local"]
    assert_exact_repeat_compatible(task, local["stretch_max_dim"])
    base_specs = [
        (permutation_index, geometry)
        for permutation_index in range(local["train_color_permutations"])
        for geometry in range(local["train_geometries"])
    ]
    specs = [
        (cycle, permutation_index, geometry)
        for cycle in range(2)
        for permutation_index, geometry in base_specs
    ]
    random.Random(stable_seed(f"{task_id}:axis-panel-schedule", config["seed"])).shuffle(specs)
    rows = []
    for cycle, permutation_index, geometry in specs:
        permutation = color_permutation(
            stable_seed(f"{task_id}:train:{permutation_index}", config["seed"])
        )
        augmented = augment_task(
            task,
            geometry=geometry,
            permutation=permutation,
            order_seed=stable_seed(
                f"{task_id}:order:{permutation_index}:{geometry}", config["seed"]
            ),
        )
        latent_mode = "identity" if cycle == 0 else (
            "rows" if (permutation_index * local["train_geometries"] + geometry) % 2 == 0
            else "cols"
        )
        applied_mode = latent_mode if arm == "exact_repeat" else "identity"
        if applied_mode != "identity":
            augmented["train"] = stretch_task_exact(
                {"train": augmented["train"], "test": []},
                mode=applied_mode,
                max_dim=local["stretch_max_dim"],
            )["train"]
        held = augmented["train"][-1]
        tokens, _, _ = serialize_chatml_bounded(
            augmented["train"][:-1],
            held["input"],
            held["output"],
            max_length=config["global_max_sequence_length"],
        )
        rows.append(
            {
                "input_ids": tokens,
                "cycle": cycle,
                "repeat_mode": applied_mode,
                "latent_mode": latent_mode,
                "permutation_index": permutation_index,
                "geometry": geometry,
                "held_shape": [len(held["input"]), len(held["input"][0])],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    challenges = json.loads(args.challenges.read_text())
    task_ids = tuple(args.task_id or TASK_IDS)
    blind = {task_id: challenges[task_id] for task_id in task_ids}
    if any("output" in item for task in blind.values() for item in task["test"]):
        raise ValueError("test outputs reached candidate worker")
    args.work.mkdir(parents=True, exist_ok=True)
    atomic_json(args.work / "label_blind_challenges.json", blind)
    base_worker.training_rows = mixed_axis_rows
    for task_id in task_ids:
        for arm in ARMS:
            base_worker.run_arm(
                arm,
                "hard-exploratory",
                config,
                {task_id: blind[task_id]},
                [task_id],
                args.work,
                model_dir=args.model_dir,
                adapter_source=args.adapter,
            )
    verify_hash_barrier(args.work, task_ids)
    solutions = json.loads(args.solutions.read_text())
    report = grade(args.work, list(task_ids), solutions)
    report["experiment"] = 41
    report["task_ids"] = list(task_ids)
    report["schedule"] = "deterministic_mixed_identity_rows_cols"
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
