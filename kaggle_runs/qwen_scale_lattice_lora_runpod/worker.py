"""One-arm worker for the exact integer scale-lattice comparison."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter, hidden_size_from_config
from arc_solver.qwen2d_augment import (
    augment_task,
    color_permutation,
    stretch_task_exact,
)
from arc_solver.qwen2d_runtime import (
    collate_token_rows,
    serialize_chatml_bounded,
    train_fixed_steps,
)
from kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker import (
    _evaluate_trained_task_impl,
    load_base,
    stable_seed,
    train_diagnostic,
)
from kaggle_runs.qwen_diarc_paired_runpod.worker import (
    active_adapter_names,
    configure_adapters,
)
from kaggle_runs.qwen_exact_repeat_lora_runpod.worker import (
    assert_exact_repeat_compatible,
)
from kaggle_runs.qwen_scale_lattice_lora_runpod.protocol import (
    ARMS,
    candidate_path,
    mode_config,
    reject_test_outputs,
    seal_candidate,
    validate_config,
)


def training_rows(
    task: dict[str, Any],
    task_id: str,
    config: dict[str, Any],
    arm: str,
):
    local = config["task_local"]
    seed = config["seed"]
    assert_exact_repeat_compatible(task, local["stretch_max_dim"])
    schedule = [
        (cycle, treatment_mode, permutation_index, geometry)
        for cycle, treatment_mode in enumerate(local["scale_modes"])
        for permutation_index in range(local["train_color_permutations"])
        for geometry in range(local["train_geometries"])
    ]
    random.Random(stable_seed(f"{task_id}:scale-schedule", seed)).shuffle(schedule)
    for cycle, treatment_mode, permutation_index, geometry in schedule:
        mode = treatment_mode if arm == "scale_lattice" else "identity"
        permutation = color_permutation(
            stable_seed(f"{task_id}:train:{permutation_index}", seed)
        )
        augmented = augment_task(
            task,
            geometry=geometry,
            permutation=permutation,
            order_seed=stable_seed(
                f"{task_id}:order:{permutation_index}:{geometry}", seed
            ),
        )
        if mode != "identity":
            train_only = {"train": augmented["train"], "test": []}
            augmented["train"] = stretch_task_exact(
                train_only,
                mode=mode,
                max_dim=local["stretch_max_dim"],
            )["train"]
        held = augmented["train"][-1]
        tokens, _, _ = serialize_chatml_bounded(
            augmented["train"][:-1],
            held["input"],
            held["output"],
            max_length=config["global_max_sequence_length"],
        )
        yield {
            "input_ids": tokens,
            "cycle": cycle,
            "schedule_mode": treatment_mode,
            "scale_mode": mode,
            "permutation_index": permutation_index,
            "geometry": geometry,
            "held_shape": [len(held["input"]), len(held["input"][0])],
        }


def _train_task(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    task: dict[str, Any],
    task_id: str,
    config: dict[str, Any],
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    pre = train_diagnostic(model, structure, task, task_id, "text_control", config)
    rows = list(training_rows(task, task_id, config, arm))
    steps = config["task_local"]["steps"]
    if len(rows) != steps:
        raise RuntimeError(f"task rows {len(rows)} != registered steps {steps}")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["task_local"]["learning_rate"],
        weight_decay=0.0,
    )
    started = time.monotonic()
    train_fixed_steps(
        model,
        structure,
        (
            collate_token_rows(
                [row], arm="text_control", shuffle_seed=config["seed"] + index
            )
            for index, row in enumerate(rows)
        ),
        optimizer=optimizer,
        steps=steps,
        scale=0.0,
    )
    elapsed = time.monotonic() - started
    post = train_diagnostic(model, structure, task, task_id, "text_control", config)
    manifest = {
        "row_count": len(rows),
        "schedule_order": config["task_local"]["schedule_order"],
        "cycle_counts": {
            str(cycle): sum(row["cycle"] == cycle for row in rows)
            for cycle in range(len(config["task_local"]["scale_modes"]))
        },
        "mode_counts": {
            mode: sum(row["scale_mode"] == mode for row in rows)
            for mode in config["task_local"]["scale_modes"]
        },
        "mean_sequence_tokens": sum(len(row["input_ids"]) for row in rows) / len(rows),
        "max_sequence_tokens": max(len(row["input_ids"]) for row in rows),
        "held_shapes": sorted({tuple(row["held_shape"]) for row in rows}),
    }
    return pre, post, elapsed, manifest


def smoke_payload(arm: str, task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    visible = [pair["output"] for pair in task.get("train", [])]
    ranked = []
    for item in task.get("test", []):
        grids = []
        for grid in [item["input"], *visible]:
            if grid not in grids:
                grids.append(grid)
        ranked.append(
            [
                {"grid": grid, "score": -float(rank), "votes": 1, "rank": rank}
                for rank, grid in enumerate(grids[:8], 1)
            ]
        )
    unavailable = {"canonical_loo_mean_sum_nll": None, "folds": 0}
    return {
        "schema_version": 1,
        "experiment": 39,
        "mode": "smoke",
        "experiment_arm": arm,
        "task_id": task_id,
        "pre_task_loo_nll": unavailable,
        "post_task_loo_nll": unavailable,
        "ranked_candidates": ranked,
        "timing": {"load_seconds": 0.0, "train_seconds": 0.0, "decode_seconds": 0.0},
        "peak_memory_bytes": 0,
        "training_manifest": {},
    }


def run_arm(
    arm: str,
    mode: str,
    config: dict[str, Any],
    challenges: dict[str, Any],
    task_ids: list[str],
    work: Path,
    *,
    model_dir: Path | None = None,
    adapter_source: str | Path | None = None,
) -> None:
    reject_test_outputs(challenges)
    effective = mode_config(config, mode)
    if mode == "smoke":
        for task_id in task_ids:
            seal_candidate(
                candidate_path(work, arm, task_id),
                smoke_payload(arm, task_id, challenges[task_id]),
            )
        return
    if model_dir is None:
        raise ValueError("model_dir is required outside smoke mode")

    load_started = time.monotonic()
    torch.manual_seed(config["seed"])
    model, _ = load_base(model_dir)
    model = configure_adapters(
        model,
        "diarc",
        effective,
        adapter_source or config["diarc_adapter"],
    )
    structure = (
        AdditiveStructureAdapter(hidden_size_from_config(model.config))
        .cuda()
        .to(torch.bfloat16)
        .freeze()
    )
    load_seconds = time.monotonic() - load_started
    try:
        for task_id in task_ids:
            torch.cuda.reset_peak_memory_stats()
            pre, post, train_seconds, manifest = _train_task(
                model,
                structure,
                challenges[task_id],
                task_id,
                effective,
                arm,
            )
            decode_started = time.monotonic()
            payload = _evaluate_trained_task_impl(
                model,
                structure,
                challenges[task_id],
                task_id,
                "text_control",
                effective,
                pre=pre,
                post=post,
                merge_for_inference=False,
            )
            decode_seconds = time.monotonic() - decode_started
            for candidates in payload["ranked_candidates"]:
                for rank, candidate in enumerate(candidates, 1):
                    candidate["rank"] = rank
            payload.update(
                {
                    "schema_version": 1,
                    "experiment": 39,
                    "mode": mode,
                    "experiment_arm": arm,
                    "active_adapters": list(active_adapter_names(model)),
                    "pre_task_loo_nll": pre,
                    "post_task_loo_nll": post,
                    "training_manifest": manifest,
                    "timing": {
                        "load_seconds": load_seconds,
                        "train_seconds": train_seconds,
                        "decode_seconds": decode_seconds,
                    },
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
            seal_candidate(candidate_path(work, arm, task_id), payload)
    finally:
        if hasattr(model, "delete_adapter"):
            model.delete_adapter("task")
            model.delete_adapter("diarc")
        if hasattr(model, "unload"):
            model.unload()
        del structure, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "calibrate", "hard-exploratory"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    challenges = json.loads(args.challenges.read_text())
    run_arm(
        args.arm,
        args.mode,
        config,
        challenges,
        args.task_id,
        args.work,
        model_dir=args.model_dir,
        adapter_source=args.adapter,
    )


if __name__ == "__main__":
    main()
