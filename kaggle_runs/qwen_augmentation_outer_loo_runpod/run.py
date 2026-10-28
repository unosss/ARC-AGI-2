"""GPU runner for true train-pair outer-LOO augmentation gating."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter, hidden_size_from_config
from arc_solver.qwen2d_metadata import LayoutMetadata, build_layout
from arc_solver.qwen2d_runtime import (
    collate_token_rows,
    serialize_chatml_bounded,
    teacher_forced_metrics,
    train_fixed_steps,
)
from kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker import (
    _evaluate_trained_task_impl,
    load_base,
)
from kaggle_runs.qwen_augmentation_outer_loo_runpod.protocol import (
    MODES,
    outer_fold,
    training_rows,
)
from kaggle_runs.qwen_diarc_paired_runpod.protocol import atomic_json, grade_item
from kaggle_runs.qwen_diarc_paired_runpod.worker import configure_adapters


def held_metrics(model, structure, adaptation, held, config, held_index):
    tokens, start, _ = serialize_chatml_bounded(
        adaptation["train"],
        held["input"],
        held["output"],
        max_length=config["global_max_sequence_length"],
    )
    layout = LayoutMetadata.stack(
        [build_layout(tokens, arm="text_control", shuffle_seed=held_index)]
    )
    return teacher_forced_metrics(
        model,
        structure,
        torch.tensor([tokens], device="cuda"),
        layout,
        start,
        scale=0.0,
    )


def run_fold(config, task, mode, held_index, model_dir, adapter):
    adaptation, held = outer_fold(task, held_index)
    torch.manual_seed(config["seed"] + held_index)
    model, _ = load_base(model_dir)
    model = configure_adapters(model, "diarc", config, adapter)
    structure = (
        AdditiveStructureAdapter(hidden_size_from_config(model.config))
        .cuda()
        .to(torch.bfloat16)
        .freeze()
    )
    try:
        rows = training_rows(
            adaptation, config["task_id"], held_index, mode, config
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config["task_local"]["learning_rate"],
            weight_decay=0.0,
        )
        started = time.monotonic()
        losses = train_fixed_steps(
            model,
            structure,
            (
                collate_token_rows(
                    [row],
                    arm="text_control",
                    shuffle_seed=config["seed"] + index,
                )
                for index, row in enumerate(rows)
            ),
            optimizer=optimizer,
            steps=config["task_local"]["steps"],
            scale=0.0,
        )
        train_seconds = time.monotonic() - started
        teacher = held_metrics(
            model, structure, adaptation, held, config, held_index
        )
        generated = _evaluate_trained_task_impl(
            model,
            structure,
            adaptation,
            f"{config['task_id']}:outer:{held_index}",
            "text_control",
            config,
            pre={},
            post={},
            merge_for_inference=False,
        )
        candidates = generated["ranked_candidates"][0]
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank
        grade = grade_item(candidates, held["output"])
        return {
            "held_index": held_index,
            "mode": mode,
            "adaptation_pair_count": len(adaptation["train"]),
            "outer_label_used_in_training": False,
            "teacher_forced": teacher,
            "grade": grade,
            "candidate_count": len(candidates),
            "nodes_used": generated["nodes_used"],
            "model_forward_tokens": generated["model_forward_tokens"],
            "train_seconds": train_seconds,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
    finally:
        del structure, model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    challenges = json.loads(args.challenges.read_text())
    task = challenges[config["task_id"]]
    report = {
        "schema_version": 1,
        "experiment": 40,
        "task_id": config["task_id"],
        "modes": {},
    }
    for mode in MODES:
        folds = []
        for held_index in range(len(task["train"])):
            folds.append(
                run_fold(
                    config,
                    task,
                    mode,
                    held_index,
                    args.model_dir,
                    args.adapter,
                )
            )
            report["modes"][mode] = {"folds": folds}
            atomic_json(args.output, report)
        report["modes"][mode]["aggregate"] = {
            "mean_sum_nll": sum(
                fold["teacher_forced"]["sum_nll"] for fold in folds
            )
            / len(folds),
            "exact_in_pool": sum(
                fold["grade"]["truth_in_pool"] for fold in folds
            ),
            "top2_exact": sum(fold["grade"]["top2_exact"] for fold in folds),
            "mean_best_top2_cell_accuracy": sum(
                fold["grade"]["best_top2_cell_accuracy"] for fold in folds
            )
            / len(folds),
        }
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
