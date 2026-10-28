"""Compare sequential and length-exact batched rescoring after one DFS run."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

import kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker as worker
from arc_solver.qwen2d_adapter import AdditiveStructureAdapter, hidden_size_from_config


def synchronize() -> None:
    torch.cuda.synchronize()


def grid_key(grid: list[list[int]]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def score_pool(
    task_model: Any,
    structure: Any,
    task: dict[str, Any],
    task_id: str,
    arm: str,
    config: dict[str, Any],
    pool: list[list[dict[str, Any]]],
    batch_size: int,
) -> tuple[list[list[dict[str, Any]]], float]:
    mode_config = copy.deepcopy(config)
    mode_config["task_local"]["rescore_batch_size"] = batch_size
    synchronize()
    started = time.perf_counter()
    scored_items = []
    for test_index, candidates in enumerate(pool):
        scored = []
        for candidate in candidates:
            rescore = worker.score_candidate_views(
                task_model,
                structure,
                task,
                task_id,
                test_index,
                candidate["grid"],
                arm,
                mode_config,
            )
            scored.append(
                {
                    **candidate,
                    "score": candidate["votes"] - rescore["mean_sum_nll"],
                    "rescore": rescore,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        scored_items.append(scored)
    synchronize()
    return scored_items, time.perf_counter() - started


def compare(
    sequential: list[list[dict[str, Any]]],
    batched: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    metric_differences: dict[str, list[float]] = {
        "sum_nll": [],
        "mean_nll": [],
        "first_token_nll": [],
        "per_row_mean_nll": [],
    }
    metric_schema_exact = True
    token_counts_exact = True
    per_row_ids_exact = True
    per_candidate = []
    for test_index, (left_item, right_item) in enumerate(zip(sequential, batched)):
        left = {grid_key(item["grid"]): item for item in left_item}
        right = {grid_key(item["grid"]): item for item in right_item}
        if left.keys() != right.keys():
            raise AssertionError("rescore modes did not use the same candidate pool")
        for key in left:
            left_views = left[key]["rescore"]["views"]
            right_views = right[key]["rescore"]["views"]
            if len(left_views) != len(right_views):
                raise AssertionError("rescore view count changed")
            view_differences = []
            for left_view, right_view in zip(left_views, right_views):
                metric_schema_exact &= left_view.keys() == right_view.keys()
                token_counts_exact &= (
                    left_view["token_count"] == right_view["token_count"]
                )
                left_rows = left_view["per_row_nll"]
                right_rows = right_view["per_row_nll"]
                per_row_ids_exact &= [row["row"] for row in left_rows] == [
                    row["row"] for row in right_rows
                ]
                for metric in ("sum_nll", "mean_nll", "first_token_nll"):
                    delta = abs(left_view[metric] - right_view[metric])
                    metric_differences[metric].append(delta)
                for left_row, right_row in zip(left_rows, right_rows):
                    metric_differences["per_row_mean_nll"].append(
                        abs(left_row["mean_nll"] - right_row["mean_nll"])
                    )
                view_differences.append(
                    {
                        metric: metric_differences[metric][-1]
                        for metric in ("sum_nll", "mean_nll", "first_token_nll")
                    }
                )
            mean_delta = abs(
                left[key]["rescore"]["mean_sum_nll"]
                - right[key]["rescore"]["mean_sum_nll"]
            )
            score_delta = abs(left[key]["score"] - right[key]["score"])
            per_candidate.append(
                {
                    "test_index": test_index,
                    "grid": left[key]["grid"],
                    "view_sum_nll_abs_differences": view_differences,
                    "mean_sum_nll_abs_difference": mean_delta,
                    "score_abs_difference": score_delta,
                    "sequential": left[key],
                    "batched": right[key],
                }
            )
    sequential_order = [[grid_key(row["grid"]) for row in item] for item in sequential]
    batched_order = [[grid_key(row["grid"]) for row in item] for item in batched]
    return {
        "candidate_pool_exact": True,
        "candidate_order_exact": sequential_order == batched_order,
        "top2_exact": [item[:2] for item in sequential_order]
        == [item[:2] for item in batched_order],
        "metric_schema_exact": metric_schema_exact,
        "token_counts_exact": token_counts_exact,
        "per_row_ids_exact": per_row_ids_exact,
        "per_view_values_compared": {
            metric: len(differences)
            for metric, differences in metric_differences.items()
        },
        "per_view_metrics_exact": {
            metric: all(delta == 0.0 for delta in differences)
            for metric, differences in metric_differences.items()
        },
        "per_view_metric_max_abs_difference": {
            metric: max(differences, default=0.0)
            for metric, differences in metric_differences.items()
        },
        "all_mean_nll_exact": all(
            row["mean_sum_nll_abs_difference"] == 0.0 for row in per_candidate
        ),
        "mean_nll_max_abs_difference": max(
            (row["mean_sum_nll_abs_difference"] for row in per_candidate),
            default=0.0,
        ),
        "all_scores_exact": all(
            row["score_abs_difference"] == 0.0 for row in per_candidate
        ),
        "score_max_abs_difference": max(
            (row["score_abs_difference"] for row in per_candidate), default=0.0
        ),
        "all_values_finite": all(
            math.isfinite(delta)
            for differences in metric_differences.values()
            for delta in differences
        ),
        "candidates": per_candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", default="195c6913")
    parser.add_argument("--arm", default="rowcol_component")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    expected = {
        "steps": 128,
        "decode_color_permutations": 2,
        "decode_geometries": 8,
        "rescore_views": 8,
    }
    for key, value in expected.items():
        if config["task_local"].get(key) != value:
            raise RuntimeError(f"production budget changed: task_local.{key}")
    expected_dfs = {
        "batch_size": 4,
        "max_nodes_per_view": 4096,
        "max_model_forward_tokens_per_view": 12288,
        "max_candidates_per_view": 128,
        "max_new_tokens": 931,
    }
    for key, value in expected_dfs.items():
        if config["dfs"].get(key) != value:
            raise RuntimeError(f"production budget changed: dfs.{key}")
    config["task_local"]["merge_for_inference"] = False

    challenges = json.loads(args.challenges.read_text())
    task = challenges[args.task_id]
    model, _ = worker.load_base(args.model_dir)
    structure = (
        AdditiveStructureAdapter(hidden_size_from_config(model.config))
        .cuda()
        .to(torch.bfloat16)
        .freeze()
    )
    task_model, pre, post = worker.train_task_adapter(
        model, structure, task, args.task_id, args.arm, config
    )
    try:
        original_rescorer = worker.score_candidate_views

        def defer_rescore(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"view_count": 0, "mean_sum_nll": 0.0, "views": []}

        worker.score_candidate_views = defer_rescore
        try:
            synchronize()
            dfs_started = time.perf_counter()
            generated = worker.evaluate_trained_task(
                task_model,
                structure,
                task,
                args.task_id,
                args.arm,
                config,
                pre=pre,
                post=post,
                merge_for_inference=False,
            )
            synchronize()
            dfs_seconds = time.perf_counter() - dfs_started
        finally:
            worker.score_candidate_views = original_rescorer

        pool = [
            [
                {
                    "grid": candidate["grid"],
                    "votes": candidate["votes"],
                    "generation_nll": candidate["generation_nll"],
                }
                for candidate in item
            ]
            for item in generated["ranked_candidates"]
        ]
        sequential, sequential_seconds = score_pool(
            task_model, structure, task, args.task_id, args.arm, config, pool, 1
        )
        batched, batched_seconds = score_pool(
            task_model, structure, task, args.task_id, args.arm, config, pool, 8
        )
    finally:
        task_model.unload()

    result = {
        "schema_version": 1,
        "task_id": args.task_id,
        "arm": args.arm,
        "adapter_merged": False,
        "single_training": True,
        "single_dfs_candidate_pool": True,
        "config": config,
        "candidate_counts": [len(item) for item in pool],
        "nodes_used": generated["nodes_used"],
        "model_forward_tokens": generated["model_forward_tokens"],
        "runtime_seconds": {
            "dfs_candidate_generation": dfs_seconds,
            "sequential_rescore": sequential_seconds,
            "batched_rescore": batched_seconds,
            "speedup": sequential_seconds / batched_seconds,
            "percent_faster": 100.0
            * (sequential_seconds - batched_seconds)
            / sequential_seconds,
        },
        "parity": compare(sequential, batched),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
