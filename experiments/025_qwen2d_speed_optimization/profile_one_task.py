"""Profile the production-budget task-local phase on one RTX 4090.

This deliberately keeps Experiment 024's 128 training steps, 16 decode views,
DFS budgets, and 8 rescore views.  It uses the downloaded SFT checkpoint with
a zero-initialized structural sidecar because the two expensive global arm
checkpoints do not yet exist on RunPod.  The measured task-local tensor shapes
and budgets match production; candidate quality is not an accuracy result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import torch

import kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker as worker
from arc_solver.qwen2d_adapter import AdditiveStructureAdapter, hidden_size_from_config


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_wrapper(
    name: str,
    function: Callable[..., Any],
    records: list[dict[str, Any]],
    context: dict[str, str],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        synchronize()
        started = time.perf_counter()
        result = function(*args, **kwargs)
        synchronize()
        record = {
            "phase": name,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        record.update(context)
        if hasattr(result, "results"):
            record["candidates"] = sum(len(item.candidates) for item in result.results)
            record["nodes"] = sum(item.nodes_used for item in result.results)
        elif isinstance(result, dict) and "view_count" in result:
            record["views"] = result["view_count"]
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        return result

    return wrapped


def candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ranked_candidates": payload["ranked_candidates"],
        "decode_diagnostics": payload.get("decode_diagnostics"),
    }


def compare_payloads(
    unmerged: dict[str, Any], merged: dict[str, Any]
) -> dict[str, Any]:
    unmerged_ranked = [
        [candidate["grid"] for candidate in item]
        for item in unmerged["ranked_candidates"]
    ]
    merged_ranked = [
        [candidate["grid"] for candidate in item]
        for item in merged["ranked_candidates"]
    ]
    unmerged_tokens = [
        [
            [candidate["tokens"] for candidate in view["candidates"]]
            for view in item
        ]
        for item in unmerged["decode_diagnostics"]
    ]
    merged_tokens = [
        [
            [candidate["tokens"] for candidate in view["candidates"]]
            for view in item
        ]
        for item in merged["decode_diagnostics"]
    ]

    def collect_nll_differences(
        a: Any, b: Any, differences: list[float], key: str = ""
    ) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for child_key in a.keys() & b.keys():
                collect_nll_differences(
                    a[child_key], b[child_key], differences, child_key
                )
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for left, right in zip(a, b):
                collect_nll_differences(left, right, differences, key)
        elif (
            "nll" in key
            and isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and math.isfinite(float(a))
            and math.isfinite(float(b))
        ):
            differences.append(abs(float(a) - float(b)))

    decode_nll_differences: list[float] = []
    for unmerged_item, merged_item in zip(
        unmerged["decode_diagnostics"], merged["decode_diagnostics"]
    ):
        for unmerged_view, merged_view in zip(unmerged_item, merged_item):
            unmerged_by_tokens = {
                tuple(candidate["tokens"]): candidate
                for candidate in unmerged_view["candidates"]
            }
            merged_by_tokens = {
                tuple(candidate["tokens"]): candidate
                for candidate in merged_view["candidates"]
            }
            for tokens in unmerged_by_tokens.keys() & merged_by_tokens.keys():
                decode_nll_differences.append(
                    abs(
                        float(unmerged_by_tokens[tokens]["nll"])
                        - float(merged_by_tokens[tokens]["nll"])
                    )
                )

    ranked_nll_differences: list[float] = []
    common_ranked_grids = 0
    for unmerged_item, merged_item in zip(
        unmerged["ranked_candidates"], merged["ranked_candidates"]
    ):
        unmerged_by_grid = {
            tuple(map(tuple, candidate["grid"])): candidate
            for candidate in unmerged_item
        }
        merged_by_grid = {
            tuple(map(tuple, candidate["grid"])): candidate
            for candidate in merged_item
        }
        common = unmerged_by_grid.keys() & merged_by_grid.keys()
        common_ranked_grids += len(common)
        for grid in common:
            collect_nll_differences(
                unmerged_by_grid[grid],
                merged_by_grid[grid],
                ranked_nll_differences,
            )

    nll_differences = decode_nll_differences + ranked_nll_differences
    counters = {
        key: {
            "unmerged": unmerged.get(key),
            "merged": merged.get(key),
            "equal": unmerged.get(key) == merged.get(key),
        }
        for key in ("nodes_used", "model_forward_tokens", "batched_dfs_telemetry")
    }
    return {
        "ranked_grid_order_exact": unmerged_ranked == merged_ranked,
        "decode_candidate_tokens_and_order_exact": unmerged_tokens == merged_tokens,
        "ranked_candidate_counts": {
            "unmerged": [len(item) for item in unmerged["ranked_candidates"]],
            "merged": [len(item) for item in merged["ranked_candidates"]],
        },
        "common_ranked_grids": common_ranked_grids,
        "decode_nll_values_compared": len(decode_nll_differences),
        "decode_nll_max_abs_difference": max(
            decode_nll_differences, default=0.0
        ),
        "ranked_nll_values_compared": len(ranked_nll_differences),
        "ranked_nll_max_abs_difference": max(
            ranked_nll_differences, default=0.0
        ),
        "nll_values_compared": len(nll_differences),
        "nll_max_abs_difference": max(nll_differences, default=0.0),
        "nll_mean_abs_difference": (
            sum(nll_differences) / len(nll_differences)
            if nll_differences
            else 0.0
        ),
        "counters": counters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", default="195c6913")
    parser.add_argument("--arm", default="rowcol_component")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare-merge",
        action="store_true",
        help="Train once, then run the full inference budget unmerged and merged.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.compare_merge:
        config["task_local"]["preserve_decode_diagnostics"] = True
    challenges = json.loads(args.challenges.read_text())
    records: list[dict[str, Any]] = []
    context: dict[str, str] = {}

    synchronize()
    started = time.perf_counter()
    model, _ = worker.load_base(args.model_dir)
    structure = (
        AdditiveStructureAdapter(hidden_size_from_config(model.config))
        .cuda()
        .to(torch.bfloat16)
        .freeze()
    )
    synchronize()
    records.append(
        {
            "phase": "base_model_and_zero_sidecar_load",
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
    )
    print(json.dumps(records[-1], sort_keys=True), flush=True)

    worker.train_diagnostic = timed_wrapper(
        "train_diagnostic", worker.train_diagnostic, records, context
    )
    worker.train_fixed_steps = timed_wrapper(
        "task_lora_train_128_steps", worker.train_fixed_steps, records, context
    )
    worker.batched_deterministic_cached_dfs = timed_wrapper(
        "batched_dfs_group",
        worker.batched_deterministic_cached_dfs,
        records,
        context,
    )
    worker.multi_view_nll = timed_wrapper(
        "candidate_8view_nll", worker.multi_view_nll, records, context
    )

    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    if args.compare_merge:
        context["stage"] = "training"
        task_model, pre, post = worker.train_task_adapter(
            model,
            structure,
            challenges[args.task_id],
            args.task_id,
            args.arm,
            config,
        )
        payloads = {}
        inference_summaries = {}
        try:
            for mode, merged in (("unmerged", False), ("merged", True)):
                context["stage"] = mode
                synchronize()
                mode_started = time.perf_counter()
                payloads[mode] = worker.evaluate_trained_task(
                    task_model,
                    structure,
                    challenges[args.task_id],
                    args.task_id,
                    args.arm,
                    config,
                    pre=pre,
                    post=post,
                    merge_for_inference=merged,
                )
                synchronize()
                inference_summaries[mode] = {
                    "seconds": time.perf_counter() - mode_started,
                    "candidate_payload_sha256": worker.sha256_json(
                        candidate_payload(payloads[mode])
                    ),
                    "nodes_used": payloads[mode]["nodes_used"],
                    "model_forward_tokens": payloads[mode]["model_forward_tokens"],
                }
        finally:
            task_model.unload()
        payload = payloads["merged"]
    else:
        context["stage"] = "unmerged"
        payload = worker.evaluate_task(
            model,
            structure,
            challenges[args.task_id],
            args.task_id,
            args.arm,
            config,
        )
    synchronize()
    total = {
        "phase": "evaluate_task_total",
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "ranked_candidate_counts": [
            len(candidates) for candidates in payload["ranked_candidates"]
        ],
        "nodes_used": payload["nodes_used"],
        "model_forward_tokens": payload["model_forward_tokens"],
    }
    records.append(total)
    print(json.dumps(total, sort_keys=True), flush=True)

    result = {
        "schema_version": 1,
        "task_id": args.task_id,
        "arm": args.arm,
        "config": config,
        "profile_scope": (
            "Production task-local budgets on the SFT base checkpoint with a "
            "zero-initialized frozen structural sidecar; not an accuracy result."
        ),
        "records": records,
        "summary": total,
    }
    if args.compare_merge:
        result["single_training_comparison"] = {
            "inference": inference_summaries,
            "parity": compare_payloads(payloads["unmerged"], payloads["merged"]),
            "payloads": payloads,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
