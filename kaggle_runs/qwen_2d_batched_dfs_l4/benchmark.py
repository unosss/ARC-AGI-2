"""Label-blind Experiment 022 sequential-vs-batched DFS benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter, hidden_size_from_config
from arc_solver.qwen2d_augment import augment_task, color_permutation
from arc_solver.qwen2d_metadata import LayoutMetadata, assert_grids15_tokenizer, build_layout
from arc_solver.qwen2d_runtime import (
    DFSBudget,
    batched_deterministic_cached_dfs,
    deterministic_cached_dfs,
    serialize_chatml_bounded,
)


def stable_seed(text: str, base: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{text}".encode()).digest()[:8], "big")


def budget(config: dict) -> DFSBudget:
    values = config["per_view_budget"]
    return DFSBudget(
        max_nodes=values["max_nodes"],
        max_model_forward_tokens=values["max_model_forward_tokens"],
        max_candidates=values["max_candidates"],
        max_new_tokens=values["max_new_tokens"],
        emergency_seconds=values["emergency_seconds"],
        min_prob=config["min_prob"],
    )


def candidate_signature(result) -> list[list[int]]:
    return [item["tokens"] for item in result.candidates]


def run(config_path: Path, model_dir: Path, challenges_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    assert_grids15_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    model.config.use_cache = True
    structure = (
        AdditiveStructureAdapter(hidden_size_from_config(model.config))
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    # No adapter was trained in Phase A. Scale zero preserves the exact base
    # logits while exercising metadata construction and the structural forward.
    scale = 0.0
    challenges = json.loads(challenges_path.read_text())
    task = challenges[config["task_id"]]
    test_index = config["test_index"]
    views = []
    for permutation_index in range(config["color_permutations"]):
        permutation = color_permutation(
            stable_seed(
                f"{config['task_id']}:decode:{permutation_index}", config["seed"]
            )
        )
        for geometry in range(config["geometries"]):
            view_seed = stable_seed(
                f"{config['task_id']}:decode-order:{test_index}:"
                f"{permutation_index}:{geometry}",
                config["seed"],
            )
            augmented = augment_task(
                task,
                geometry=geometry,
                permutation=permutation,
                order_seed=view_seed,
            )
            prefix, _, _ = serialize_chatml_bounded(
                augmented["train"],
                augmented["test"][test_index]["input"],
                max_length=config["max_sequence_length"],
                reserve_new_tokens=config["per_view_budget"]["max_new_tokens"],
            )
            views.append(
                {
                    "permutation_index": permutation_index,
                    "geometry": geometry,
                    "view_seed": view_seed,
                    "prefix": prefix,
                    "grid_bucket": 2 * len(augmented["train"]) + 2,
                }
            )
    if len(views) != 16:
        raise AssertionError(f"expected 16 views, got {len(views)}")

    def run_sequential():
        torch.cuda.synchronize()
        started = time.monotonic()
        results = []
        for view in views:
            results.append(
                deterministic_cached_dfs(
                    model,
                    structure,
                    torch.tensor([view["prefix"]], device="cuda"),
                    LayoutMetadata.stack(
                        [
                            build_layout(
                                view["prefix"],
                                arm=config["arm"],
                                shuffle_seed=view["view_seed"],
                            )
                        ]
                    ),
                    grid_bucket=view["grid_bucket"],
                    budget=budget(config),
                    scale=scale,
                    generated_arm=config["arm"],
                    generated_shuffle_seed=view["view_seed"],
                )
            )
        torch.cuda.synchronize()
        return results, time.monotonic() - started

    sequential, sequential_seconds = run_sequential()

    first = views[0]
    real_batch1 = batched_deterministic_cached_dfs(
        model,
        structure,
        torch.tensor([first["prefix"]], device="cuda"),
        LayoutMetadata.stack(
            [
                build_layout(
                    first["prefix"],
                    arm=config["arm"],
                    shuffle_seed=first["view_seed"],
                )
            ]
        ),
        grid_buckets=[first["grid_bucket"]],
        budgets=[budget(config)],
        arms=[config["arm"]],
        shuffle_seeds=[first["view_seed"]],
        scale=scale,
        max_physical_batch=1,
    ).results[0]
    real_batch1_parity = {
        "tokens_equal": (
            candidate_signature(sequential[0]) == candidate_signature(real_batch1)
        ),
        "counters_equal": (
            sequential[0].nodes_used == real_batch1.nodes_used
            and sequential[0].model_forward_tokens
            == real_batch1.model_forward_tokens
            and sequential[0].stopped_by == real_batch1.stopped_by
        ),
        "max_nll_delta": max(
            (
                abs(a["nll"] - b["nll"])
                for a, b in zip(sequential[0].candidates, real_batch1.candidates)
            ),
            default=0.0,
        ),
    }

    group_indices = [
        [0, 2, 8, 10],
        [1, 3, 9, 11],
        [4, 6, 12, 14],
        [5, 7, 13, 15],
    ]
    groups = [[(index, views[index]) for index in indices] for indices in group_indices]
    if sorted(index for group in groups for index, _ in group) != list(range(16)):
        raise AssertionError("batched groups do not cover every view exactly once")
    if any(len({len(view["prefix"]) for _, view in group}) != 1 for group in groups):
        raise AssertionError("public-NVARC geometry group mixes prefix lengths")

    def run_batched():
        results = [None] * len(views)
        telemetry = {
            "model_calls": 0,
            "physical_forward_tokens": 0,
            "active_lanes_per_call": [],
            "physical_tokens_per_call": [],
        }
        torch.cuda.synchronize()
        started = time.monotonic()
        for group in groups:
            selected = [view for _, view in group]
            result = batched_deterministic_cached_dfs(
                model,
                structure,
                torch.tensor([view["prefix"] for view in selected], device="cuda"),
                LayoutMetadata.stack(
                    [
                        build_layout(
                            view["prefix"],
                            arm=config["arm"],
                            shuffle_seed=view["view_seed"],
                        )
                        for view in selected
                    ]
                ),
                grid_buckets=[view["grid_bucket"] for view in selected],
                budgets=[budget(config) for _ in selected],
                arms=[config["arm"]] * len(selected),
                shuffle_seeds=[view["view_seed"] for view in selected],
                scale=scale,
                max_physical_batch=config["batch_size"],
            )
            telemetry["model_calls"] += result.telemetry.model_calls
            telemetry["physical_forward_tokens"] += (
                result.telemetry.physical_forward_tokens
            )
            telemetry["active_lanes_per_call"].extend(
                result.telemetry.active_lanes_per_call
            )
            telemetry["physical_tokens_per_call"].extend(
                result.telemetry.physical_tokens_per_call
            )
            for (index, _), lane in zip(group, result.results):
                results[index] = lane
        torch.cuda.synchronize()
        return results, telemetry, time.monotonic() - started

    batched_results, telemetry, batched_seconds = run_batched()
    sequential_timings = [sequential_seconds]
    batched_timings = [batched_seconds]
    repeated_token_parity = True
    for repetition in range(1, config["timing_repetitions"]):
        if repetition % 2:
            repeated_batched, _, measured_batched = run_batched()
            repeated_sequential, measured_sequential = run_sequential()
        else:
            repeated_sequential, measured_sequential = run_sequential()
            repeated_batched, _, measured_batched = run_batched()
        sequential_timings.append(measured_sequential)
        batched_timings.append(measured_batched)
        repeated_token_parity = repeated_token_parity and all(
            candidate_signature(reference) == candidate_signature(repeated)
            for reference, repeated in zip(sequential, repeated_sequential)
        )
        repeated_token_parity = repeated_token_parity and all(
            candidate_signature(reference) == candidate_signature(repeated)
            for reference, repeated in zip(batched_results, repeated_batched)
        )
    median_sequential = statistics.median(sequential_timings)
    median_batched = statistics.median(batched_timings)
    median_speedup = median_sequential / median_batched

    parity = []
    for index, (single, batched) in enumerate(zip(sequential, batched_results)):
        tokens_equal = candidate_signature(single) == candidate_signature(batched)
        counters_equal = (
            single.nodes_used == batched.nodes_used
            and single.model_forward_tokens == batched.model_forward_tokens
            and single.stopped_by == batched.stopped_by
        )
        max_nll_delta = max(
            (
                abs(a["nll"] - b["nll"])
                for a, b in zip(single.candidates, batched.candidates)
            ),
            default=0.0,
        )
        parity.append(
            {
                "view_index": index,
                "tokens_equal": tokens_equal,
                "counters_equal": counters_equal,
                "max_nll_delta": max_nll_delta,
            }
        )
    report = {
        "experiment": 22,
        "phase": "label_blind_engine_throughput",
        "profile": config["profile"],
        "accuracy_interpretation_allowed": False,
        "heldout_test_outputs_opened": False,
        "task_id": config["task_id"],
        "test_index": test_index,
        "view_count": len(views),
        "prefix_lengths": [len(view["prefix"]) for view in views],
        "groups": [[index for index, _ in group] for group in groups],
        "logical_budget_per_view": config["per_view_budget"],
        "timing_repetitions": config["timing_repetitions"],
        "sequential_seconds": sequential_timings,
        "batched_seconds": batched_timings,
        "median_sequential_seconds": median_sequential,
        "median_batched_seconds": median_batched,
        "median_speedup": median_speedup,
        "throughput_success_ratio": config["throughput_success_ratio"],
        "throughput_success": median_speedup >= config["throughput_success_ratio"],
        "repeated_token_parity": repeated_token_parity,
        "real_qwen_batch1_parity": real_batch1_parity,
        "all_tokens_equal": all(item["tokens_equal"] for item in parity),
        "all_counters_equal": all(item["counters_equal"] for item in parity),
        "nll_tolerance": config["nll_tolerance"],
        "all_nll_within_tolerance": all(
            item["max_nll_delta"] <= config["nll_tolerance"] for item in parity
        ),
        "parity": parity,
        "sequential_logical_nodes": sum(item.nodes_used for item in sequential),
        "sequential_logical_forward_tokens": sum(
            item.model_forward_tokens for item in sequential
        ),
        "batched_logical_nodes": sum(item.nodes_used for item in batched_results),
        "batched_logical_forward_tokens": sum(
            item.model_forward_tokens for item in batched_results
        ),
        "batched_physical_telemetry": telemetry,
    }
    report["phase_a_pass"] = (
        report["throughput_success"]
        and report["all_tokens_equal"]
        and report["all_counters_equal"]
        and report["all_nll_within_tolerance"]
        and repeated_token_parity
        and real_batch1_parity["tokens_equal"]
        and real_batch1_parity["counters_equal"]
        and real_batch1_parity["max_nll_delta"] <= config["nll_tolerance"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
