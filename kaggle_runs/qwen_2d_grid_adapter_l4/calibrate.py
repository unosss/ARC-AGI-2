"""Measured pre-arm throughput projection for the 24/32 task gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter, hidden_size_from_config
from arc_solver.qwen2d_metadata import LayoutMetadata, build_layout
from arc_solver.qwen2d_runtime import (
    DFSBudget,
    collate_token_rows,
    deterministic_cached_dfs,
    serialize_chatml_bounded,
    teacher_forced_metrics,
    train_fixed_steps,
)
from gpu_arm_worker import (
    batches_for,
    find_training_challenges,
    load_base,
    lora_config,
    training_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prepared = json.loads(args.prepared.read_text())
    from peft import get_peft_model

    base, _ = load_base(args.model_dir)
    structure = AdditiveStructureAdapter(hidden_size_from_config(base.config)).cuda().to(torch.bfloat16)
    model = get_peft_model(base, lora_config(config["global_lora"]))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": config["global_learning_rate"],
            },
            {"params": list(structure.parameters()), "lr": config["structure_learning_rate"]},
        ]
    )
    corpus = [
        json.loads(line)
        for line in (args.work / "manifests" / "global_corpus.jsonl").read_text().splitlines()
    ]
    target_tokens = config["throughput_calibration"]["global_loss_tokens"]
    # Long rows dominate transformer cost.  Sampling longest-first is an
    # intentionally conservative projection, unlike an easy corpus prefix.
    corpus.sort(
        key=lambda row: len(
            serialize_chatml_bounded(
                row.get("train", []),
                row["query"],
                row["target"],
                max_length=config["global_max_sequence_length"],
            )[0]
        ),
        reverse=True,
    )
    sample, measured_tokens = [], 0
    for row in corpus:
        sample.append(row)
        measured_tokens += row["loss_token_limit"]
        if measured_tokens >= target_tokens:
            break
    torch.cuda.synchronize()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    train_fixed_steps(
        model,
        structure,
        batches_for(sample, "text_control", config["global_max_sequence_length"], config["seed"]),
        optimizer=optimizer,
        steps=len(sample),
    )
    torch.cuda.synchronize()
    global_seconds = time.monotonic() - started
    global_peak_allocated = torch.cuda.max_memory_allocated()
    global_peak_reserved = torch.cuda.max_memory_reserved()
    checkpoint_probe = args.work / "calibration_checkpoint_probe"
    shutil.rmtree(checkpoint_probe, ignore_errors=True)
    (checkpoint_probe / "lora").mkdir(parents=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    model.save_pretrained(checkpoint_probe / "lora")
    torch.save(structure.state_dict(), checkpoint_probe / "structure.pt")
    torch.save(optimizer.state_dict(), checkpoint_probe / "optimizer.pt")
    torch.cuda.synchronize()
    checkpoint_seconds = time.monotonic() - started
    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint_probe.rglob("*") if path.is_file()
    )
    shutil.rmtree(checkpoint_probe)

    base = model.unload()
    structure.freeze()
    torch.manual_seed(config["seed"])
    model = get_peft_model(base, lora_config(config["task_local"]))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["task_local"]["learning_rate"],
    )
    challenges = json.loads(find_training_challenges().read_text())
    def task_size(task_id):
        candidate = challenges[task_id]
        train_length = len(next(training_rows(candidate, task_id, config))["input_ids"])
        test_length = max(
            len(
                serialize_chatml_bounded(
                    candidate["train"],
                    item["input"],
                    max_length=config["global_max_sequence_length"],
                    reserve_new_tokens=config["dfs"]["max_new_tokens"],
                )[0]
            )
            for item in candidate["test"]
        )
        return max(train_length, test_length)

    task_id = max(prepared["manifest"]["primary_ids"], key=task_size)
    task = challenges[task_id]
    test_index = max(
        range(len(task["test"])),
        key=lambda index: len(
            serialize_chatml_bounded(
                task["train"],
                task["test"][index]["input"],
                max_length=config["global_max_sequence_length"],
                reserve_new_tokens=config["dfs"]["max_new_tokens"],
            )[0]
        ),
    )
    task_sample = list(training_rows(task, task_id, config))[
        : config["throughput_calibration"]["task_local_steps"]
    ]
    torch.cuda.synchronize()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    train_fixed_steps(
        model,
        structure,
        (
            collate_token_rows([row], arm="text_control", shuffle_seed=config["seed"] + index)
            for index, row in enumerate(task_sample)
        ),
        optimizer=optimizer,
        steps=len(task_sample),
    )
    torch.cuda.synchronize()
    task_train_seconds = time.monotonic() - started
    task_peak_allocated = torch.cuda.max_memory_allocated()
    task_peak_reserved = torch.cuda.max_memory_reserved()

    prefix, _, _ = serialize_chatml_bounded(
        task["train"],
        task["test"][test_index]["input"],
        max_length=config["global_max_sequence_length"],
        reserve_new_tokens=config["dfs"]["max_new_tokens"],
    )
    torch.cuda.synchronize()
    started = time.monotonic()
    dfs = deterministic_cached_dfs(
        model,
        structure,
        torch.tensor([prefix], device="cuda"),
        LayoutMetadata.stack([build_layout(prefix, arm="text_control")]),
        grid_bucket=2 * len(task["train"]) + 2,
        budget=DFSBudget(
            max_nodes=config["throughput_calibration"]["dfs_nodes"],
            max_model_forward_tokens=len(prefix) + config["throughput_calibration"]["dfs_nodes"],
            max_candidates=16,
            max_new_tokens=config["dfs"]["max_new_tokens"],
            emergency_seconds=600,
            min_prob=config["dfs"]["min_prob"],
        ),
    )
    torch.cuda.synchronize()
    dfs_seconds = time.monotonic() - started
    calibration_truth = task["train"][0]["output"]
    nll_tokens, nll_start, _ = serialize_chatml_bounded(
        task["train"][1:],
        task["train"][0]["input"],
        calibration_truth,
        max_length=config["global_max_sequence_length"],
    )
    nll_ids = torch.tensor([nll_tokens], device="cuda")
    nll_layout = LayoutMetadata.stack([build_layout(nll_tokens, arm="text_control")])
    torch.cuda.synchronize()
    started = time.monotonic()
    teacher_forced_metrics(model, structure, nll_ids, nll_layout, nll_start)
    torch.cuda.synchronize()
    nll_seconds = time.monotonic() - started
    global_arm = global_seconds * (
        config["global_loss_token_budget_per_arm"] / measured_tokens
    )
    global_arm += checkpoint_seconds * math.ceil(
        len(corpus) / config["checkpoint_every"]["global_optimizer_steps"]
    )
    task_train = task_train_seconds * (
        config["task_local"]["steps"] / len(task_sample)
    )
    view_decode = dfs_seconds * (
        config["dfs"]["max_nodes_per_view"] / max(1, dfs.nodes_used)
    )
    decode_views = (
        config["task_local"]["decode_geometries"]
        * config["task_local"]["decode_color_permutations"]
    )
    decode = decode_views * view_decode
    # Every generated candidate can trigger all rescore views. Using the
    # observed per-view pool without assuming cross-view deduplication is
    # deliberately conservative.
    candidates_per_view = min(
        config["dfs"]["max_candidates_per_view"],
        max(
            1,
            math.ceil(
                len(dfs.candidates)
                * config["dfs"]["max_nodes_per_view"]
                / max(1, dfs.nodes_used)
            ),
        ),
    )
    rescore = (
        decode_views
        * candidates_per_view
        * config["task_local"]["rescore_views"]
        * nll_seconds
    )
    train_diagnostics = 2 * len(task["train"]) * nll_seconds
    truth_nll_per_item = (2 + 2 * decode_views) * nll_seconds

    def projection(manifest):
        primary_ids = manifest["primary_ids"]
        negative_ids = manifest["negative_control_ids"]
        replay_ids = manifest["replay_ids"]
        primary_items = sum(len(challenges[item]["test"]) for item in primary_ids)
        negative_items = sum(len(challenges[item]["test"]) for item in negative_ids)
        replay_items = sum(len(challenges[item]["test"]) for item in replay_ids)
        candidate_per_task = task_train + train_diagnostics
        candidate_per_item = decode + rescore
        truth_per_task = task_train
        # Four-GPU scheduling has a main-arm-dominated first wave and the fifth
        # (negative) arm in a second wave, in both candidate and truth phases.
        candidate_main = (
            len(primary_ids) * candidate_per_task
            + primary_items * candidate_per_item
            + config["replay_repetitions"]
            * (len(replay_ids) * candidate_per_task + replay_items * candidate_per_item)
        )
        candidate_negative = (
            len(negative_ids) * candidate_per_task
            + negative_items * candidate_per_item
        )
        truth_main = (
            len(primary_ids) * truth_per_task
            + primary_items * truth_nll_per_item
        )
        truth_negative = (
            len(negative_ids) * truth_per_task
            + negative_items * truth_nll_per_item
        )
        probe_per_arm = prepared["probe_count"] * (
            nll_seconds
            + dfs_seconds * 128 / max(1, dfs.nodes_used)
        )
        # Covers model loading, probes, serialization, process startup, and
        # timing variance that are not represented by the microbenchmarks.
        return 1.25 * (
            2 * global_arm
            + 2 * probe_per_arm
            + candidate_main
            + candidate_negative
            + truth_main
            + truth_negative
        )

    projected_full = projection(prepared["manifest"])
    projected_minimum = projection(prepared["minimum_manifest"])
    report = {
        "measured_global_loss_tokens": measured_tokens,
        "global_seconds": global_seconds,
        "global_peak_allocated_bytes": global_peak_allocated,
        "global_peak_reserved_bytes": global_peak_reserved,
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint_bytes": checkpoint_bytes,
        "representative_task_id": task_id,
        "representative_test_index": test_index,
        "task_steps": len(task_sample),
        "task_train_seconds": task_train_seconds,
        "task_peak_allocated_bytes": task_peak_allocated,
        "task_peak_reserved_bytes": task_peak_reserved,
        "dfs_nodes_used": dfs.nodes_used,
        "dfs_candidates_observed": len(dfs.candidates),
        "dfs_seconds": dfs_seconds,
        "nll_seconds": nll_seconds,
        "projection_safety_multiplier": 1.25,
        "projected_full_seconds": projected_full,
        "projected_minimum_seconds": projected_minimum,
    }
    (args.work / "manifests" / "throughput_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
