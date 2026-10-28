"""One-GPU worker for a complete Experiment 021 arm."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable

import torch

from arc_solver.qwen2d_adapter import (
    AdditiveStructureAdapter,
    assert_scale_zero_equivalence,
    hidden_size_from_config,
)
from arc_solver.qwen2d_augment import (
    augment_task,
    color_permutation,
    invert_prediction,
)
from arc_solver.qwen2d_experiment import MAIN_ARMS, NEGATIVE_ARMS
from arc_solver.qwen2d_experiment import sha256_json
from arc_solver.qwen2d_metadata import LayoutMetadata, PINS, assert_grids15_tokenizer, build_layout
from arc_solver.qwen2d_runtime import (
    DFSBudget,
    batched_multi_view_nll,
    batched_deterministic_cached_dfs,
    collate_token_rows,
    deterministic_cached_dfs,
    multi_view_nll,
    serialize_chatml,
    serialize_chatml_bounded,
    teacher_forced_metrics,
    train_fixed_steps,
)


def stable_seed(text: str, base: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{text}".encode()).digest()[:8], "big")


def check_deadline() -> None:
    deadline = float(os.getenv("ARC021_DEADLINE", "inf"))
    if time.time() > deadline - 120:
        raise RuntimeError("ARC021 emergency deadline reached; resume from checkpoints")


def deadline_epoch() -> float | None:
    value = os.getenv("ARC021_DEADLINE")
    return None if value is None else float(value) - 120


def find_training_challenges() -> Path:
    matches = sorted(
        path
        for path in Path("/kaggle/input").rglob("*.json")
        if "training_challenges" in path.name
    )
    if len(matches) != 1:
        raise RuntimeError(f"training challenges unavailable: {matches}")
    return matches[0]


def token_row(row: dict[str, Any], max_length: int) -> dict[str, Any]:
    tokens, _, dropped = serialize_chatml_bounded(
        row.get("train", []),
        row["query"],
        row["target"],
        max_length=max_length,
    )
    if dropped and row.get("loss_token_limit") is not None:
        raise AssertionError(
            f"budgeted row {row['row_id']} was not truncated before budgeting"
        )
    return {
        "input_ids": tokens,
        "loss_token_limit": row.get("loss_token_limit"),
    }


def batches_for(rows: Iterable[dict[str, Any]], arm: str, max_length: int, seed: int):
    for index, row in enumerate(rows):
        yield collate_token_rows(
            [token_row(row, max_length)], arm=arm, shuffle_seed=seed + index
        )


def load_base(model_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    assert_grids15_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        # The pinned Kaggle Transformers/Qwen3 implementation still forwards
        # ``dtype`` into Qwen3ForCausalLM.__init__, where it is rejected.
        # ``torch_dtype`` is deprecated in newer Transformers but is the
        # compatible argument for this exact offline runtime.
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    # Full-profile calibration uses the longest legal rows. Without activation
    # checkpointing an 8k-token rank-64 forward alone exhausts a 22 GiB L4
    # before the first optimizer step. Checkpointing changes memory/time, not
    # the objective, examples, token budget, or trainable parameters.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    return model, tokenizer


def lora_config(spec: dict[str, Any]):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=spec["rank"],
        lora_alpha=spec["alpha"],
        lora_dropout=spec["dropout"],
        bias="none",
        use_rslora=True,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",
        ],
    )


def global_train(
    arm: str,
    config: dict[str, Any],
    work: Path,
    model_dir: Path,
) -> tuple[torch.nn.Module, AdditiveStructureAdapter]:
    from peft import PeftModel, get_peft_model

    root = work / "arms" / arm / "global"
    complete = root / "COMPLETE"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    model, _ = load_base(model_dir)
    if complete.is_file():
        final = root / "final"
        expected = complete.read_text().strip()
        actual = hashlib.sha256(
            (final / "lora" / "adapter_config.json").read_bytes()
        ).hexdigest()
        if expected != actual:
            raise RuntimeError("completed global checkpoint digest mismatch")
        model = PeftModel.from_pretrained(model, final / "lora").merge_and_unload()
        structure = AdditiveStructureAdapter.load_arm(
            final / "structure", model.config, freeze=True
        ).cuda().to(torch.bfloat16)
        shutil.rmtree(checkpoints, ignore_errors=True)
        (root / "state.json").unlink(missing_ok=True)
        return model, structure
    structure = AdditiveStructureAdapter(hidden_size_from_config(model.config)).cuda().to(torch.bfloat16)
    equivalence_ids = torch.tensor(
        [[
            PINS.im_start, PINS.user, PINS.newline, 0, PINS.eos,
            PINS.im_start, PINS.assistant, PINS.newline, 0, PINS.eos,
        ]],
        device="cuda",
    )
    equivalence = assert_scale_zero_equivalence(
        model,
        structure,
        equivalence_ids,
        LayoutMetadata.stack([build_layout(equivalence_ids[0])]),
        atol=1e-3,
        rtol=1e-4,
    )
    (root / "scale_zero_equivalence.json").write_text(json.dumps(equivalence))
    corpus_path = work / "manifests" / "global_corpus.jsonl"
    fingerprint = sha256_json(
        {
            "arm": arm,
            "config": config,
            "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "model_dir": str(model_dir),
        }
    )
    state_path = root / "state.json"
    start = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("refusing to resume global training with changed inputs")
        start = state["next_row"]
        checkpoint = root / state["checkpoint"]
        if not checkpoint.is_dir():
            raise RuntimeError(f"resume checkpoint is missing: {checkpoint}")
        for stale in checkpoints.glob("step_*"):
            if stale != checkpoint:
                shutil.rmtree(stale, ignore_errors=True)
        model = PeftModel.from_pretrained(model, checkpoint / "lora", is_trainable=True)
        structure.load_state_dict(
            torch.load(checkpoint / "structure.pt", map_location="cuda", weights_only=True)
        )
    else:
        model = get_peft_model(model, lora_config(config["global_lora"]))
    corpus = [
        json.loads(line)
        for line in corpus_path.read_text().splitlines()
    ]
    actual_loss_tokens = sum(int(row["loss_token_limit"]) for row in corpus)
    if actual_loss_tokens != config["global_loss_token_budget_per_arm"]:
        raise AssertionError(
            f"global loss-token budget mismatch: {actual_loss_tokens}"
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters, "lr": config["global_learning_rate"]},
            {"params": list(structure.parameters()), "lr": config["structure_learning_rate"]},
        ],
        weight_decay=0.0,
    )
    if start:
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cuda"))
    interval = config["checkpoint_every"]["global_optimizer_steps"]
    losses = []
    while start < len(corpus):
        check_deadline()
        end = min(len(corpus), start + interval)
        losses.extend(
            train_fixed_steps(
                model,
                structure,
                batches_for(
                    corpus[start:end],
                    arm,
                    config["global_max_sequence_length"],
                    config["seed"] + start,
                ),
                optimizer=optimizer,
                steps=end - start,
                deadline_epoch=deadline_epoch(),
            )
        )
        checkpoint = checkpoints / f"step_{end:07d}"
        temporary = checkpoints / f".step_{end:07d}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        (temporary / "lora").mkdir(parents=True)
        model.save_pretrained(temporary / "lora")
        torch.save(structure.state_dict(), temporary / "structure.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        os.replace(temporary, checkpoint)
        state_tmp = state_path.with_suffix(".tmp")
        state_tmp.write_text(
            json.dumps(
                {
                    "next_row": end,
                    "checkpoint": str(checkpoint.relative_to(root)),
                    "fingerprint": fingerprint,
                }
            )
        )
        os.replace(state_tmp, state_path)
        for stale in checkpoints.glob("step_*"):
            if stale != checkpoint:
                shutil.rmtree(stale, ignore_errors=True)
        start = end
    final = root / "final"
    final_tmp = root / ".final.tmp"
    shutil.rmtree(final_tmp, ignore_errors=True)
    model.save_pretrained(final_tmp / "lora")
    structure.save_arm(
        final_tmp / "structure",
        arm,
        {
            "global_loss_tokens": actual_loss_tokens,
            "global_rows": len(corpus),
        },
    )
    shutil.rmtree(final, ignore_errors=True)
    os.replace(final_tmp, final)
    complete_tmp = complete.with_suffix(".tmp")
    complete_tmp.write_text(
        hashlib.sha256((final / "lora" / "adapter_config.json").read_bytes()).hexdigest()
    )
    os.replace(complete_tmp, complete)
    shutil.rmtree(checkpoints, ignore_errors=True)
    state_path.unlink(missing_ok=True)
    return model.merge_and_unload(), structure.freeze()


def parse_candidate(tokens: list[int]) -> list[list[int]] | None:
    if not tokens or tokens[-1] != PINS.eos:
        return None
    rows, row = [], []
    for token in tokens[:-1]:
        if token == PINS.newline:
            if not row:
                return None
            rows.append(row)
            row = []
        elif token in PINS.cells:
            row.append(token)
        else:
            return None
    if not row:
        return None
    rows.append(row)
    if not 1 <= len(rows) <= 30 or any(len(value) != len(rows[0]) for value in rows):
        return None
    if not 1 <= len(rows[0]) <= 30:
        return None
    return rows


def training_rows(task: dict[str, Any], task_id: str, config: dict[str, Any]):
    pairs = task["train"]
    count = config["task_local"]["train_color_permutations"]
    seed = config["seed"]
    for permutation_index in range(count):
        permutation = color_permutation(stable_seed(f"{task_id}:train:{permutation_index}", seed))
        for geometry in range(config["task_local"]["train_geometries"]):
            augmented = augment_task(
                task,
                geometry=geometry,
                permutation=permutation,
                order_seed=stable_seed(f"{task_id}:order:{permutation_index}:{geometry}", seed),
            )
            held = augmented["train"][-1]
            tokens, _, _ = serialize_chatml_bounded(
                augmented["train"][:-1],
                held["input"],
                held["output"],
                max_length=config["global_max_sequence_length"],
            )
            yield {"input_ids": tokens}


def train_diagnostic(
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    arm: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    values = []
    for held_index, held in enumerate(task["train"]):
        demos = [pair for index, pair in enumerate(task["train"]) if index != held_index]
        tokens, answer_start, _ = serialize_chatml_bounded(
            demos,
            held["input"],
            held["output"],
            max_length=config["global_max_sequence_length"],
        )
        layout_seed = stable_seed(f"{task_id}:diagnostic:{held_index}", config["seed"])
        ids = torch.tensor([tokens], device="cuda")
        layout = LayoutMetadata.stack(
            [build_layout(tokens, arm=arm, shuffle_seed=layout_seed)]
        )
        values.append(
            teacher_forced_metrics(model, structure, ids, layout, answer_start)["sum_nll"]
        )
    return {"canonical_loo_mean_sum_nll": sum(values) / len(values), "folds": len(values)}


def score_candidate_views(
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    test_index: int,
    candidate: list[list[int]],
    arm: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    views = []
    permutation = color_permutation(
        stable_seed(f"{task_id}:{test_index}:rescore", config["seed"])
    )
    for geometry in range(config["task_local"]["rescore_views"]):
        view_seed = stable_seed(
            f"{task_id}:{test_index}:rescore:{geometry}", config["seed"]
        )
        augmented = augment_task(
            task,
            geometry=geometry,
            permutation=permutation,
            order_seed=view_seed,
        )
        transformed = augment_task(
            {"train": [], "test": [{"input": candidate}]},
            geometry=geometry,
            permutation=permutation,
            order_seed=0,
        )["test"][0]["input"]
        tokens, start, _ = serialize_chatml_bounded(
            augmented["train"],
            augmented["test"][test_index]["input"],
            transformed,
            max_length=config["global_max_sequence_length"],
        )
        views.append(
            (
                torch.tensor([tokens], device="cuda"),
                LayoutMetadata.stack(
                    [build_layout(tokens, arm=arm, shuffle_seed=view_seed)]
                ),
                start,
            )
        )
    batch_size = int(config["task_local"].get("rescore_batch_size", 1))
    if batch_size > 1:
        return batched_multi_view_nll(
            model,
            structure,
            views,
            max_batch_size=batch_size,
            deadline_epoch=deadline_epoch(),
        )
    return multi_view_nll(model, structure, views, deadline_epoch=deadline_epoch())


def train_task_adapter(
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    arm: str,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from peft import get_peft_model

    torch.manual_seed(config["seed"])
    task_model = get_peft_model(model, lora_config(config["task_local"]))
    pre = train_diagnostic(task_model, structure, task, task_id, arm, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in task_model.parameters() if parameter.requires_grad],
        lr=config["task_local"]["learning_rate"],
        weight_decay=0.0,
    )
    rows = list(training_rows(task, task_id, config))
    expected = config["task_local"]["steps"]
    if len(rows) != expected:
        raise AssertionError(f"task rows {len(rows)} != steps {expected}")
    train_fixed_steps(
        task_model,
        structure,
        (
            collate_token_rows([row], arm=arm, shuffle_seed=config["seed"] + index)
            for index, row in enumerate(rows)
        ),
        optimizer=optimizer,
        steps=expected,
        deadline_epoch=deadline_epoch(),
    )
    post = train_diagnostic(task_model, structure, task, task_id, arm, config)
    return task_model, pre, post


def _evaluate_trained_task_impl(
    task_model,
    structure,
    task: dict[str, Any],
    task_id: str,
    arm: str,
    config: dict[str, Any],
    *,
    pre: dict[str, Any],
    post: dict[str, Any],
    merge_for_inference: bool,
) -> dict[str, Any]:
    ranked_items, diagnostics = [], []
    decode_diagnostics: list[list[dict[str, Any]]] = []
    total_nodes = total_forward = 0
    batch_telemetry = {
        "model_calls": 0,
        "physical_forward_tokens": 0,
        "active_lanes_per_call": [],
        "physical_tokens_per_call": [],
    }
    for test_index, test_item in enumerate(task["test"]):
        samples: dict[tuple[tuple[int, ...], ...], list[float]] = defaultdict(list)
        item_decode_diagnostics: list[dict[str, Any]] = []
        views = []
        for permutation_index in range(config["task_local"]["decode_color_permutations"]):
            permutation = color_permutation(
                stable_seed(f"{task_id}:decode:{permutation_index}", config["seed"])
            )
            for geometry in range(config["task_local"]["decode_geometries"]):
                view_seed = stable_seed(
                    f"{task_id}:decode-order:{test_index}:{permutation_index}:{geometry}",
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
                    max_length=config["global_max_sequence_length"],
                    reserve_new_tokens=config["dfs"]["max_new_tokens"],
                )
                views.append(
                    {
                        "permutation": permutation,
                        "geometry": geometry,
                        "view_seed": view_seed,
                        "prefix": prefix,
                        "grid_bucket": 2 * len(augmented["train"]) + 2,
                    }
                )

        def collect(view: dict[str, Any], result) -> None:
            nonlocal total_nodes, total_forward
            total_nodes += result.nodes_used
            total_forward += result.model_forward_tokens
            if config["task_local"].get("preserve_decode_diagnostics", False):
                item_decode_diagnostics.append(
                    {
                        "permutation": view["permutation"],
                        "geometry": view["geometry"],
                        "view_seed": view["view_seed"],
                        "prefix_length": len(view["prefix"]),
                        "candidates": result.candidates,
                        "nodes_used": result.nodes_used,
                        "model_forward_tokens": result.model_forward_tokens,
                        "stopped_by": result.stopped_by,
                    }
                )
            for candidate in result.candidates:
                grid = parse_candidate(candidate["tokens"])
                if grid is not None:
                    canonical = invert_prediction(
                        grid,
                        geometry=view["geometry"],
                        permutation=view["permutation"],
                    )
                    samples[tuple(map(tuple, canonical))].append(candidate["nll"])

        batch_size = int(config["dfs"].get("batch_size", 1))
        if batch_size == 1:
            for view in views:
                result = deterministic_cached_dfs(
                    task_model,
                    structure,
                    torch.tensor([view["prefix"]], device="cuda"),
                    LayoutMetadata.stack(
                        [
                            build_layout(
                                view["prefix"],
                                arm=arm,
                                shuffle_seed=view["view_seed"],
                            )
                        ]
                    ),
                    grid_bucket=view["grid_bucket"],
                    budget=DFSBudget(
                        max_nodes=config["dfs"]["max_nodes_per_view"],
                        max_model_forward_tokens=config["dfs"][
                            "max_model_forward_tokens_per_view"
                        ],
                        max_candidates=config["dfs"]["max_candidates_per_view"],
                        max_new_tokens=config["dfs"]["max_new_tokens"],
                        emergency_seconds=config["dfs"]["emergency_seconds_per_view"],
                        min_prob=config["dfs"]["min_prob"],
                        deadline_epoch=deadline_epoch(),
                    ),
                    generated_arm=arm,
                    generated_shuffle_seed=view["view_seed"],
                )
                collect(view, result)
        else:
            if not 2 <= batch_size <= 4:
                raise ValueError("dfs.batch_size must be in 1..4")
            if (
                batch_size == 4
                and config["task_local"]["decode_color_permutations"] == 2
                and config["task_local"]["decode_geometries"] == 8
            ):
                # Exact public-NVARC grouping: each shape-preserving geometry
                # pair contains both color permutations.
                group_indices = [
                    [0, 2, 8, 10],
                    [1, 3, 9, 11],
                    [4, 6, 12, 14],
                    [5, 7, 13, 15],
                ]
            else:
                by_length: dict[int, list[int]] = defaultdict(list)
                for index, view in enumerate(views):
                    by_length[len(view["prefix"])].append(index)
                group_indices = [
                    indices[offset : offset + batch_size]
                    for indices in by_length.values()
                    for offset in range(0, len(indices), batch_size)
                ]
            if sorted(index for group in group_indices for index in group) != list(
                range(len(views))
            ):
                raise AssertionError("batched groups must cover every view exactly once")
            for indices in group_indices:
                group = [views[index] for index in indices]
                if len({len(view["prefix"]) for view in group}) != 1:
                    raise AssertionError("batched DFS group mixes prefix lengths")
                batched = batched_deterministic_cached_dfs(
                    task_model,
                    structure,
                    torch.tensor([view["prefix"] for view in group], device="cuda"),
                    LayoutMetadata.stack(
                        [
                            build_layout(
                                view["prefix"],
                                arm=arm,
                                shuffle_seed=view["view_seed"],
                            )
                            for view in group
                        ]
                    ),
                    grid_buckets=[view["grid_bucket"] for view in group],
                    budgets=[
                        DFSBudget(
                            max_nodes=config["dfs"]["max_nodes_per_view"],
                            max_model_forward_tokens=config["dfs"][
                                "max_model_forward_tokens_per_view"
                            ],
                            max_candidates=config["dfs"]["max_candidates_per_view"],
                            max_new_tokens=config["dfs"]["max_new_tokens"],
                            emergency_seconds=config["dfs"][
                                "emergency_seconds_per_view"
                            ],
                            min_prob=config["dfs"]["min_prob"],
                            deadline_epoch=deadline_epoch(),
                        )
                        for _ in group
                    ],
                    arms=[arm] * len(group),
                    shuffle_seeds=[view["view_seed"] for view in group],
                    max_physical_batch=batch_size,
                )
                batch_telemetry["model_calls"] += batched.telemetry.model_calls
                batch_telemetry["physical_forward_tokens"] += (
                    batched.telemetry.physical_forward_tokens
                )
                batch_telemetry["active_lanes_per_call"].extend(
                    batched.telemetry.active_lanes_per_call
                )
                batch_telemetry["physical_tokens_per_call"].extend(
                    batched.telemetry.physical_tokens_per_call
                )
                for view, result in zip(group, batched.results):
                    collect(view, result)
        scored = []
        for key, generation_nll in samples.items():
            grid = [list(row) for row in key]
            view = score_candidate_views(
                task_model, structure, task, task_id, test_index, grid, arm, config
            )
            score = len(generation_nll) - view["mean_sum_nll"]
            scored.append(
                {
                    "grid": grid,
                    "score": score,
                    "votes": len(generation_nll),
                    "generation_nll": generation_nll,
                    "rescore": view,
                }
            )
        # Python's stable sort preserves first-generation order on exact score
        # ties, matching upstream score_sum(..., reverse=True).
        scored.sort(key=lambda item: item["score"], reverse=True)
        ranked_items.append(scored)
        decode_diagnostics.append(item_decode_diagnostics)
        diagnostics.append(
            {
                "canonical_nll": None,
                "view16_nll": None,
                "first_token_nll": None,
                "per_row_nll": None,
                "pre_task_lora": pre,
                "post_task_lora": post,
            }
        )
    payload = {
        "task_id": task_id,
        "arm": arm,
        "ranked_candidates": ranked_items,
        "item_diagnostics": diagnostics,
        "nodes_used": total_nodes,
        "model_forward_tokens": total_forward,
        "batched_dfs_telemetry": (
            batch_telemetry if int(config["dfs"].get("batch_size", 1)) > 1 else None
        ),
        "task_lora_merged_for_inference": merge_for_inference,
    }
    if config["task_local"].get("preserve_decode_diagnostics", False):
        payload["decode_diagnostics"] = decode_diagnostics
    return payload


def evaluate_trained_task(
    task_model,
    structure,
    task: dict[str, Any],
    task_id: str,
    arm: str,
    config: dict[str, Any],
    *,
    pre: dict[str, Any],
    post: dict[str, Any],
    merge_for_inference: bool,
) -> dict[str, Any]:
    if merge_for_inference:
        # Keep the task adapter reversible: merge removes LoRA wrapper overhead
        # during the many DFS/NLL forwards. Always unmerge, including when an
        # inference exception occurs, so the shared arm model cannot be changed.
        task_model.merge_adapter(safe_merge=True)
    try:
        return _evaluate_trained_task_impl(
            task_model,
            structure,
            task,
            task_id,
            arm,
            config,
            pre=pre,
            post=post,
            merge_for_inference=merge_for_inference,
        )
    finally:
        if merge_for_inference:
            task_model.unmerge_adapter()


def evaluate_task(
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    arm: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    task_model, pre, post = train_task_adapter(
        model, structure, task, task_id, arm, config
    )
    try:
        return evaluate_trained_task(
            task_model,
            structure,
            task,
            task_id,
            arm,
            config,
            pre=pre,
            post=post,
            merge_for_inference=bool(
                config["task_local"].get("merge_for_inference", False)
            ),
        )
    finally:
        task_model.unload()


def probe_metrics(model, structure, rows, arm, config):
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        probe_seed = stable_seed(row["row_id"], config["seed"])
        tokens, start, _ = serialize_chatml_bounded(
            row["train"],
            row["query"],
            row["target"],
            max_length=config["global_max_sequence_length"],
        )
        metric = teacher_forced_metrics(
            model,
            structure,
            torch.tensor([tokens], device="cuda"),
            LayoutMetadata.stack(
                [build_layout(tokens, arm=arm, shuffle_seed=probe_seed)]
            ),
            start,
        )
        # Exact/valid generation is evaluated by a deterministic low-budget DFS.
        prefix, _, _ = serialize_chatml_bounded(
            row["train"],
            row["query"],
            max_length=config["global_max_sequence_length"],
            reserve_new_tokens=config["dfs"]["max_new_tokens"],
        )
        result = deterministic_cached_dfs(
            model,
            structure,
            torch.tensor([prefix], device="cuda"),
            LayoutMetadata.stack(
                [build_layout(prefix, arm=arm, shuffle_seed=probe_seed)]
            ),
            grid_bucket=2 * len(row["train"]) + 2,
            budget=DFSBudget(
                max_nodes=128,
                max_model_forward_tokens=len(prefix) + 128,
                max_candidates=1,
                max_new_tokens=config["dfs"]["max_new_tokens"],
                emergency_seconds=60,
                min_prob=config["dfs"]["min_prob"],
                deadline_epoch=deadline_epoch(),
            ),
            generated_arm=arm,
            generated_shuffle_seed=probe_seed,
        )
        prediction = parse_candidate(result.candidates[0]["tokens"]) if result.candidates else None
        families[row["family"]].append(
            {
                "exact": prediction == row["target"],
                "valid_grid": prediction is not None,
                "sum_nll": metric["sum_nll"],
            }
        )
    return {
        family: {
            "count": len(items),
            "exact": sum(item["exact"] for item in items) / len(items),
            "valid_grid": sum(item["valid_grid"] for item in items) / len(items),
            "mean_sum_nll": sum(item["sum_nll"] for item in items) / len(items),
        }
        for family, items in families.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prepared = json.loads(args.prepared.read_text())
    if args.arm not in (*MAIN_ARMS, *NEGATIVE_ARMS):
        raise ValueError(args.arm)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, structure = global_train(args.arm, config, args.work, args.model_dir)
    probes = json.loads((args.work / "manifests" / "probe_rows.json").read_text())
    arm_root = args.work / "arms" / args.arm
    (arm_root / "probe_metrics.json").write_text(
        json.dumps(probe_metrics(model, structure, probes, args.arm, config), sort_keys=True)
    )
    challenges = json.loads(find_training_challenges().read_text())
    task_ids = (
        prepared["manifest"]["negative_control_ids"]
        if args.arm in NEGATIVE_ARMS
        else prepared["manifest"]["primary_ids"]
    )
    candidate_root = arm_root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        check_deadline()
        destination = candidate_root / f"{task_id}.json"
        if destination.is_file():
            continue
        payload = evaluate_task(
            model, structure, challenges[task_id], task_id, args.arm, config
        )
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, destination)
    replay = {}
    if args.arm in MAIN_ARMS:
        for task_id in prepared["manifest"]["replay_ids"]:
            check_deadline()
            hashes = []
            for _ in range(config["replay_repetitions"]):
                payload = evaluate_task(
                    model, structure, challenges[task_id], task_id, args.arm, config
                )
                hashes.append(
                    hashlib.sha256(
                        json.dumps(payload["ranked_candidates"], sort_keys=True).encode()
                    ).hexdigest()
                )
            replay[task_id] = {"hashes": hashes, "identical": len(set(hashes)) == 1}
    (arm_root / "replays.json").write_text(json.dumps(replay, sort_keys=True))
    digest = hashlib.sha256(
        "".join(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(candidate_root.glob("*.json"))
        ).encode()
    ).hexdigest()
    complete = arm_root / "ARM_COMPLETE.sha256"
    complete_tmp = complete.with_suffix(".tmp")
    complete_tmp.write_text(digest + "\n")
    os.replace(complete_tmp, complete)


if __name__ == "__main__":
    main()
