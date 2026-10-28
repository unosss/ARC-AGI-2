"""Label-blind, one-arm worker for the base-versus-DiARC paired run."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arc_solver.qwen2d_adapter import (
    AdditiveStructureAdapter,
    StructuralForward,
    hidden_size_from_config,
)
from arc_solver.qwen2d_runtime import collate_token_rows, train_fixed_steps
from kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker import (
    _evaluate_trained_task_impl,
    load_base,
    train_diagnostic,
    training_rows,
)
from kaggle_runs.qwen_diarc_paired_runpod.protocol import (
    ARMS,
    TARGET_MODULES,
    candidate_path,
    mode_config,
    reject_test_outputs,
    seal_candidate,
    validate_config,
)


def task_lora_config(config: dict[str, Any]):
    from peft import LoraConfig, TaskType

    values = config["task_local"]
    return LoraConfig(
        r=values["rank"],
        lora_alpha=values["alpha"],
        lora_dropout=values["dropout"],
        bias="none",
        use_rslora=True,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(TARGET_MODULES),
    )


def validate_diarc_peft_config(peft_config: Any, expected_base: str) -> None:
    """Fail closed if the released adapter no longer has the registered shape."""
    rank = int(getattr(peft_config, "r"))
    targets = set(getattr(peft_config, "target_modules"))
    use_rslora = bool(getattr(peft_config, "use_rslora", False))
    if rank != 256 or targets != set(TARGET_MODULES) or use_rslora:
        raise ValueError("DiARC adapter config differs from rank-256 standard LoRA")
    # The adapter repository may retain the upstream architecture identifier in
    # this metadata. The actual comparison base is enforced by the one frozen
    # checkpoint loaded before either branch, rather than by that stale string.
    if expected_base != "sorokin/qwen3_4b_grids15_sft139":
        raise ValueError("comparison base checkpoint changed")


def freeze_all_except_task(model: torch.nn.Module) -> list[str]:
    trainable = []
    for name, parameter in model.named_parameters():
        allowed = ".task." in name and (
            ".lora_" in name or ".lora_embedding_" in name
        )
        parameter.requires_grad_(allowed)
        if allowed:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("task adapter has no trainable parameters")
    if any(".diarc." in name for name in trainable):
        raise RuntimeError("DiARC parameter became trainable")
    return trainable


def active_adapter_names(model: Any) -> tuple[str, ...]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", ())
    if isinstance(active, str):
        return (active,)
    return tuple(active or ())


def configure_adapters(
    base_model: torch.nn.Module,
    arm: str,
    config: dict[str, Any],
    adapter_source: str | Path,
) -> torch.nn.Module:
    from peft import PeftConfig, PeftModel, get_peft_model

    if arm == "base":
        torch.manual_seed(config["seed"])
        model = get_peft_model(
            base_model, task_lora_config(config), adapter_name="task"
        )
        model.set_adapter("task")
    elif arm == "diarc":
        released = PeftConfig.from_pretrained(str(adapter_source))
        validate_diarc_peft_config(released, config["checkpoint"])
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_source),
            adapter_name="diarc",
            is_trainable=False,
        )
        torch.manual_seed(config["seed"])
        model.add_adapter("task", task_lora_config(config))
        # PeftModel.set_adapter accepts one name; the underlying LoraModel API
        # is the PEFT composition entry point and accepts an additive list.
        model.base_model.set_adapter(["diarc", "task"])
    else:
        raise ValueError(f"unknown arm: {arm}")
    freeze_all_except_task(model)
    expected = ("task",) if arm == "base" else ("diarc", "task")
    if set(active_adapter_names(model)) != set(expected):
        raise RuntimeError(f"active adapters differ: {active_adapter_names(model)}")
    return model


def _rank_candidates(payload: dict[str, Any]) -> None:
    for candidates in payload["ranked_candidates"]:
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank


def _train_task(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    task: dict[str, Any],
    task_id: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    pre = train_diagnostic(model, structure, task, task_id, "text_control", config)
    rows = list(training_rows(task, task_id, config))
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
    return pre, post, elapsed


def smoke_payload(arm: str, task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    """CPU/no-model lifecycle smoke; candidates depend only on visible task data."""
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
    zero_nll = {
        "mean_sum_nll": None,
        "items": [],
        "unavailable_reason": "smoke_cpu_no_model",
    }
    return {
        "schema_version": 1,
        "experiment": 36,
        "mode": "smoke",
        "arm": arm,
        "task_id": task_id,
        "active_adapters": [],
        "trainable_parameter_names": [],
        "pre_task_loo_nll": zero_nll,
        "post_task_loo_nll": zero_nll,
        "ranked_candidates": ranked,
        "nodes_used": 0,
        "model_forward_tokens": 0,
        "timing": {"load_seconds": 0.0, "train_seconds": 0.0, "decode_seconds": 0.0},
        "peak_memory_bytes": 0,
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
        model, arm, effective, adapter_source or config["diarc_adapter"]
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
            pre, post, train_seconds = _train_task(
                model, structure, challenges[task_id], task_id, effective
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
            _rank_candidates(payload)
            payload.update(
                {
                    "schema_version": 1,
                    "experiment": 36,
                    "mode": mode,
                    "experiment_arm": arm,
                    "active_adapters": list(active_adapter_names(model)),
                    "trainable_parameter_names": [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ],
                    "pre_task_loo_nll": pre,
                    "post_task_loo_nll": post,
                    "timing": {
                        "load_seconds": load_seconds,
                        "train_seconds": train_seconds,
                        "decode_seconds": decode_seconds,
                    },
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "structure_sidecar": {
                        "arm": "text_control",
                        "scale": 0.0,
                        "trainable": False,
                    },
                }
            )
            seal_candidate(candidate_path(work, arm, task_id), payload)
    finally:
        if hasattr(model, "delete_adapter"):
            model.delete_adapter("task")
            if arm == "diarc":
                model.delete_adapter("diarc")
        if hasattr(model, "unload"):
            model.unload()
        del structure, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("smoke", "calibrate", "hard-exploratory"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--adapter", default=None)
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
