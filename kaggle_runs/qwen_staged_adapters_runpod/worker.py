"""Dedicated label-blind GPU worker for Experiment 034."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

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
from arc_solver.qwen2d_metadata import LayoutMetadata, build_layout
from arc_solver.qwen2d_runtime import (
    DFSBudget,
    collate_token_rows,
    deterministic_cached_dfs,
    serialize_chatml_bounded,
)
from arc_solver.qwen_staged_adapters import (
    activate_named_adapters,
    adapter_tensor_digest,
    assert_frozen_digest_unchanged,
    freeze_all_except_adapter,
    make_lora_config,
    stage_specs_from_config,
    validate_expected_target_count,
    validate_trainable_adapter_band,
)
from kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker import (
    _evaluate_trained_task_impl,
    batches_for,
    load_base,
    stable_seed,
    train_diagnostic,
    training_rows,
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_digest(module: torch.nn.Module) -> str:
    checksum = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        checksum.update(name.encode() + b"\0")
        checksum.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical(value) + "\n")
    os.replace(temporary, path)


def deadline() -> float | None:
    value = os.environ.get("ARC034_DEADLINE")
    return None if value is None else float(value) - 120.0


def check_deadline() -> None:
    limit = deadline()
    if limit is not None and time.time() >= limit:
        raise RuntimeError("Experiment 034 deadline reached; restart is safe")


def runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Expose the legacy runtime keys consumed by the production DFS helpers."""
    value = copy.deepcopy(config)
    value["global_max_sequence_length"] = config["corpus"][
        "global_max_sequence_length"
    ]
    return value


def adapter_config(spec, values: dict[str, Any]):
    return make_lora_config(
        spec,
        r=values["rank"],
        lora_alpha=values["alpha"],
        lora_dropout=values["dropout"],
        use_rslora=True,
    )


def _adapter_state(model) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in model.state_dict().items():
        if any(
            marker in name
            for marker in (
                ".lora_A.",
                ".lora_B.",
                ".lora_embedding_A.",
                ".lora_embedding_B.",
                ".lora_magnitude_vector.",
            )
        ):
            state[name] = value.detach().cpu()
    if not state:
        raise RuntimeError("no named-adapter tensors found for checkpoint")
    return state


def _load_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected adapter checkpoint keys: {unexpected[:3]}")
    missing_lora = [
        name
        for name in incompatible.missing_keys
        if ".lora_" in name and name in state
    ]
    if missing_lora:
        raise RuntimeError(f"adapter checkpoint keys were not loaded: {missing_lora[:3]}")


def optimizer_state_to(
    optimizer: torch.optim.Optimizer, device: torch.device | str
) -> None:
    """Move Adam moments without changing optimizer math or checkpoints."""
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def optimizer_offload_for_total_memory(total_memory_bytes: int) -> bool:
    return total_memory_bytes < 32 * 1024**3


def needs_optimizer_offload(model) -> bool:
    device = next(model.parameters()).device
    if device.type != "cuda":
        return False
    total = torch.cuda.get_device_properties(device).total_memory
    return optimizer_offload_for_total_memory(total)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def probe_rows(work: Path) -> list[dict[str, Any]]:
    partitions = json.loads(
        (work / "manifests" / "probe_rows.json").read_text()
    )
    rows = [
        row
        for stage in ("perception", "interpretation")
        for row in partitions[stage]
    ]
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["row_id"]):
        selected.setdefault(row["family"], row)
    return list(selected.values())


@torch.no_grad()
def probe_snapshot(
    model,
    structure,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Measure a fixed held-out row per family without reading ARC labels."""
    model.eval()
    forward = StructuralForward(model, structure)
    result = {}
    for row in rows:
        tokens, answer_start, _ = serialize_chatml_bounded(
            row.get("train", []),
            row["query"],
            row["target"],
            max_length=config["global_max_sequence_length"],
        )
        ids = torch.tensor([tokens], device=next(model.parameters()).device)
        layout = LayoutMetadata.stack(
            [build_layout(tokens, arm="rowcol_component")]
        )
        logits = forward(
            input_ids=ids,
            layout=layout,
            return_dict=True,
            use_cache=False,
        ).logits.float()
        target = ids[:, answer_start:]
        prediction = logits[:, answer_start - 1 : -1]
        token_nll = torch.nn.functional.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            target.reshape(-1),
            reduction="none",
        )
        result[row["family"]] = {
            "row_id": row["row_id"],
            "sum_nll": float(token_nll.sum()),
            "mean_nll": float(token_nll.mean()),
            "teacher_forced_token_exact": bool(
                torch.equal(prediction.argmax(dim=-1), target)
            ),
            "token_count": int(target.numel()),
        }
    return result


def _calibration_rows(
    rows: list[dict[str, Any]], token_budget: int
) -> list[dict[str, Any]]:
    selected = []
    used = 0
    for source in rows:
        if used >= token_budget:
            break
        row = dict(source)
        take = min(int(row["loss_token_limit"]), token_budget - used)
        row["loss_token_limit"] = take
        selected.append(row)
        used += take
    if used != token_budget:
        raise RuntimeError("calibration corpus cannot satisfy its exact token budget")
    return selected


def _fingerprint(
    config_path: Path,
    corpus_path: Path,
    model_dir: Path,
    global_kind: str,
    stage_name: str,
) -> str:
    return hashlib.sha256(
        canonical(
            {
                "config": file_digest(config_path),
                "corpus": file_digest(corpus_path),
                "model": str(model_dir.resolve()),
                "global_kind": global_kind,
                "stage": stage_name,
            }
        ).encode()
    ).hexdigest()


def train_one_stage(
    model,
    structure,
    *,
    train_adapter: str,
    active_adapters: list[str],
    stage_spec,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint: Path,
    fingerprint: str,
    train_structure: bool,
    offload_optimizer: bool,
) -> tuple[float, int]:
    """Train one named adapter with optimizer/checkpoint restart support."""
    activate_named_adapters(model, active_adapters)
    freeze_all_except_adapter(model, train_adapter)
    validate_trainable_adapter_band(model, train_adapter, stage_spec)
    validate_expected_target_count(model, train_adapter, stage_spec)
    structure.requires_grad_(train_structure)
    structure.train(train_structure)
    parameter_groups = [
        {
            "params": [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            "lr": config["global_lora"]["learning_rate"],
        }
    ]
    if train_structure:
        parameter_groups.append(
            {
                "params": list(structure.parameters()),
                "lr": config["structure_learning_rate"],
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=0.0,
    )
    start = 0
    previous_elapsed = 0.0
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError(f"stale global checkpoint: {checkpoint}")
        _load_adapter_state(model, saved["adapter_state"])
        structure.load_state_dict(saved["structure_state"])
        optimizer.load_state_dict(saved["optimizer"])
        start = int(saved["next_row"])
        previous_elapsed = float(saved.get("elapsed_seconds", 0.0))
    if offload_optimizer:
        optimizer_state_to(optimizer, "cpu")
    interval = int(config["checkpoint_every_optimizer_steps"])
    began = time.monotonic()
    while start < len(rows):
        check_deadline()
        stop = min(len(rows), start + interval)
        forward = StructuralForward(model, structure)
        for batch in batches_for(
            rows[start:stop],
            "rowcol_component",
            config["global_max_sequence_length"],
            config["seed"] + start,
        ):
            device = next(model.parameters()).device
            optimizer.zero_grad(set_to_none=True)
            output = forward(
                input_ids=batch["input_ids"].to(device),
                layout=batch["layout"],
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                return_dict=True,
                use_cache=False,
            )
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("non-finite global-stage loss")
            output.loss.backward()
            if offload_optimizer:
                optimizer_state_to(optimizer, device)
            optimizer.step()
            if offload_optimizer:
                optimizer_state_to(optimizer, "cpu")
        start = stop
        temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "fingerprint": fingerprint,
                "next_row": start,
                "adapter_state": _adapter_state(model),
                "structure_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in structure.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "elapsed_seconds": previous_elapsed
                + (time.monotonic() - began),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    return previous_elapsed + (time.monotonic() - began), len(rows)


def build_global_model(
    global_kind: str,
    config: dict[str, Any],
    config_path: Path,
    model_dir: Path,
    work: Path,
    *,
    calibration_tokens: int | None = None,
):
    """Build or resume the two unmerged early/middle global adapters."""
    from peft import get_peft_model

    model, _ = load_base(model_dir)
    memory_constrained = needs_optimizer_offload(model)
    if memory_constrained:
        # Reentrant activation checkpointing retains fewer saved tensors on
        # the 24 GB profile. The faster non-reentrant path remains active on
        # 32+ GiB devices.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        model.enable_input_require_grads()
    structure = AdditiveStructureAdapter(
        hidden_size_from_config(model.config)
    ).cuda().to(torch.bfloat16)
    early, middle, _ = stage_specs_from_config(model.config)
    if global_kind == "reversed":
        perception_spec, interpretation_spec = middle, early
    else:
        perception_spec, interpretation_spec = early, middle
    model = get_peft_model(
        model,
        adapter_config(perception_spec, config["global_lora"]),
        adapter_name="perception",
    )
    model.add_adapter(
        "interpretation",
        adapter_config(interpretation_spec, config["global_lora"]),
    )
    root = work / (
        f"calibration/global/{global_kind}"
        if calibration_tokens
        else f"global/{global_kind}"
    )
    root.mkdir(parents=True, exist_ok=True)
    timings = {}
    frozen_digest = None
    for stage_name, spec, active in (
        ("perception", perception_spec, ["perception"]),
        (
            "interpretation",
            interpretation_spec,
            ["perception", "interpretation"],
        ),
    ):
        if global_kind == "mixed":
            corpus_name = f"mixed_{stage_name}"
        elif global_kind == "shuffled" and stage_name == "interpretation":
            corpus_name = "shuffled_interpretation"
        else:
            corpus_name = f"staged_{stage_name}"
        corpus_path = work / "manifests" / "corpora" / f"{corpus_name}.jsonl"
        rows = _rows(corpus_path)
        if calibration_tokens is not None:
            rows = _calibration_rows(rows, calibration_tokens)
        checkpoint = root / f"{stage_name}.pt"
        before = (
            adapter_tensor_digest(model, "perception")
            if stage_name == "interpretation"
            else None
        )
        structure_before = (
            module_digest(structure) if stage_name == "interpretation" else None
        )
        elapsed, count = train_one_stage(
            model,
            structure,
            train_adapter=stage_name,
            active_adapters=active,
            stage_spec=spec,
            rows=rows,
            config=config,
            checkpoint=checkpoint,
            fingerprint=_fingerprint(
                config_path, corpus_path, model_dir, global_kind, stage_name
            )
            + (f":calibration:{calibration_tokens}" if calibration_tokens else ""),
            train_structure=stage_name == "perception",
            offload_optimizer=memory_constrained,
        )
        if before is not None:
            assert_frozen_digest_unchanged(model, "perception", before)
        if structure_before is not None and module_digest(structure) != structure_before:
            raise AssertionError("frozen 2D structure adapter changed")
        timings[stage_name] = {"seconds": elapsed, "rows": count}
        frozen_digest = adapter_tensor_digest(model, stage_name)
        atomic_json(
            root / f"{stage_name}.json",
            {
                "adapter": stage_name,
                "active_adapters": active,
                "band": list(spec.layer_indices),
                "tensor_sha256": frozen_digest,
                "structure_sha256": module_digest(structure),
                **timings[stage_name],
            },
        )
    activate_named_adapters(model, ["perception", "interpretation"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    structure.freeze()
    return model, structure, timings


def train_task(
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    task_band,
    config: dict[str, Any],
    probes: list[dict[str, Any]],
    *,
    steps: int | None = None,
):
    """Add, train, and return a reversible named task adapter."""
    torch.manual_seed(config["seed"])
    model.add_adapter("task", adapter_config(task_band, config["task_local"]))
    activate_named_adapters(model, ["perception", "interpretation", "task"])
    freeze_all_except_adapter(model, "task")
    validate_trainable_adapter_band(model, "task", task_band)
    validate_expected_target_count(model, "task", task_band)
    probe_pre = probe_snapshot(model, structure, probes, config)
    pre = train_diagnostic(
        model, structure, task, task_id, "rowcol_component", config
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["task_local"]["learning_rate"],
        weight_decay=0.0,
    )
    offload_optimizer = needs_optimizer_offload(model)
    all_rows = list(training_rows(task, task_id, config))
    count = config["task_local"]["steps"] if steps is None else steps
    if len(all_rows) < count:
        raise RuntimeError("task augmentation produced too few rows")
    started = time.monotonic()
    model.train()
    structure.eval()
    forward = StructuralForward(model, structure)
    for index, row in enumerate(all_rows[:count]):
        batch = collate_token_rows(
            [row],
            arm="rowcol_component",
            shuffle_seed=config["seed"] + index,
        )
        check_deadline()
        device = next(model.parameters()).device
        optimizer.zero_grad(set_to_none=True)
        output = forward(
            input_ids=batch["input_ids"].to(device),
            layout=batch["layout"],
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            return_dict=True,
            use_cache=False,
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("non-finite task-stage loss")
        output.loss.backward()
        if offload_optimizer:
            optimizer_state_to(optimizer, device)
        optimizer.step()
        if offload_optimizer:
            optimizer_state_to(optimizer, "cpu")
    elapsed = time.monotonic() - started
    post = train_diagnostic(
        model, structure, task, task_id, "rowcol_component", config
    )
    probe_post = probe_snapshot(model, structure, probes, config)
    return pre, post, elapsed, {"before": probe_pre, "after": probe_post}


def _candidate_fingerprint(
    config_path: Path,
    challenges_path: Path,
    arm: str,
    task_id: str,
    global_digests: dict[str, str],
) -> str:
    return hashlib.sha256(
        canonical(
            {
                "config_sha256": file_digest(config_path),
                "challenges_sha256": file_digest(challenges_path),
                "worker_sha256": file_digest(Path(__file__)),
                "arm": arm,
                "task_id": task_id,
                "global_adapter_sha256": global_digests,
            }
        ).encode()
    ).hexdigest()


def existing_candidate(path: Path, fingerprint: str) -> bool:
    marker = path.with_suffix(".sha256")
    if not path.is_file() and not marker.is_file():
        return False
    if not path.is_file() or not marker.is_file():
        raise RuntimeError(f"orphaned candidate checkpoint: {path}")
    metadata = json.loads(marker.read_text())
    if metadata.get("fingerprint") != fingerprint:
        raise RuntimeError(f"stale candidate checkpoint: {path}")
    if metadata.get("sha256") != file_digest(path):
        raise RuntimeError(f"candidate digest mismatch: {path}")
    return True


def write_candidate(path: Path, payload: dict[str, Any], fingerprint: str) -> None:
    atomic_json(path, payload)
    atomic_json(
        path.with_suffix(".sha256"),
        {"fingerprint": fingerprint, "sha256": file_digest(path)},
    )


def run_arm(
    arm: str,
    model,
    structure,
    challenges: dict[str, Any],
    task_ids: Iterable[str],
    config: dict[str, Any],
    config_path: Path,
    challenges_path: Path,
    work: Path,
) -> None:
    early, _, late = stage_specs_from_config(model.config)
    task_band = early if arm == "staged_early_collision" else late
    global_before = {
        name: adapter_tensor_digest(model, name)
        for name in ("perception", "interpretation")
    }
    probes = probe_rows(work)
    probe_path = work / "arms" / arm / "global_probe_metrics.json"
    if not probe_path.is_file():
        atomic_json(
            probe_path,
            probe_snapshot(model, structure, probes, config),
        )
    for task_id in task_ids:
        check_deadline()
        destination = work / "arms" / arm / "candidates" / f"{task_id}.json"
        fingerprint = _candidate_fingerprint(
            config_path, challenges_path, arm, task_id, global_before
        )
        if existing_candidate(destination, fingerprint):
            continue
        pre, post, _, retention = train_task(
            model,
            structure,
            challenges[task_id],
            task_id,
            task_band,
            config,
            probes,
        )
        try:
            payload = _evaluate_trained_task_impl(
                model,
                structure,
                challenges[task_id],
                task_id,
                "rowcol_component",
                config,
                pre=pre,
                post=post,
                merge_for_inference=False,
            )
            payload["experiment_arm"] = arm
            payload["task_band"] = task_band.name
            payload["active_named_adapters"] = [
                "perception",
                "interpretation",
                "task",
            ]
            payload["probe_retention"] = retention
            write_candidate(destination, payload, fingerprint)
        finally:
            model.delete_adapter("task")
            activate_named_adapters(model, ["perception", "interpretation"])
        for name, expected in global_before.items():
            assert_frozen_digest_unchanged(model, name, expected)
    hashes = {
        task_id: file_digest(
            work / "arms" / arm / "candidates" / f"{task_id}.json"
        )
        for task_id in task_ids
    }
    atomic_json(work / "arms" / arm / "COMPLETE.json", hashes)


def project_full_runtime(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    primary_item_count: int,
    control_item_count: int,
    global_sample_seconds: float,
    global_sample_tokens_per_stage: int,
    task_sample_total_seconds: float,
    dfs_sample_seconds: float,
) -> float:
    global_scale = sum(config["stage_budgets"].values()) / (
        2 * global_sample_tokens_per_stage
    )
    task_scale = config["task_local"]["steps"] / config["calibration"][
        "task_local_steps"
    ]
    task_runs = (
        len(config["arms"]) * len(manifest["primary_ids"])
        + len(config["control_arms"]) * len(manifest["control_ids"])
    )
    item_runs = (
        primary_item_count * len(config["arms"])
        + control_item_count * len(config["control_arms"])
    )
    view_count = (
        config["task_local"]["decode_color_permutations"]
        * config["task_local"]["decode_geometries"]
        + config["task_local"]["rescore_views"]
    )
    return (
        global_sample_seconds * global_scale
        + task_runs * task_sample_total_seconds * task_scale
        + item_runs * view_count * dfs_sample_seconds
    )


def calibrate(
    config: dict[str, Any],
    config_path: Path,
    challenges: dict[str, Any],
    manifest: dict[str, Any],
    model_dir: Path,
    work: Path,
) -> None:
    token_budget = config["calibration"]["global_loss_tokens_per_stage"]
    started = time.monotonic()
    calibration_timings = {}
    for global_kind in ("staged", "mixed", "reversed", "shuffled"):
        current_model, current_structure, timings = build_global_model(
            global_kind,
            config,
            config_path,
            model_dir,
            work,
            calibration_tokens=token_budget,
        )
        calibration_timings[global_kind] = timings
        del current_model, current_structure
        torch.cuda.empty_cache()
    # Reload the completed staged checkpoints only after all four global
    # compositions have been calibrated, so two 4B models never coexist.
    model, structure, _ = build_global_model(
        "staged",
        config,
        config_path,
        model_dir,
        work,
        calibration_tokens=token_budget,
    )
    task_id = manifest["primary_ids"][0]
    task = challenges[task_id]
    _, _, late = stage_specs_from_config(model.config)
    task_started = time.monotonic()
    pre, post, task_seconds, retention = train_task(
        model,
        structure,
        task,
        task_id,
        late,
        config,
        probe_rows(work),
        steps=config["calibration"]["task_local_steps"],
    )
    task_total_seconds = time.monotonic() - task_started
    test = task["test"][0]
    prefix, _, _ = serialize_chatml_bounded(
        task["train"],
        test["input"],
        max_length=config["global_max_sequence_length"],
        reserve_new_tokens=config["dfs"]["max_new_tokens"],
    )
    prefix_ids = torch.tensor([prefix], device="cuda")
    prefix_layout = LayoutMetadata.stack(
        [build_layout(prefix, arm="rowcol_component")]
    )
    with torch.no_grad():
        direct_logits = model(
            input_ids=prefix_ids,
            return_dict=True,
            use_cache=False,
        ).logits
        scale_zero_logits = StructuralForward(
            model, structure, scale=0.0
        )(
            input_ids=prefix_ids,
            layout=prefix_layout,
            return_dict=True,
            use_cache=False,
        ).logits
    scale_zero_parity = bool(
        torch.allclose(direct_logits, scale_zero_logits, atol=1e-3, rtol=1e-3)
    )
    if not scale_zero_parity:
        raise RuntimeError("2D structure scale-zero parity failed")
    dfs_started = time.monotonic()
    result = deterministic_cached_dfs(
        model,
        structure,
        prefix_ids,
        prefix_layout,
        grid_bucket=2 * len(task["train"]) + 2,
        budget=DFSBudget(
            max_nodes=config["calibration"]["dfs_nodes"],
            max_model_forward_tokens=config["dfs"][
                "max_model_forward_tokens_per_view"
            ],
            max_candidates=config["dfs"]["max_candidates_per_view"],
            max_new_tokens=config["dfs"]["max_new_tokens"],
            emergency_seconds=config["dfs"]["emergency_seconds_per_view"],
            min_prob=config["dfs"]["min_prob"],
            deadline_epoch=deadline(),
        ),
        generated_arm="rowcol_component",
    )
    dfs_seconds = time.monotonic() - dfs_started
    replay = deterministic_cached_dfs(
        model,
        structure,
        prefix_ids,
        prefix_layout,
        grid_bucket=2 * len(task["train"]) + 2,
        budget=DFSBudget(
            max_nodes=config["calibration"]["dfs_nodes"],
            max_model_forward_tokens=config["dfs"][
                "max_model_forward_tokens_per_view"
            ],
            max_candidates=config["dfs"]["max_candidates_per_view"],
            max_new_tokens=config["dfs"]["max_new_tokens"],
            emergency_seconds=config["dfs"]["emergency_seconds_per_view"],
            min_prob=config["dfs"]["min_prob"],
            deadline_epoch=deadline(),
        ),
        generated_arm="rowcol_component",
    )
    deterministic_replay = canonical(result.candidates) == canonical(
        replay.candidates
    )
    if not deterministic_replay:
        raise RuntimeError("calibration DFS replay was not deterministic")
    model.delete_adapter("task")

    global_seconds = sum(
        item["seconds"]
        for timings in calibration_timings.values()
        for item in timings.values()
    )
    primary_item_count = sum(
        len(challenges[task_id]["test"]) for task_id in manifest["primary_ids"]
    )
    control_item_count = sum(
        len(challenges[task_id]["test"]) for task_id in manifest["control_ids"]
    )
    # Calibration and production use the same model-forward-token cap per
    # view. The sample stopped at 42 nodes because that cap/frontier ended,
    # not because its 512-node cap was reached. Scaling elapsed time to the
    # production max_nodes value would count work the token cap cannot admit.
    projected = project_full_runtime(
        config,
        manifest,
        primary_item_count=primary_item_count,
        control_item_count=control_item_count,
        global_sample_seconds=global_seconds,
        global_sample_tokens_per_stage=token_budget,
        task_sample_total_seconds=task_total_seconds,
        dfs_sample_seconds=dfs_seconds,
    )
    atomic_json(
        work / "calibration" / "result.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "global_sample_seconds": global_seconds,
            "global_sample_timings": calibration_timings,
            "task_sample_seconds": task_seconds,
            "task_sample_total_seconds": task_total_seconds,
            "dfs_sample_seconds": dfs_seconds,
            "dfs_sample_nodes": result.nodes_used,
            "projected_full_seconds": projected,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "total_gpu_memory_bytes": torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory,
            "task_id": task_id,
            "pre": pre,
            "post": post,
            "probe_retention": retention,
            "checks": {
                "tokenizer_contract": True,
                "scale_zero_parity": scale_zero_parity,
                "all_global_compositions_trained": set(
                    calibration_timings
                )
                == {"staged", "mixed", "reversed", "shuffled"},
                "probe_isolation": True,
                "collision_audit": True,
                "deterministic_replay": deterministic_replay,
                "decode_without_oom_or_nan": True,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "calibrate",
            "pilot",
            "exploratory",
            "hard_exploratory",
            "full",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    config = runtime_config(json.loads(args.config.read_text()))
    challenges = json.loads(args.challenges.read_text())
    if any("output" in item for task in challenges.values() for item in task["test"]):
        raise ValueError("worker challenge contains test labels")
    manifest = json.loads(
        (args.work / "manifests" / "task_manifest.json").read_text()
    )
    task_ids = manifest["primary_ids"]
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.mode == "calibrate":
        if (args.work / "calibration" / "result.json").is_file():
            return
        calibrate(
            config,
            args.config,
            challenges,
            manifest,
            args.model_dir,
            args.work,
        )
        return
    if args.mode == "pilot":
        task_ids = manifest["pilot_ids"]
    elif args.mode == "exploratory":
        task_ids = manifest["control_ids"]
    elif args.mode == "hard_exploratory":
        task_ids = json.loads(
            (args.work / "manifests" / "hard_exploratory_ids.json").read_text()
        )

    # staged_late and staged_early_collision intentionally share one immutable
    # global checkpoint. mixed_late receives an independently trained control.
    staged_model, staged_structure, _ = build_global_model(
        "staged", config, args.config, args.model_dir, args.work
    )
    run_arm(
        "staged_late",
        staged_model,
        staged_structure,
        challenges,
        task_ids,
        config,
        args.config,
        args.challenges,
        args.work,
    )
    run_arm(
        "staged_early_collision",
        staged_model,
        staged_structure,
        challenges,
        task_ids,
        config,
        args.config,
        args.challenges,
        args.work,
    )
    del staged_model, staged_structure
    torch.cuda.empty_cache()
    mixed_model, mixed_structure, _ = build_global_model(
        "mixed", config, args.config, args.model_dir, args.work
    )
    run_arm(
        "mixed_late",
        mixed_model,
        mixed_structure,
        challenges,
        task_ids,
        config,
        args.config,
        args.challenges,
        args.work,
    )
    del mixed_model, mixed_structure
    torch.cuda.empty_cache()
    if args.mode == "full":
        for arm, global_kind in (
            ("reversed_bands", "reversed"),
            ("shuffled_interpretation", "shuffled"),
        ):
            control_model, control_structure, _ = build_global_model(
                global_kind,
                config,
                args.config,
                args.model_dir,
                args.work,
            )
            run_arm(
                arm,
                control_model,
                control_structure,
                challenges,
                manifest["control_ids"],
                config,
                args.config,
                args.challenges,
                args.work,
            )
            del control_model, control_structure
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
