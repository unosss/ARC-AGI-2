"""Label-blind GPU worker and offline smoke test for Experiment 035."""

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

from arc_solver.qwen2d_adapter import StructuralForward, hidden_size_from_config
from arc_solver.qwen2d_experiment import refiner_prompt_payload
from arc_solver.qwen2d_metadata import LayoutMetadata, build_layout
from arc_solver.qwen2d_refiner import (
    Qwen2DRefiner,
    RefinerConfig,
    decode_stages,
    refiner_loss,
)
from arc_solver.qwen2d_runtime import serialize_chatml_bounded
from arc_solver.qwen_staged_adapters import (
    activate_named_adapters,
    stage_specs_from_config,
)
from kaggle_runs.qwen_2d_grid_adapter_l4.gpu_arm_worker import (
    _evaluate_trained_task_impl,
    score_candidate_views,
)
from kaggle_runs.qwen_staged_adapters_runpod.worker import (
    build_global_model,
    train_task,
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical(value) + "\n")
    os.replace(temporary, path)


def reject_test_outputs(challenges: dict[str, Any]) -> None:
    if any(
        "output" in item
        for task in challenges.values()
        for item in task.get("test", [])
    ):
        raise ValueError("worker challenge contains test outputs")


def deadline() -> float | None:
    value = os.environ.get("ARC035_DEADLINE")
    return None if value is None else float(value) - 120.0


def check_deadline() -> None:
    limit = deadline()
    if limit is not None and time.time() >= limit:
        raise RuntimeError("Experiment 035 deadline reached; restart is safe")


def runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["global_max_sequence_length"] = config["corpus"][
        "global_max_sequence_length"
    ]
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def refiner_config(model, config: dict[str, Any]) -> RefinerConfig:
    values = config["refiner"]
    return RefinerConfig(
        hidden_size=hidden_size_from_config(model.config),
        model_dim=values["model_dim"],
        latent_dim=values["latent_dim"],
        latent_slots=values["latent_slots"],
        num_heads=values["num_heads"],
        ff_multiplier=values["ff_multiplier"],
        num_stages=values["num_stages"],
    )


def serialize_label_blind(
    row: dict[str, Any], config: dict[str, Any]
) -> tuple[list[int], LayoutMetadata]:
    prompt = refiner_prompt_payload(row)
    tokens, _, _ = serialize_chatml_bounded(
        prompt["train"],
        prompt["query"],
        target=None,
        max_length=config["global_max_sequence_length"],
    )
    return tokens, LayoutMetadata.stack(
        [build_layout(tokens, arm="rowcol_component")]
    )


@torch.no_grad()
def extract_hidden(
    model,
    structure,
    row: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract final Qwen states from demos/query only; no teacher target."""
    tokens, layout = serialize_label_blind(row, config)
    device = next(model.parameters()).device
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    output = StructuralForward(model, structure)(
        input_ids=ids,
        layout=layout,
        attention_mask=torch.ones_like(ids),
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden = output.hidden_states[-1].detach()
    return hidden, torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)


def train_row(
    refiner: Qwen2DRefiner,
    model,
    structure,
    row: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> dict[str, float]:
    prompt = refiner_prompt_payload(row)
    hidden, mask = extract_hidden(model, structure, row, config)
    draft = torch.tensor([prompt["draft"]], dtype=torch.long, device=hidden.device)
    optimizer.zero_grad(set_to_none=True)
    output = refiner(hidden, mask, draft)
    losses = refiner_loss(
        output,
        [row["target"]],
        stage_weights=config["refiner"]["stage_weights"],
        shape_weight=config["refiner"]["shape_weight"],
    )
    if not torch.isfinite(losses.loss):
        raise RuntimeError("non-finite refiner loss")
    losses.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        refiner.parameters(), config["refiner"]["gradient_clip"]
    )
    optimizer.step()
    return {
        "loss": float(losses.loss.detach()),
        "cell_loss": float(losses.cell_loss.detach()),
        "shape_loss": float(losses.shape_loss.detach()),
    }


def refiner_fingerprint(
    config_path: Path, corpus_path: Path, model_dir: Path, suffix: str
) -> str:
    return hashlib.sha256(canonical({
        "config": file_digest(config_path),
        "corpus": file_digest(corpus_path),
        "model": str(model_dir.resolve()),
        "suffix": suffix,
        "worker": file_digest(Path(__file__)),
    }).encode()).hexdigest()


def train_global_refiner(
    model,
    structure,
    config: dict[str, Any],
    config_path: Path,
    model_dir: Path,
    work: Path,
    *,
    calibration: bool = False,
) -> tuple[Qwen2DRefiner, dict[str, Any]]:
    corpus_path = work / "manifests" / "corpora" / "global_refiner.jsonl"
    source = rows(corpus_path)
    if calibration:
        source = source[: config["calibration"]["refiner_rows"]]
    root = work / ("calibration" if calibration else "global")
    checkpoint = root / "refiner.pt"
    fingerprint = refiner_fingerprint(
        config_path, corpus_path, model_dir, "calibration" if calibration else "full"
    )
    refiner = Qwen2DRefiner(refiner_config(model, config)).cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=config["refiner"]["global_learning_rate"],
        weight_decay=0.0,
    )
    start = 0
    losses: list[dict[str, float]] = []
    sequence = source * config["refiner"]["global_epochs"]
    elapsed_before = 0.0
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("stale global refiner checkpoint")
        refiner.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.cuda()
        start = saved["next_row"]
        elapsed_before = saved.get("elapsed_seconds", 0.0)
        losses = saved.get("recent_losses", [])
    began = time.monotonic()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    structure.freeze()
    interval = config["checkpoint_every_optimizer_steps"]
    for index in range(start, len(sequence)):
        check_deadline()
        refiner.train()
        losses.append(train_row(refiner, model, structure, sequence[index], optimizer, config))
        losses = losses[-32:]
        if (index + 1) % interval == 0 or index + 1 == len(sequence):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
            torch.save({
                "fingerprint": fingerprint,
                "next_row": index + 1,
                "model": {k: v.detach().cpu() for k, v in refiner.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "elapsed_seconds": elapsed_before + time.monotonic() - began,
                "recent_losses": losses,
            }, temporary)
            os.replace(temporary, checkpoint)
    refiner.eval()
    return refiner, {
        "rows": len(sequence),
        "elapsed_seconds": elapsed_before + time.monotonic() - began,
        "recent_losses": losses,
        "checkpoint_sha256": file_digest(checkpoint),
    }


def empty_diagnostic() -> dict[str, Any]:
    return {"canonical_loo_mean_sum_nll": None, "folds": 0}


def generate_drafts(
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Use the existing canonicalized DFS/rescore path and retain its top two."""
    started = time.monotonic()
    payload = _evaluate_trained_task_impl(
        model,
        structure,
        task,
        task_id,
        "rowcol_component",
        config,
        pre=empty_diagnostic(),
        post=empty_diagnostic(),
        merge_for_inference=False,
    )
    for item in payload["ranked_candidates"]:
        del item[2:]
    payload["timing"] = {"draft_generation_seconds": time.monotonic() - started}
    return payload


def loo_rows(
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate held-out drafts before attaching the held output as target."""
    result = []
    for held_index, held in enumerate(task["train"]):
        demos = [
            copy.deepcopy(pair)
            for index, pair in enumerate(task["train"])
            if index != held_index
        ]
        blind_task = {"train": demos, "test": [{"input": copy.deepcopy(held["input"])}]}
        generated = generate_drafts(
            model, structure, blind_task, f"{task_id}:loo:{held_index}", config
        )
        candidates = generated["ranked_candidates"][0]
        if not candidates:
            # A search miss must not abort task adaptation or tempt us to use
            # the held target as an oracle draft.  The held input is the only
            # always-available, label-blind rectangular fallback.
            candidates = [{"grid": copy.deepcopy(held["input"]), "source": "input_fallback"}]
        for rank, candidate in enumerate(candidates[:2]):
            # The target is attached only after generation has completed.
            result.append({
                "row_id": f"task:{task_id}:loo:{held_index}:draft:{rank}",
                "train": demos,
                "query": copy.deepcopy(held["input"]),
                "draft": copy.deepcopy(candidate["grid"]),
                "target": copy.deepcopy(held["output"]),
                "held_index": held_index,
                "draft_rank": rank,
                "draft_source": candidate.get("source", "dfs"),
            })
    return result


def adapt_task_refiner(
    global_refiner: Qwen2DRefiner,
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    config: dict[str, Any],
) -> tuple[Qwen2DRefiner, dict[str, Any]]:
    started = time.monotonic()
    folds = loo_rows(model, structure, task, task_id, config)
    task_refiner = Qwen2DRefiner(global_refiner.config).to(
        device=next(global_refiner.parameters()).device,
        dtype=next(global_refiner.parameters()).dtype,
    )
    task_refiner.load_state_dict(global_refiner.state_dict())
    optimizer = torch.optim.AdamW(
        task_refiner.parameters(),
        lr=config["refiner"]["task_learning_rate"],
        weight_decay=0.0,
    )
    losses = []
    for step in range(config["refiner"]["task_steps"]):
        check_deadline()
        task_refiner.train()
        losses.append(train_row(
            task_refiner, model, structure, folds[step % len(folds)], optimizer, config
        ))
    task_refiner.eval()
    return task_refiner, {
        "folds": len(task["train"]),
        "draft_rows": len(folds),
        "steps": config["refiner"]["task_steps"],
        "qwen_frozen": True,
        "held_output_absent_during_draft_generation": True,
        "elapsed_seconds": time.monotonic() - started,
        "recent_losses": losses[-8:],
    }


def grid_edits(left: list[list[int]], right: list[list[int]]) -> int:
    height, width = max(len(left), len(right)), max(len(left[0]), len(right[0]))
    return sum(
        (left[r][c] if r < len(left) and c < len(left[r]) else None)
        != (right[r][c] if r < len(right) and c < len(right[r]) else None)
        for r in range(height) for c in range(width)
    )


@torch.no_grad()
def refine_draft(
    refiner: Qwen2DRefiner,
    model,
    structure,
    task: dict[str, Any],
    query: list[list[int]],
    draft: list[list[int]],
    config: dict[str, Any],
) -> tuple[list[list[list[int]]], dict[str, Any]]:
    row = {"train": task["train"], "query": query, "draft": draft}
    hidden, mask = extract_hidden(model, structure, row, config)
    canvas = torch.tensor([draft], dtype=torch.long, device=hidden.device)
    output = refiner(hidden, mask, canvas)
    trajectory = [stage[0] for stage in decode_stages(output, draft_canvas=canvas)]
    diagnostics = []
    previous = draft
    for stage, grid in zip(output.stages, trajectory):
        height, width = len(grid), len(grid[0])
        diagnostics.append({
            "mean_confidence": float(stage.confidence[0, :height, :width].mean()),
            "mean_update_gate": float(stage.update_gate[0, :height, :width].mean()),
            "changed_cells_from_previous": grid_edits(previous, grid),
            "predicted_shape": [height, width],
        })
        previous = grid
    return trajectory, {"stages": diagnostics}


def refine_payload(
    base: dict[str, Any],
    refiner: Qwen2DRefiner,
    model,
    structure,
    task: dict[str, Any],
    task_id: str,
    arm: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    ranked, trajectories = [], []
    for test_index, (test, drafts) in enumerate(
        zip(task["test"], base["ranked_candidates"])
    ):
        pool: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}
        item_trajectories = []
        for rank, candidate in enumerate(drafts[:2]):
            draft = candidate["grid"]
            stages, diagnostics = refine_draft(
                refiner, model, structure, task, test["input"], draft, config
            )
            item_trajectories.append({
                "draft_rank": rank,
                "G0": draft,
                "G1": stages[0],
                "G2": stages[1],
                "G3": stages[2],
                **diagnostics,
            })
            for stage_index, grid in enumerate([draft, *stages]):
                key = tuple(map(tuple, grid))
                pool.setdefault(key, {
                    "grid": grid,
                    "sources": [],
                    "stage_trajectory_refs": [],
                })
                pool[key]["sources"].append(f"draft{rank}:G{stage_index}")
                pool[key]["stage_trajectory_refs"].append(
                    {"draft_rank": rank, "stage": stage_index}
                )
        scored = []
        for item in pool.values():
            view = score_candidate_views(
                model, structure, task, task_id, test_index,
                item["grid"], "rowcol_component", config,
            )
            scored.append({
                **item,
                "rescore": view,
                "score": -view["mean_sum_nll"],
                "votes": len(item["sources"]),
                "generation_nll": [],
            })
        scored.sort(key=lambda item: (-item["score"], canonical(item["grid"])))
        ranked.append(scored)
        trajectories.append(item_trajectories)
    return {
        **{key: value for key, value in base.items() if key != "ranked_candidates"},
        "task_id": task_id,
        "arm": arm,
        "experiment_arm": arm,
        "ranked_candidates": ranked,
        "stage_trajectories": trajectories,
        "refiner_diagnostics": {
            "stages": 3,
            "initial_drafts_per_item": 2,
            "merged_generations": ["G0", "G1", "G2", "G3"],
            "canonical_drafts": True,
            "rescored_with_existing_views": True,
        },
        "timing": {
            **base.get("timing", {}),
            "refinement_and_rescore_seconds": time.monotonic() - started,
        },
    }


def candidate_fingerprint(
    config_path: Path, challenges_path: Path, task_id: str, global_checkpoint: Path
) -> str:
    return hashlib.sha256(canonical({
        "config": file_digest(config_path),
        "challenges": file_digest(challenges_path),
        "task_id": task_id,
        "global_refiner": file_digest(global_checkpoint),
        "worker": file_digest(Path(__file__)),
    }).encode()).hexdigest()


def existing_candidate(path: Path, fingerprint: str) -> bool:
    marker = path.with_suffix(".sha256")
    if not path.exists() and not marker.exists():
        return False
    if not path.is_file() or not marker.is_file():
        raise RuntimeError(f"orphaned candidate artifact: {path}")
    metadata = json.loads(marker.read_text())
    if metadata.get("fingerprint") != fingerprint:
        raise RuntimeError(f"stale candidate artifact: {path}")
    if metadata.get("sha256") != file_digest(path):
        raise RuntimeError(f"candidate digest mismatch: {path}")
    return True


def write_candidate(path: Path, payload: dict[str, Any], fingerprint: str) -> None:
    atomic_json(path, payload)
    atomic_json(path.with_suffix(".sha256"), {
        "fingerprint": fingerprint, "sha256": file_digest(path)
    })


def run_tasks(
    model,
    structure,
    global_refiner: Qwen2DRefiner,
    challenges: dict[str, Any],
    task_ids: Iterable[str],
    config: dict[str, Any],
    config_path: Path,
    challenges_path: Path,
    work: Path,
) -> None:
    global_checkpoint = work / "global" / "refiner.pt"
    for task_id in task_ids:
        check_deadline()
        fingerprint = candidate_fingerprint(
            config_path, challenges_path, task_id, global_checkpoint
        )
        destinations = {
            arm: work / "arms" / arm / "candidates" / f"{task_id}.json"
            for arm in config["arms"]
        }
        if all(existing_candidate(path, fingerprint) for path in destinations.values()):
            continue
        task = challenges[task_id]
        # Refiner adaptation uses genuinely held-out drafts from the immutable
        # global Qwen.  The per-task Qwen adapter is deliberately added only
        # afterwards, otherwise the held output could already be in its weights.
        task_refiner, adaptation = adapt_task_refiner(
            global_refiner, model, structure, task, task_id, config
        )
        _, _, late = stage_specs_from_config(model.config)
        train_task(
            model,
            structure,
            task,
            task_id,
            late,
            config,
            [],
            steps=config["task_local"]["steps"],
        )
        try:
            base = generate_drafts(model, structure, task, task_id, config)
            baseline = {
                **base,
                "arm": "baseline",
                "experiment_arm": "baseline",
                "stage_trajectories": [
                    [
                        {"draft_rank": rank, "G0": item["grid"]}
                        for rank, item in enumerate(items)
                    ]
                    for items in base["ranked_candidates"]
                ],
                "refiner_diagnostics": {"stages": 0, "initial_drafts_per_item": 2},
            }
            global_payload = refine_payload(
                base, global_refiner, model, structure, task, task_id,
                "global_refiner", config,
            )
            task_payload = refine_payload(
                base, task_refiner, model, structure, task, task_id,
                "task_refiner", config,
            )
            task_payload["task_adaptation"] = adaptation
            for arm, payload in (
                ("baseline", baseline),
                ("global_refiner", global_payload),
                ("task_refiner", task_payload),
            ):
                if not existing_candidate(destinations[arm], fingerprint):
                    write_candidate(destinations[arm], payload, fingerprint)
        finally:
            model.delete_adapter("task")
            activate_named_adapters(model, ["perception", "interpretation"])
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            del task_refiner
    for arm in config["arms"]:
        atomic_json(work / "arms" / arm / "COMPLETE.json", {
            task_id: file_digest(
                work / "arms" / arm / "candidates" / f"{task_id}.json"
            )
            for task_id in task_ids
        })


def calibrate(
    model,
    structure,
    challenges: dict[str, Any],
    manifest: dict[str, Any],
    staged_timings: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    model_dir: Path,
    work: Path,
) -> None:
    destination = work / "calibration" / "result.json"
    if destination.is_file():
        return
    started = time.monotonic()
    calibration_refiner, training = train_global_refiner(
        model, structure, config, config_path, model_dir, work, calibration=True
    )
    sample_config = copy.deepcopy(config)
    sample_config["dfs"]["max_nodes_per_view"] = config["calibration"]["dfs_nodes"]
    sample_config["task_local"]["decode_geometries"] = 1
    sample_config["task_local"]["rescore_views"] = 1
    sample_config["refiner"]["task_steps"] = config["calibration"]["task_steps"]
    task_id = manifest["primary_ids"][0]
    task_started = time.monotonic()
    task_refiner, adaptation = adapt_task_refiner(
        calibration_refiner, model, structure, challenges[task_id],
        task_id, sample_config,
    )
    _, _, late = stage_specs_from_config(model.config)
    train_task(
        model,
        structure,
        challenges[task_id],
        task_id,
        late,
        sample_config,
        [],
        steps=config["calibration"]["task_steps"],
    )
    try:
        base = generate_drafts(
            model, structure, challenges[task_id], task_id, sample_config
        )
        refine_payload(
            base, task_refiner, model, structure, challenges[task_id],
            task_id, "task_refiner", sample_config,
        )
    finally:
        model.delete_adapter("task")
        activate_named_adapters(model, ["perception", "interpretation"])
    task_seconds = time.monotonic() - task_started
    elapsed = time.monotonic() - started
    full_rows = (
        config["corpus"]["global_refiner_rows"]
        * config["refiner"]["global_epochs"]
    )
    staged_sample_seconds = sum(
        float(item["seconds"]) for item in staged_timings.values()
    )
    staged_scale = sum(config["stage_budgets"].values()) / (
        2 * config["calibration"]["global_loss_tokens_per_stage"]
    )
    projected = (
        staged_sample_seconds * staged_scale
        + training["elapsed_seconds"] * full_rows / max(1, training["rows"])
        + task_seconds * len(manifest["primary_ids"])
    )
    atomic_json(destination, {
        "calibration_mode": config["calibration"]["mode"],
        "elapsed_seconds": elapsed,
        "sample_refiner_rows": training["rows"],
        "sample_task_id": task_id,
        "sample_task_seconds": task_seconds,
        "sample_task_adaptation": adaptation,
        "staged_sample_timings": staged_timings,
        "projected_full_seconds": projected,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "total_gpu_memory_bytes": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory,
        "checks": {
            "output_hidden_states": True,
            "label_blind_prompts": True,
            "qwen_frozen": True,
            "three_refinement_stages": True,
        },
    })


def smoke(config_path: Path, work: Path) -> None:
    """CPU-only architecture/loss/checkpoint smoke; never imports Transformers."""
    torch.manual_seed(35)
    config = json.loads(config_path.read_text())
    tiny = Qwen2DRefiner(RefinerConfig(
        hidden_size=16, model_dim=16, latent_dim=16, latent_slots=4,
        num_heads=4, ff_multiplier=1, num_stages=3,
    ))
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=1e-3)
    hidden = torch.randn(1, 7, 16)
    mask = torch.ones(1, 7, dtype=torch.bool)
    draft = torch.tensor([[[1, 10], [0, 2]]])
    target = [[[1, 3], [0, 2]]]
    output = tiny(hidden, mask, draft)
    loss = refiner_loss(output, target)
    optimizer.zero_grad(set_to_none=True)
    loss.loss.backward()
    optimizer.step()
    decoded = decode_stages(output, draft_canvas=draft)
    rejected = False
    try:
        reject_test_outputs({"x": {"test": [{"input": [[0]], "output": [[0]]}]}})
    except ValueError:
        rejected = True
    artifact = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "mode": "cpu_offline_no_model",
        "passed": (
            len(output.stages) == 3
            and len(decoded) == 3
            and torch.isfinite(loss.loss).item()
            and rejected
        ),
        "loss": float(loss.loss.detach()),
        "trajectory_shapes": [
            [len(stage[0]), len(stage[0][0])] for stage in decoded
        ],
        "worker_rejects_test_outputs": rejected,
        "model_download_attempted": False,
    }
    atomic_json(work / "calibration" / "smoke.json", artifact)
    if not artifact["passed"]:
        raise RuntimeError("offline smoke failed")
    print(canonical(artifact))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "pilot", "full"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        smoke(args.config, args.work)
        return
    if args.mode is None or args.challenges is None or args.model_dir is None:
        parser.error("production mode requires --mode, --challenges, and --model-dir")
    config = runtime_config(json.loads(args.config.read_text()))
    challenges = json.loads(args.challenges.read_text())
    reject_test_outputs(challenges)
    manifest = json.loads(
        (args.work / "manifests" / "task_manifest.json").read_text()
    )
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    calibration_tokens = (
        config["calibration"]["global_loss_tokens_per_stage"]
        if args.mode == "calibrate" else None
    )
    model, structure, staged_timings = build_global_model(
        "staged", config, args.config, args.model_dir, args.work,
        calibration_tokens=calibration_tokens,
    )
    if args.mode == "calibrate":
        calibrate(
            model, structure, challenges, manifest, staged_timings, config,
            args.config, args.model_dir, args.work,
        )
        return
    global_refiner, _ = train_global_refiner(
        model, structure, config, args.config, args.model_dir, args.work
    )
    task_ids = (
        manifest["pilot_ids"] if args.mode == "pilot" else manifest["primary_ids"]
    )
    run_tasks(
        model, structure, global_refiner, challenges, task_ids, config,
        args.config, args.challenges, args.work,
    )


if __name__ == "__main__":
    main()
