"""Train one task-local adapter and emit sealed predictions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from core import (
    apply_template,
    augment,
    digest,
    file_sha256,
    messages,
    parse_grids,
    step_transform,
    supervised_encoding,
)


LANGUAGE_TARGETS = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
EXPECTED_MODULES = 252
EXPECTED_TASK_TRAINABLE = 132_120_576
RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "qwen-vl-utils",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def tensor_digest(model, adapter_name: str) -> str:
    checksum = hashlib.sha256()
    matched = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" in name and f".{adapter_name}." in name:
            checksum.update(name.encode())
            checksum.update(
                parameter.detach().to(dtype=torch.float32, device="cpu")
                .contiguous()
                .numpy()
                .tobytes()
            )
            matched += 1
    if matched == 0:
        raise RuntimeError(f"no tensors found for adapter {adapter_name}")
    return checksum.hexdigest()


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES
        },
    }


def model_identity(model_path: str | None, config: dict[str, Any]) -> dict[str, Any]:
    if model_path is None:
        return {
            "source": config["model"],
            "revision": config["model_revision"],
        }
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError("--model-path must be a directory")
    files = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise ValueError("--model-path contains no files")
    return {"source": str(root), "files": files, "tree_sha256": digest(files)}


def resolve_targets(model) -> list[str]:
    expression = re.compile(LANGUAGE_TARGETS)
    return sorted(
        name
        for name, module in model.named_modules()
        if expression.fullmatch(name) and isinstance(module, torch.nn.Linear)
    )


def move_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def generate_grid_candidates(
    model,
    processor,
    grid,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[list[list[int]]]]:
    payload = messages(grid, None, config["image_cell_size"])
    encoded = apply_template(processor, payload, generation=True)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    encoded = move_to_device(dict(encoded), model.device)
    model.eval()
    with torch.inference_mode():
        sequences = model.generate(
            **encoded,
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
            num_beams=config["num_beams"],
            num_return_sequences=config["num_return_sequences"],
            use_cache=True,
        )
    records = []
    candidates = []
    seen = set()
    for sequence in sequences:
        generated = sequence[prompt_tokens:]
        text = processor.decode(generated, skip_special_tokens=True)
        parsed = parse_grids(text, limit=2)
        records.append(
            {
                "text": text,
                "generated_tokens": int(generated.shape[0]),
                "parsed_grids": parsed,
            }
        )
        for candidate in parsed:
            key = digest(candidate)
            if key not in seen and len(candidates) < 2:
                seen.add(key)
                candidates.append(candidate)
    return records, candidates


def invocation_fingerprint(
    config_path: Path,
    challenge_path: Path,
    arm: str,
    task_id: str,
    global_adapter: Path | None,
    model_source: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "experiment": 29,
        "arm": arm,
        "task_id": task_id,
        "config_sha256": file_sha256(config_path),
        "challenge_sha256": file_sha256(challenge_path),
        "worker_sha256": file_sha256(Path(__file__)),
        "core_sha256": file_sha256(HERE / "core.py"),
        "model": model_source,
        "runtime": runtime,
    }
    if global_adapter is not None:
        result["global_adapter_sha256"] = {
            name: file_sha256(global_adapter / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
    return result


def verify_resume(run_dir: Path, fingerprint: dict[str, Any]) -> bool:
    candidate_path = run_dir / "candidate.json"
    manifest_path = run_dir / "manifest.json"
    if not candidate_path.exists() and not manifest_path.exists():
        return False
    if not candidate_path.exists() or not manifest_path.exists():
        raise RuntimeError("orphaned partial run cannot be resumed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("fingerprint") != fingerprint:
        raise RuntimeError("stale run fingerprint; refusing resume")
    if file_sha256(candidate_path) != manifest.get("candidate_sha256"):
        raise RuntimeError("candidate hash mismatch; refusing resume")
    adapter_dir = run_dir / "adapter"
    adapter_hashes = manifest.get("task_adapter_sha256", {})
    if {Path(name).name for name in adapter_hashes} != {
        "adapter_config.json",
        "adapter_model.safetensors",
    }:
        raise RuntimeError("task adapter hash set is incomplete")
    for name, expected in adapter_hashes.items():
        if file_sha256(adapter_dir / name) != expected:
            raise RuntimeError("task adapter hash mismatch; refusing resume")
    print(candidate_path.read_text())
    return True


def main() -> None:
    process_start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("base_task_lora", "perception_task_lora"),
        required=True,
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--global-adapter", type=Path)
    parser.add_argument("--model-path")
    args = parser.parse_args()

    from bitsandbytes.optim import PagedAdamW8bit
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    config = json.loads(args.config.read_text())
    if file_sha256(args.challenge) != config["challenge_sha256"]:
        raise ValueError("challenge does not match fixed SHA-256")
    if args.task_id not in config["task_ids"]:
        raise ValueError("task is outside the fixed hard set")
    if args.arm == "perception_task_lora" and args.global_adapter is None:
        raise ValueError("treatment arm requires --global-adapter")
    if args.arm == "base_task_lora" and args.global_adapter is not None:
        raise ValueError("baseline arm cannot receive a global adapter")
    if args.global_adapter is not None:
        expected = {
            "adapter_config.json": config["global_adapter_config_sha256"],
            "adapter_model.safetensors": config["global_adapter_model_sha256"],
        }
        for name, expected_hash in expected.items():
            if file_sha256(args.global_adapter / name) != expected_hash:
                raise ValueError(f"global adapter hash mismatch: {name}")

    challenge = json.loads(args.challenge.read_text())
    if not set(config["task_ids"]).issubset(challenge):
        raise ValueError("challenge is missing a task from the fixed hard set")
    task = challenge[args.task_id]
    if any("output" not in pair for pair in task["train"]):
        raise ValueError("training pair has no output")
    if any("output" in pair for pair in task["test"]):
        raise ValueError("worker challenge unexpectedly contains test labels")

    run_dir = args.output_root / args.arm / args.task_id
    global_path = args.global_adapter if args.arm == "perception_task_lora" else None
    source_identity = model_identity(args.model_path, config)
    active_runtime = runtime_identity()
    fingerprint = invocation_fingerprint(
        args.config,
        args.challenge,
        args.arm,
        args.task_id,
        global_path,
        source_identity,
        active_runtime,
    )
    if verify_resume(run_dir, fingerprint):
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("partial output exists; clean or quarantine it explicitly")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = config["seed"] + int(args.task_id, 16)
    random.seed(seed)
    set_seed(seed)
    dtype = getattr(torch, config["compute_dtype"])
    source = args.model_path or config["model"]
    revision = None if args.model_path else config["model_revision"]
    local_only = bool(args.model_path)
    processor = AutoProcessor.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_only,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=dtype,
        trust_remote_code=False,
    )
    targets = resolve_targets(model)
    if len(targets) != EXPECTED_MODULES:
        raise RuntimeError(
            f"resolved {len(targets)} task-LoRA modules; expected {EXPECTED_MODULES}"
        )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    task_config = LoraConfig(
        r=config["task_lora_rank"],
        lora_alpha=config["task_lora_alpha"],
        lora_dropout=config["task_lora_dropout"],
        bias="none",
        target_modules=LANGUAGE_TARGETS,
        task_type="CAUSAL_LM",
    )
    if global_path is not None:
        model = PeftModel.from_pretrained(
            model, global_path, adapter_name="global", is_trainable=False
        )
        global_before = tensor_digest(model, "global")
        set_seed(seed)
        model.add_adapter("task", task_config)
        model.base_model.set_adapter(["global", "task"])
    else:
        set_seed(seed)
        model = get_peft_model(model, task_config, adapter_name="task")
        model.base_model.set_adapter(["task"])
        global_before = None

    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name and ".task." in name
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != EXPECTED_TASK_TRAINABLE:
        raise RuntimeError(
            f"task adapter has {trainable} trainable parameters; "
            f"expected {EXPECTED_TASK_TRAINABLE}"
        )
    if any(
        parameter.requires_grad
        and not ("lora_" in name and ".task." in name)
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("a parameter outside the task adapter is trainable")
    initial_task_digest = tensor_digest(model, "task")

    optimizer = PagedAdamW8bit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    warmup = round(config["optimizer_steps"] * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup, config["optimizer_steps"]
    )
    losses = []
    truncations = 0
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(config["optimizer_steps"]):
        permutation, geometry = step_transform(config, step)
        optimizer.zero_grad(set_to_none=True)
        pair_losses = []
        for pair in task["train"]:
            input_grid = augment(pair["input"], permutation, geometry)
            output_grid = augment(pair["output"], permutation, geometry)
            encoded = supervised_encoding(
                processor,
                input_grid,
                output_grid,
                config["image_cell_size"],
                config["max_sequence_length"],
            )
            if encoded["input_ids"].shape[1] == config["max_sequence_length"]:
                truncations += 1
            encoded = move_to_device(dict(encoded), model.device)
            loss = model(**encoded).loss / len(task["train"])
            loss.backward()
            pair_losses.append(float(loss.detach()) * len(task["train"]))
            del encoded, loss
        if any(
            parameter.grad is not None and ".global." in name
            for name, parameter in model.named_parameters()
        ):
            raise RuntimeError("global adapter received a gradient")
        optimizer.step()
        scheduler.step()
        losses.append(sum(pair_losses) / len(pair_losses))

    train_seconds = time.monotonic() - start
    global_after = tensor_digest(model, "global") if global_path else None
    if global_before != global_after:
        raise RuntimeError("global adapter changed during task training")

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir, selected_adapters=["task"])
    saved_adapter_dir = (
        adapter_dir / "task"
        if (adapter_dir / "task" / "adapter_config.json").exists()
        else adapter_dir
    )
    task_adapter_hashes = {
        str((saved_adapter_dir / name).relative_to(adapter_dir)): file_sha256(
            saved_adapter_dir / name
        )
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }

    model.config.use_cache = True
    train_reconstructions = []
    for index, pair in enumerate(task["train"]):
        raw, candidates = generate_grid_candidates(
            model, processor, pair["input"], config
        )
        train_reconstructions.append(
            {
                "index": index,
                "raw_generations": raw,
                "candidates": candidates,
                "top2_exact": any(value == pair["output"] for value in candidates),
            }
        )
    test_predictions = []
    for index, pair in enumerate(task["test"]):
        raw, candidates = generate_grid_candidates(
            model, processor, pair["input"], config
        )
        test_predictions.append(
            {
                "index": index,
                "raw_generations": raw,
                "candidates": candidates,
            }
        )
    candidate = {
        "experiment": 29,
        "arm": args.arm,
        "task_id": args.task_id,
        "fingerprint": fingerprint,
        "initial_task_adapter_digest": initial_task_digest,
        "global_adapter_digest_before": global_before,
        "global_adapter_digest_after": global_after,
        "trainable_parameters": trainable,
        "optimizer_steps": config["optimizer_steps"],
        "train_pair_count": len(task["train"]),
        "losses": losses,
        "truncated_sequences": truncations,
        "train_seconds": train_seconds,
        "total_seconds": time.monotonic() - process_start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "train_reconstructions": train_reconstructions,
        "test_predictions": test_predictions,
    }
    candidate_path = run_dir / "candidate.json"
    atomic_json(candidate_path, candidate)
    manifest = {
        "fingerprint": fingerprint,
        "candidate_sha256": file_sha256(candidate_path),
        "task_adapter_sha256": task_adapter_hashes,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    print(json.dumps(candidate, ensure_ascii=False, indent=2))
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
