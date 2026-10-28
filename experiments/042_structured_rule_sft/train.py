"""Matched QLoRA training for the direct and executable-program arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import torch

from config_fingerprint import corpus_config_sha256
from model_io import EpisodeDataset, single_item_collator


LANGUAGE_TARGETS = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
EXPECTED_MODULE_COUNT = 252
EXPECTED_TRAINABLE = 33_030_144


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolved_model_identity(
    model_path: str | None, configured_model: str, revision: str
) -> dict[str, Any]:
    if model_path is None:
        return {"source": configured_model, "revision": revision, "local": False}
    path = Path(model_path).resolve()
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"local model has no config.json: {path}")
    weight_paths = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    if not weight_paths:
        raise FileNotFoundError(f"local model has no weight files: {path}")
    return {
        "source": str(path),
        "revision": None,
        "local": True,
        "config_sha256": file_sha256(config_path),
        "weight_sha256": {
            weight.name: file_sha256(weight) for weight in weight_paths
        },
    }


def resolve_targets(model) -> list[str]:
    expression = re.compile(LANGUAGE_TARGETS)
    return sorted(
        name
        for name, module in model.named_modules()
        if expression.fullmatch(name) and isinstance(module, torch.nn.Linear)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--arm", choices=("direct", "program"), required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = json.loads(args.config.read_text())
    allowed_budgets = [int(value) for value in config["train_budgets"]]
    if args.budget not in allowed_budgets:
        raise ValueError(f"budget must be one of {allowed_budgets}")
    corpus_manifest_path = args.train_file.parent / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text())
    if corpus_manifest.get("corpus_config_sha256") != corpus_config_sha256(config):
        raise ValueError("corpus was built with different data-selection settings")
    expected_train_hash = corpus_manifest["files"][args.train_file.name]["sha256"]
    if file_sha256(args.train_file) != expected_train_hash:
        raise ValueError("train corpus hash does not match its manifest")
    rows = load_jsonl(args.train_file)
    if len(rows) < args.budget:
        raise ValueError("train corpus is smaller than requested budget")
    rows = rows[: args.budget]

    random.seed(int(config["train_seed"]))
    set_seed(int(config["train_seed"]))
    model_source = args.model_path or config["model"]
    local_only = bool(args.model_path)
    revision = None if local_only else config["model_revision"]
    dtype = getattr(torch, config["compute_dtype"])
    processor = AutoProcessor.from_pretrained(
        model_source,
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
        model_source,
        revision=revision,
        local_files_only=local_only,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=dtype,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    resolved_modules = resolve_targets(model)
    if len(resolved_modules) != EXPECTED_MODULE_COUNT:
        raise RuntimeError(
            f"resolved {len(resolved_modules)} language modules; "
            f"expected {EXPECTED_MODULE_COUNT}"
        )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            target_modules=LANGUAGE_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != EXPECTED_TRAINABLE:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable}")
    if any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("a non-LoRA parameter is unexpectedly trainable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": 42,
        "arm": args.arm,
        "budget": args.budget,
        "config_sha256": file_sha256(args.config),
        "train_file_sha256": file_sha256(args.train_file),
        "resolved_model": resolved_model_identity(
            args.model_path, config["model"], config["model_revision"]
        ),
        "seed": config["train_seed"],
        "resolved_modules": resolved_modules,
        "trainable_parameters": trainable,
        "row_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in rows).encode()
        ).hexdigest(),
    }
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing run manifest does not match this invocation")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    adapter_dir = args.output_dir / "adapter"
    completion_path = args.output_dir / "completed.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if any(completion.get(key) != value for key, value in manifest.items()):
            raise ValueError("completed run does not match manifest")
        for name, expected in completion["adapter_sha256"].items():
            if file_sha256(adapter_dir / name) != expected:
                raise ValueError(f"completed adapter hash mismatch: {name}")
        print(completion_path.read_text())
        return

    dataset = EpisodeDataset(
        rows=rows,
        processor=processor,
        arm=args.arm,
        modality=config["modality"],
        cell_size=int(config["image_cell_size"]),
        max_length=int(config["max_sequence_length"]),
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["micro_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler"],
        optim=config["optimizer"],
        bf16=True,
        fp16=False,
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=8,
        save_strategy="steps",
        save_steps=64,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=config["train_seed"],
        data_seed=config["train_seed"],
        full_determinism=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=single_item_collator,
    )
    result = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    adapter_hashes = {
        path.relative_to(adapter_dir).as_posix(): file_sha256(path)
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    }
    completion = {
        **manifest,
        "train_metrics": result.metrics,
        "adapter_sha256": adapter_hashes,
    }
    completion_path.write_text(
        json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(completion_path.read_text())


if __name__ == "__main__":
    main()
