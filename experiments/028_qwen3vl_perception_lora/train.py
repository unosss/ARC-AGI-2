"""QLoRA training for the two fixed Experiment 028 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_solver.perception import fixed_probe_prompt, render_grid_image


LANGUAGE_TARGETS = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
MULTIMODAL_TARGETS = (
    r"^(?:"
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))|"
    r"model\.visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))|"
    r"model\.visual\.(?:merger|deepstack_merger_list\.\d+)\."
    r"(?:linear_fc1|linear_fc2)"
    r")$"
)
TARGETS = {
    "language_lora": (LANGUAGE_TARGETS, 252),
    "multimodal_lora": (MULTIMODAL_TARGETS, 356),
}
EXPECTED_TRAINABLE = {
    "language_lora": 33_030_144,
    "multimodal_lora": 40_271_872,
}
SYSTEM_PROMPT = (
    "You measure properties of one ARC grid. Be literal and precise. "
    "Return only the requested JSON object."
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_messages(row: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
    modality = row["modality"]
    content: list[dict[str, Any]] = []
    if modality in {"image", "both"}:
        content.append(
            {
                "type": "image",
                "image": render_grid_image(row["grid"], cell_size=24),
            }
        )
    content.append(
        {
            "type": "text",
            "text": fixed_probe_prompt(row["grid"], modality),
        }
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            row["target"], ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                ],
            }
        )
    return messages


def apply_template(processor, messages, add_generation_prompt: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


class PerceptionDataset(Dataset):
    def __init__(self, rows, processor, max_length: int):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt = apply_template(
            self.processor,
            build_messages(row, include_answer=False),
            add_generation_prompt=True,
        )
        full = apply_template(
            self.processor,
            build_messages(row, include_answer=True),
            add_generation_prompt=False,
        )
        prompt_ids = prompt["input_ids"][0]
        full_ids = full["input_ids"][0]
        prompt_length = int(prompt_ids.shape[0])
        full_length = int(full_ids.shape[0])
        if not torch.equal(full_ids[:prompt_length], prompt_ids):
            raise RuntimeError(f"{row['id']} prompt is not a prefix of full sequence")
        if prompt_length >= self.max_length:
            raise RuntimeError(
                f"{row['id']} prompt has {prompt_length} tokens and leaves no "
                f"answer tokens within limit {self.max_length}"
            )
        if full_length > self.max_length:
            for key, value in tuple(full.items()):
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 2
                    and value.shape[0] == 1
                    and value.shape[1] == full_length
                ):
                    full[key] = value[:, : self.max_length]
        labels = full["input_ids"].clone()
        labels[:, :prompt_length] = -100
        full["labels"] = labels
        return dict(full)


def single_item_collator(features: list[dict[str, torch.Tensor]]):
    if len(features) != 1:
        raise RuntimeError("Experiment 028 requires physical micro-batch size 1")
    return features[0]


def resolve_targets(model, pattern: str) -> list[str]:
    expression = re.compile(pattern)
    return sorted(
        name
        for name, module in model.named_modules()
        if expression.fullmatch(name) and isinstance(module, torch.nn.Linear)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(TARGETS), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--pilot-rows", type=int)
    args = parser.parse_args()

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    config = json.loads(args.config.read_text())
    corpus_manifest = json.loads(args.corpus_manifest.read_text())
    config_hash = file_sha256(args.config)
    corpus_hash = file_sha256(args.corpus)
    if corpus_manifest["config_sha256"] != config_hash:
        raise ValueError("corpus manifest config hash mismatch")
    if corpus_hash != config["train_corpus_sha256"]:
        raise ValueError("training corpus does not match fixed config SHA-256")
    if corpus_hash != corpus_manifest["train_file_sha256"]:
        raise ValueError("training corpus hash does not match its manifest")
    rows = load_jsonl(args.corpus)
    if len(rows) != config["train_grid_count"]:
        raise ValueError("training row count does not match fixed config")
    if args.pilot_rows is not None:
        if args.pilot_rows != config["pilot_rows"]:
            raise ValueError("pilot row count must match fixed config")
        rows = rows[: args.pilot_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = args.output_dir / "completed.json"
    manifest_path = args.output_dir / "run_manifest.json"
    checkpoint_dir = args.output_dir / "checkpoints"
    adapter_dir = args.output_dir / "adapter"
    if not manifest_path.exists() and (
        completion_path.exists() or checkpoint_dir.exists() or adapter_dir.exists()
    ):
        raise ValueError("orphan training state exists without a run manifest")

    active_seed = config["pilot_seed"] if args.pilot_rows is not None else config["seed"]
    random.seed(active_seed)
    set_seed(active_seed)
    model_source = args.model_path or config["model"]
    dtype = getattr(torch, config["compute_dtype"])
    processor = AutoProcessor.from_pretrained(
        model_source,
        revision=None if args.model_path else config["model_revision"],
        local_files_only=bool(args.model_path),
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
        revision=None if args.model_path else config["model_revision"],
        local_files_only=bool(args.model_path),
        quantization_config=quantization,
        device_map={"": 0},
        dtype=dtype,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    pattern, expected_module_count = TARGETS[args.arm]
    resolved_modules = resolve_targets(model, pattern)
    if len(resolved_modules) != expected_module_count:
        raise RuntimeError(
            f"{args.arm} resolved {len(resolved_modules)} modules, "
            f"expected {expected_module_count}"
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
            target_modules=pattern,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != EXPECTED_TRAINABLE[args.arm]:
        raise RuntimeError(
            f"{args.arm} has {trainable} trainable parameters, "
            f"expected {EXPECTED_TRAINABLE[args.arm]}"
        )
    if any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("a non-LoRA parameter is unexpectedly trainable")

    run_manifest = {
        "experiment": 28,
        "arm": args.arm,
        "pilot": args.pilot_rows is not None,
        "row_count": len(rows),
        "seed": active_seed,
        "config_sha256": config_hash,
        "corpus_sha256": corpus_hash,
        "model_revision": config["model_revision"],
        "target_pattern": pattern,
        "resolved_module_count": len(resolved_modules),
        "resolved_modules": resolved_modules,
        "trainable_parameters": trainable,
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != run_manifest:
        raise ValueError("existing run manifest does not match this invocation")
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2))
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        for key, value in run_manifest.items():
            if completion.get(key) != value:
                raise ValueError(f"completed run mismatch: {key}")
        for name, expected_hash in completion.get("adapter_sha256", {}).items():
            path = adapter_dir / name
            if not path.exists() or file_sha256(path) != expected_hash:
                raise ValueError(f"completed adapter mismatch: {name}")
        if set(completion.get("adapter_sha256", {})) != {
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            raise ValueError("completed adapter hash set is incomplete")
        print(completion_path.read_text())
        return

    dataset = PerceptionDataset(rows, processor, config["max_sequence_length"])
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
        seed=active_seed,
        data_seed=active_seed,
        full_determinism=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=single_item_collator,
    )
    checkpoint_dir = Path(training_args.output_dir)
    resume = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=resume)
    final_dir = adapter_dir
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir / "processor"))
    adapter_sha256 = {
        name: file_sha256(final_dir / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    completion = {
        **run_manifest,
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "adapter_files": sorted(path.name for path in final_dir.iterdir()),
        "adapter_sha256": adapter_sha256,
    }
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2))
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
