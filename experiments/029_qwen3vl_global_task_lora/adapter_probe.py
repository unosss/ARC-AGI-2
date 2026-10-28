"""GPU integration probe for additive PEFT adapters used by Experiment 029."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from core import apply_template, file_sha256
from worker import (
    EXPECTED_MODULES,
    EXPECTED_TASK_TRAINABLE,
    LANGUAGE_TARGETS,
    resolve_targets,
    tensor_digest,
)


def activate(model, adapters: list[str], train_task: bool = False) -> None:
    """PEFT set_adapter re-enables adapters, so always restore the freeze mask."""
    model.base_model.set_adapter(adapters)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            train_task and "lora_" in name and ".task." in name
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--global-adapter", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        set_seed,
    )

    config = json.loads(args.config.read_text())
    for name, expected in {
        "adapter_config.json": config["global_adapter_config_sha256"],
        "adapter_model.safetensors": config["global_adapter_model_sha256"],
    }.items():
        if file_sha256(args.global_adapter / name) != expected:
            raise ValueError(f"global adapter hash mismatch: {name}")
    source = args.model_path or config["model"]
    revision = None if args.model_path else config["model_revision"]
    dtype = getattr(torch, config["compute_dtype"])
    processor = AutoProcessor.from_pretrained(
        source, revision=revision, local_files_only=bool(args.model_path)
    )
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        revision=revision,
        local_files_only=bool(args.model_path),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map={"": 0},
        dtype=dtype,
    )
    if len(resolve_targets(model)) != EXPECTED_MODULES:
        raise RuntimeError("base model target-module resolution changed")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        model, args.global_adapter, adapter_name="global", is_trainable=False
    )
    global_digest = tensor_digest(model, "global")
    set_seed(config["seed"])
    model.add_adapter(
        "task",
        LoraConfig(
            r=config["task_lora_rank"],
            lora_alpha=config["task_lora_alpha"],
            lora_dropout=0,
            bias="none",
            target_modules=LANGUAGE_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name and ".task." in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != EXPECTED_TASK_TRAINABLE:
        raise RuntimeError(f"unexpected task trainable count: {trainable}")

    payload = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Return the JSON grid [[0]]."}],
        }
    ]
    inputs = apply_template(processor, payload, generation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    model.eval()
    activate(model, ["global"])
    with torch.inference_mode():
        global_logits = model(**inputs).logits[:, -1, :].float().cpu()
    activate(model, ["global", "task"])
    with torch.inference_mode():
        zero_task_logits = model(**inputs).logits[:, -1, :].float().cpu()
    if not torch.equal(global_logits, zero_task_logits):
        raise RuntimeError("fresh zero-output task adapter changed the forward pass")

    changed = False
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name and ".task." in name:
                parameter.fill_(0.01)
                changed = True
                break
    if not changed:
        raise RuntimeError("could not perturb a task adapter tensor")
    activate(model, ["global", "task"])
    with torch.inference_mode():
        combined_logits = model(**inputs).logits[:, -1, :].float().cpu()
    if torch.equal(global_logits, combined_logits):
        raise RuntimeError("task adapter has no simultaneous forward effect")
    activate(model, ["global"])
    with torch.inference_mode():
        restored_global_logits = model(**inputs).logits[:, -1, :].float().cpu()
    if not torch.equal(global_logits, restored_global_logits):
        raise RuntimeError("task adapter contaminated global-only forward")

    activate(model, ["global", "task"], train_task=True)
    model.train()
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()
    if any(
        parameter.grad is not None and ".global." in name
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("global adapter received gradients")
    if not any(
        parameter.grad is not None and ".task." in name
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("task adapter received no gradients")

    model.eval()
    with tempfile.TemporaryDirectory() as temporary:
        saved = Path(temporary)
        model.save_pretrained(saved, selected_adapters=["task"])
        task_dir = saved / "task" if (saved / "task").exists() else saved
        model.delete_adapter("task")
        model.load_adapter(task_dir, adapter_name="task_reloaded", is_trainable=False)
        activate(model, ["global", "task_reloaded"])
        with torch.inference_mode():
            reloaded_logits = model(**inputs).logits[:, -1, :].float().cpu()
    if not torch.equal(combined_logits, reloaded_logits):
        maximum = float((combined_logits - reloaded_logits).abs().max())
        raise RuntimeError(f"task adapter save/reload parity failed: max_abs={maximum}")
    if tensor_digest(model, "global") != global_digest:
        raise RuntimeError("global tensor digest changed during integration probe")

    result = {
        "passed": True,
        "resolved_modules": EXPECTED_MODULES,
        "task_trainable_parameters": trainable,
        "fresh_task_forward_parity": True,
        "simultaneous_task_effect": True,
        "global_gradient_free": True,
        "task_gradient_present": True,
        "save_reload_exact": True,
        "global_unchanged": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
