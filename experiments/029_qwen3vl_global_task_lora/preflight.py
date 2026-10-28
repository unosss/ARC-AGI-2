"""Exhaustively verify sequence budgets for the fixed augmented corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import (
    apply_template,
    augment,
    file_sha256,
    messages,
    step_transform,
    supervised_encoding,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()

    from transformers import AutoProcessor

    config = json.loads(args.config.read_text())
    if file_sha256(args.challenge) != config["challenge_sha256"]:
        raise ValueError("challenge does not match fixed SHA-256")
    challenge = json.loads(args.challenge.read_text())
    source = args.model_path or config["model"]
    processor = AutoProcessor.from_pretrained(
        source,
        revision=None if args.model_path else config["model_revision"],
        local_files_only=bool(args.model_path),
        trust_remote_code=False,
    )
    max_prompt = 0
    max_supervised = 0
    truncated = 0
    examples = 0
    for task_id in config["task_ids"]:
        task = challenge[task_id]
        for step in range(config["optimizer_steps"]):
            permutation, geometry = step_transform(config, step)
            for pair in task["train"]:
                input_grid = augment(pair["input"], permutation, geometry)
                output_grid = augment(pair["output"], permutation, geometry)
                prompt = apply_template(
                    processor,
                    messages(input_grid, None, config["image_cell_size"]),
                    generation=True,
                )
                full = apply_template(
                    processor,
                    messages(
                        input_grid, output_grid, config["image_cell_size"]
                    ),
                    generation=False,
                )
                prompt_length = int(prompt["input_ids"].shape[1])
                full_length = int(full["input_ids"].shape[1])
                max_prompt = max(max_prompt, prompt_length)
                max_supervised = max(max_supervised, full_length)
                truncated += int(full_length > config["max_sequence_length"])
                if full_length <= config["max_sequence_length"]:
                    supervised_encoding(
                        processor,
                        input_grid,
                        output_grid,
                        config["image_cell_size"],
                        config["max_sequence_length"],
                    )
                examples += 1
        for pair in task["test"]:
            for geometry in range(config["geometric_transforms"]):
                input_grid = augment(pair["input"], list(range(10)), geometry)
                prompt = apply_template(
                    processor,
                    messages(input_grid, None, config["image_cell_size"]),
                    generation=True,
                )
                prompt_length = int(prompt["input_ids"].shape[1])
                if prompt_length >= config["max_sequence_length"]:
                    raise RuntimeError(
                        f"{task_id} test prompt exceeds sequence budget"
                    )
                max_prompt = max(max_prompt, prompt_length)
    result = {
        "passed": truncated == 0,
        "augmented_train_examples_checked": examples,
        "max_prompt_tokens": max_prompt,
        "max_supervised_tokens_before_truncation": max_supervised,
        "supervised_truncation_count": truncated,
        "max_sequence_length": config["max_sequence_length"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if truncated:
        raise RuntimeError(
            f"{truncated} supervised examples exceed the fixed sequence budget"
        )


if __name__ == "__main__":
    main()
