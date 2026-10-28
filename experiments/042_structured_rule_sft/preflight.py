"""Token-length and paired-prompt preflight before any Experiment 042 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_io import EpisodeDataset, apply_template, build_messages


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()

    from transformers import AutoProcessor

    config = json.loads(args.config.read_text())
    if args.budget not in [int(value) for value in config["train_budgets"]]:
        raise ValueError("budget is not preregistered")
    rows = load_jsonl(args.train_file)[: args.budget]
    if len(rows) != args.budget:
        raise ValueError("insufficient training rows")
    source = args.model_path or config["model"]
    processor = AutoProcessor.from_pretrained(
        source,
        revision=None if args.model_path else config["model_revision"],
        local_files_only=bool(args.model_path),
        trust_remote_code=False,
    )

    prompt_hashes = {}
    episode_hash = hashlib.sha256(
        "\n".join(
            json.dumps(
                {
                    "id": row["id"],
                    "demos": row["demos"],
                    "query_input": row["query_input"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
        ).encode()
    ).hexdigest()
    arm_stats = {}
    for arm in ("direct", "program"):
        dataset = EpisodeDataset(
            rows,
            processor,
            arm,
            config["modality"],
            int(config["image_cell_size"]),
            int(config["max_sequence_length"]),
        )
        lengths = []
        prompts = []
        for index, row in enumerate(rows):
            item = dataset[index]
            lengths.append(int(item["input_ids"].shape[1]))
            prompt = apply_template(
                processor,
                build_messages(
                    row,
                    arm,
                    config["modality"],
                    int(config["image_cell_size"]),
                    include_answer=False,
                ),
                add_generation_prompt=True,
            )
            prompts.append(prompt["input_ids"].numpy().tobytes())
        prompt_hashes[arm] = hashlib.sha256(b"".join(prompts)).hexdigest()
        arm_stats[arm] = {
            "minimum_tokens": min(lengths),
            "maximum_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "over_limit": sum(
                length > int(config["max_sequence_length"]) for length in lengths
            ),
        }
    report = {
        "schema_version": 1,
        "experiment": 42,
        "budget": args.budget,
        "config_sha256": file_sha256(args.config),
        "train_file_sha256": file_sha256(args.train_file),
        "paired_episode_sha256": episode_hash,
        "prompt_sha256": prompt_hashes,
        "arms": arm_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
