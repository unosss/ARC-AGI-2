#!/usr/bin/env python
"""Run objective single-grid perception probes with a multimodal Qwen model."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from arc_solver.perception import (
    build_probe_prompt,
    describe_grid,
    parse_json_object,
    render_grid_image,
    score_description,
)


DEFAULT_TASK_IDS = (
    "03560426",
    "045e512c",
    "05269061",
    "05a7bcf2",
    "05f2a901",
    "0607ce86",
    "06df4c85",
    "09629e4f",
    "0962bcdd",
    "0a1d4ef5",
)


def load_grids(data_dir: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    """Select the first training input and output from each task."""
    grids = []
    for task_id in task_ids:
        path = data_dir / f"{task_id}.json"
        task = json.loads(path.read_text())
        pair = task["train"][0]
        for role in ("input", "output"):
            grids.append({"task_id": task_id, "role": role, "grid": pair[role]})
    return grids


def build_messages(grid, modality: str):
    prompt = build_probe_prompt(grid, modality)
    content = []
    if modality in {"image", "both"}:
        content.append({"type": "image", "image": render_grid_image(grid)})
    content.append({"type": "text", "text": prompt})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You measure properties of one ARC grid. Be literal and "
                        "precise. Return only the requested JSON object."
                    ),
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["modality"], record["role"])].append(record["score"])
    summaries = {}
    for (modality, role), scores in sorted(groups.items()):
        key = f"{modality}/{role}"
        summaries[key] = {
            "count": len(scores),
            "structured_rate": sum(score["structured"] for score in scores)
            / len(scores),
            "reading": sum(score["reading"] for score in scores) / len(scores),
            "objects": sum(score["objects"] for score in scores) / len(scores),
            "relations": sum(score["relations"] for score in scores) / len(scores),
            "overall": sum(score["overall"] for score in scores) / len(scores),
            "exact_rate": sum(score["exact"] for score in scores) / len(scores),
        }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/ARC-AGI-2/data/training")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", default=list(DEFAULT_TASK_IDS))
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("text", "image", "both"),
        default=["text", "image", "both"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started_all = time.perf_counter()
    grids = load_grids(args.data_dir, args.task_ids)
    result_path = args.output_dir / "perception_results.jsonl"
    with result_path.open("w") as output:
        for index, item in enumerate(grids, 1):
            expected = describe_grid(item["grid"])
            for modality in args.modalities:
                messages = build_messages(item["grid"], modality)
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to(model.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                elapsed = time.perf_counter() - started
                tokens = generated[0, inputs["input_ids"].shape[1] :]
                raw = processor.decode(tokens, skip_special_tokens=True)
                prediction = parse_json_object(raw)
                record = {
                    "task_id": item["task_id"],
                    "role": item["role"],
                    "modality": modality,
                    "expected": expected,
                    "prediction": prediction,
                    "score": score_description(prediction, expected),
                    "output_tokens": int(tokens.shape[0]),
                    "elapsed_seconds": round(elapsed, 3),
                    "raw": raw,
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"[{index}/{len(grids)}] {item['task_id']} {item['role']} "
                    f"{modality}: {record['score']['overall']:.3f}",
                    flush=True,
                )
    summary = {
        "model": args.model,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "task_ids": args.task_ids,
        "grid_count": len(grids),
        "response_count": len(records),
        "modalities": args.modalities,
        "seconds": round(time.perf_counter() - started_all, 3),
        "metrics": aggregate(records),
    }
    (args.output_dir / "perception_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
