"""Run the fixed Qwen3-VL isolated-grid perception comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import arc_solver.perception as perception_module
from arc_solver.perception import (
    describe_grid,
    parse_json_object,
    render_grid_image,
    score_description,
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model",
        "model_revision",
        "task_ids",
        "roles",
        "modalities",
        "train_pair_index",
        "max_new_tokens",
        "do_sample",
        "enable_thinking",
        "load_in_4bit",
        "compute_dtype",
        "image_cell_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if config["modalities"] != ["text", "image", "both"]:
        raise ValueError("result-bearing modalities must be text, image, both")
    if config["roles"] != ["input", "output"]:
        raise ValueError("result-bearing roles must be input, output")
    if len(config["task_ids"]) != 10 or len(set(config["task_ids"])) != 10:
        raise ValueError("exactly ten unique task IDs are required")
    if config["do_sample"]:
        raise ValueError("result-bearing decoding must be greedy")


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return record["task_id"], record["role"], record["modality"]


def load_completed(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record_key(record)
        if key in completed:
            raise ValueError(f"duplicate result key at line {line_number}: {key}")
        completed[key] = record
    return completed


def fixed_probe_description(grid: list[list[int]]) -> dict[str, Any]:
    """Describe a grid without selecting questions from its visual content."""
    expected = describe_grid(grid)
    height, width = len(grid), len(grid[0])
    fixed_cells = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    expected["sampled_cells"] = {
        f"{row},{column}": (
            grid[row][column] if row < height and column < width else None
        )
        for row, column in fixed_cells
    }
    expected["sampled_rows"] = {"0": grid[0], "1": grid[1]}
    return expected


def fixed_probe_prompt(grid: list[list[int]], modality: str) -> str:
    """Use fixed questions so the image arm receives no content-derived hints."""
    if modality not in {"text", "image", "both"}:
        raise ValueError("modality must be text, image, or both")
    source = ""
    if modality in {"text", "both"}:
        source = "\nGrid as JSON:\n" + json.dumps(grid, separators=(",", ":"))
    image_note = (
        "\nA color-grid image is attached. Rows run top-to-bottom and columns "
        "left-to-right, both zero-indexed."
        if modality in {"image", "both"}
        else ""
    )
    return f"""Analyze this single ARC grid independently. Do not infer any input-output rule.
Use same-color nonzero connected components. For 4-connectivity use only orthogonal
neighbors; for 8-connectivity include diagonals. Bounding boxes are inclusive
[min_row,min_col,max_row,max_col]. Component shapes are cell coordinates normalized
to the bounding-box top-left. Sort components by decreasing size, then color,
min_row, min_col; return at most six. same_shape_pairs_4 contains index pairs into
largest_components_4 whose normalized shapes are exactly equal, regardless of color.
Horizontal symmetry means top-bottom reflection; vertical means left-right reflection.
Report sampled cells ["0,0","0,1","1,0","1,1","2,2"] and complete rows ["0","1"].
Return only one JSON object with exactly these keys:
dimensions, color_counts, sampled_cells, sampled_rows, nonzero_bbox, zero_rows,
zero_cols, component_count_4, component_count_8, largest_components_4,
same_shape_pairs_4, symmetry.{image_note}{source}"""


def build_messages(grid: list[list[int]], modality: str, cell_size: int):
    content: list[dict[str, Any]] = []
    if modality in {"image", "both"}:
        content.append(
            {
                "type": "image",
                "image": render_grid_image(grid, cell_size=cell_size),
            }
        )
    content.append(
        {"type": "text", "text": fixed_probe_prompt(grid, modality=modality)}
    )
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


def prepare_inputs(processor, messages, enable_thinking: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        if enable_thinking:
            raise
        return processor.apply_chat_template(messages, **kwargs)


def load_model(config: dict[str, Any]):
    import torch
    import transformers
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    dtype = getattr(torch, config["compute_dtype"])
    model_kwargs: dict[str, Any] = {
        "revision": config["model_revision"],
        "device_map": "auto",
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if config["load_in_4bit"]:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=dtype,
        )
    processor = AutoProcessor.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        trust_remote_code=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config["model"], **model_kwargs
    ).eval()
    return model, processor, transformers.__version__


def run(args: argparse.Namespace) -> None:
    import torch

    config = load_json(args.config)
    validate_config(config)
    challenges = load_json(args.challenges)
    missing_tasks = [task_id for task_id in config["task_ids"] if task_id not in challenges]
    if missing_tasks:
        raise KeyError(f"tasks missing from challenges: {missing_tasks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    completed = load_completed(results_path)
    expected_keys = {
        (task_id, role, modality)
        for task_id in config["task_ids"]
        for role in config["roles"]
        for modality in config["modalities"]
    }
    unknown_keys = sorted(set(completed) - expected_keys)
    if unknown_keys:
        raise ValueError(f"resume file contains unknown keys: {unknown_keys}")

    manifest = {
        "experiment": 27,
        "config": config,
        "config_sha256": canonical_hash(config),
        "challenges_sha256": hashlib.sha256(args.challenges.read_bytes()).hexdigest(),
        "probe_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(perception_module.__file__).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "expected_response_count": len(expected_keys),
    }
    if results_path.exists() and not manifest_path.exists():
        raise ValueError("results exist without a matching resume manifest")
    if manifest_path.exists():
        previous = load_json(manifest_path)
        for key in ("config_sha256", "challenges_sha256", "probe_code_sha256"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"resume manifest mismatch: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    model, processor, transformers_version = load_model(config)
    manifest["transformers"] = transformers_version
    manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    total_started = time.perf_counter()
    with results_path.open("a") as output:
        for task_id in config["task_ids"]:
            pair = challenges[task_id]["train"][config["train_pair_index"]]
            for role in config["roles"]:
                grid = pair[role]
                expected = fixed_probe_description(grid)
                for modality in config["modalities"]:
                    key = (task_id, role, modality)
                    if key in completed:
                        continue
                    messages = build_messages(
                        grid, modality, cell_size=config["image_cell_size"]
                    )
                    inputs = prepare_inputs(
                        processor, messages, config["enable_thinking"]
                    ).to(model.device)
                    prompt_tokens = int(inputs["input_ids"].shape[1])
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=config["max_new_tokens"],
                            do_sample=config["do_sample"],
                        )
                    elapsed = time.perf_counter() - started
                    generated_tokens = generated[0, prompt_tokens:]
                    raw = processor.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    prediction = parse_json_object(raw)
                    record = {
                        "task_id": task_id,
                        "role": role,
                        "modality": modality,
                        "grid_sha256": canonical_hash(grid),
                        "expected": expected,
                        "prediction": prediction,
                        "score": score_description(prediction, expected),
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": int(generated_tokens.shape[0]),
                        "elapsed_seconds": round(elapsed, 6),
                        "raw": raw,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    completed[key] = record
                    print(
                        f"{len(completed):02d}/{len(expected_keys)} "
                        f"{task_id}/{role}/{modality} "
                        f"overall={record['score']['overall']:.4f} "
                        f"seconds={elapsed:.2f}",
                        flush=True,
                    )

    if set(completed) != expected_keys:
        missing = sorted(expected_keys - set(completed))
        raise RuntimeError(f"incomplete run; missing {missing}")
    manifest["completed"] = True
    manifest["elapsed_seconds_this_invocation"] = round(
        time.perf_counter() - total_started, 6
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
