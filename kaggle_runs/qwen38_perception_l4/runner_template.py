"""Kaggle-specific runner appended to arc_solver/perception.py."""

import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


TASK_IDS = (
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
MODALITIES = ("text", "image", "both")
INPUT_DIR = Path("/kaggle/input")
RESULTS_PATH = Path("/kaggle/working/perception_results.jsonl")
SUMMARY_PATH = Path("/kaggle/working/perception_summary.json")
MAX_NEW_TOKENS = 3072


def find_file(name):
    matches = list(INPUT_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_DIR}")
    return matches[0]


def find_model_dir():
    candidates = [
        path.parent
        for path in INPUT_DIR.rglob("config.json")
        if (path.parent / "processor_config.json").exists()
        or (path.parent / "preprocessor_config.json").exists()
    ]
    qwen = [path for path in candidates if "qwen" in str(path).lower()]
    if not qwen:
        raise FileNotFoundError("multimodal Qwen model directory not found")
    return qwen[0]


def build_messages(grid, modality):
    content = []
    if modality in {"image", "both"}:
        content.append({"type": "image", "image": render_grid_image(grid)})
    content.append({"type": "text", "text": build_probe_prompt(grid, modality)})
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


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["modality"], record["role"])].append(record["score"])
    result = {}
    for (modality, role), scores in sorted(groups.items()):
        result[f"{modality}/{role}"] = {
            "count": len(scores),
            "structured_rate": sum(score["structured"] for score in scores)
            / len(scores),
            "reading": sum(score["reading"] for score in scores) / len(scores),
            "objects": sum(score["objects"] for score in scores) / len(scores),
            "relations": sum(score["relations"] for score in scores) / len(scores),
            "overall": sum(score["overall"] for score in scores) / len(scores),
            "exact_rate": sum(score["exact"] for score in scores) / len(scores),
        }
    return result


def main():
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        wheels = list(INPUT_DIR.rglob("bitsandbytes-*.whl"))
        if not wheels:
            raise FileNotFoundError("offline bitsandbytes wheel not found")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
                "--find-links",
                str(wheels[0].parent),
                "bitsandbytes>=0.49.2",
            ]
        )

    import torch
    import transformers
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    challenges = json.loads(find_file("arc-agi_training_challenges.json").read_text())
    model_dir = find_model_dir()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=False)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()

    grids = []
    for task_id in TASK_IDS:
        pair = challenges[task_id]["train"][0]
        for role in ("input", "output"):
            grids.append({"task_id": task_id, "role": role, "grid": pair[role]})

    records = []
    started_all = time.perf_counter()
    with RESULTS_PATH.open("w") as output:
        for index, item in enumerate(grids, 1):
            expected = describe_grid(item["grid"])
            for modality in MODALITIES:
                inputs = processor.apply_chat_template(
                    build_messages(item["grid"], modality),
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
                        max_new_tokens=MAX_NEW_TOKENS,
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
        "model": str(model_dir),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "task_ids": TASK_IDS,
        "grid_count": len(grids),
        "response_count": len(records),
        "modalities": MODALITIES,
        "seconds": round(time.perf_counter() - started_all, 3),
        "metrics": aggregate(records),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
