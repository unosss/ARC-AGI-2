"""Evaluate a completed Experiment 028 adapter on sealed synthetic and ARC grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_solver.perception import (
    fixed_probe_description,
    fixed_probe_prompt,
    parse_json_object,
    render_grid_image,
    score_description,
)


SYSTEM_PROMPT = (
    "You measure properties of one ARC grid. Be literal and precise. "
    "Return only the requested JSON object."
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def messages(grid: list[list[int]], modality: str, cell_size: int):
    content: list[dict[str, Any]] = []
    if modality in {"image", "both"}:
        content.append(
            {
                "type": "image",
                "image": render_grid_image(grid, cell_size=cell_size),
            }
        )
    content.append({"type": "text", "text": fixed_probe_prompt(grid, modality)})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


def apply_template(processor, payload):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(
            payload, enable_thinking=False, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(payload, **kwargs)


def build_evaluation_rows(
    config: dict[str, Any],
    synthetic_path: Path,
    official_path: Path,
    training_path: Path,
    corpus_manifest_path: Path,
) -> list[dict[str, Any]]:
    corpus_manifest = json.loads(corpus_manifest_path.read_text())
    if file_sha256(training_path) != config["train_corpus_sha256"]:
        raise ValueError("training corpus does not match fixed config SHA-256")
    if file_sha256(synthetic_path) != config["synthetic_eval_sha256"]:
        raise ValueError("synthetic evaluation does not match fixed config SHA-256")
    if file_sha256(training_path) != corpus_manifest["train_file_sha256"]:
        raise ValueError("training corpus does not match corpus manifest")
    if file_sha256(synthetic_path) != corpus_manifest["synthetic_eval_file_sha256"]:
        raise ValueError("synthetic evaluation does not match corpus manifest")
    training = load_jsonl(training_path)
    synthetic = load_jsonl(synthetic_path)
    if len(training) != config["train_grid_count"]:
        raise ValueError("training row count mismatch")
    if len(synthetic) != config["synthetic_eval_grid_count"]:
        raise ValueError("synthetic evaluation row count mismatch")
    for expected_split, source in (("train", training), ("synthetic_eval", synthetic)):
        ids = set()
        hashes = set()
        for row in source:
            if row["split"] != expected_split:
                raise ValueError(f"unexpected {expected_split} split marker")
            if row["id"] in ids or row["grid_sha256"] in hashes:
                raise ValueError(f"duplicate {expected_split} row")
            if canonical_hash(row["grid"]) != row["grid_sha256"]:
                raise ValueError(f"{row['id']} grid hash mismatch")
            if fixed_probe_description(row["grid"]) != row["target"]:
                raise ValueError(f"{row['id']} target mismatch")
            ids.add(row["id"])
            hashes.add(row["grid_sha256"])
    training_hashes = {row["grid_sha256"] for row in training}
    synthetic_hashes = {row["grid_sha256"] for row in synthetic}
    if len(training_hashes) != config["train_grid_count"]:
        raise ValueError("training grid hash count mismatch")
    if len(synthetic_hashes) != config["synthetic_eval_grid_count"]:
        raise ValueError("synthetic evaluation grid hash count mismatch")
    if canonical_hash(sorted(training_hashes)) != corpus_manifest["train_grid_set_sha256"]:
        raise ValueError("training grid set does not match corpus manifest")
    if (
        canonical_hash(sorted(synthetic_hashes))
        != corpus_manifest["synthetic_eval_grid_set_sha256"]
    ):
        raise ValueError("synthetic evaluation grid set does not match corpus manifest")
    if training_hashes & synthetic_hashes:
        raise RuntimeError("training/synthetic evaluation grid collision")
    if file_sha256(official_path) != config["official_challenges_sha256"]:
        raise ValueError("official challenges SHA-256 does not match fixed config")

    challenges = json.loads(official_path.read_text())
    official = []
    official_hashes = set()
    for task_id in config["official_task_ids"]:
        pair = challenges[task_id]["train"][0]
        for role in ("input", "output"):
            grid = pair[role]
            grid_hash = canonical_hash(grid)
            official_hashes.add(grid_hash)
            for modality in config["modalities"]:
                official.append(
                    {
                        "id": f"{task_id}/{role}/{modality}",
                        "split": "official",
                        "task_id": task_id,
                        "role": role,
                        "modality": modality,
                        "grid": grid,
                        "grid_sha256": grid_hash,
                        "target": fixed_probe_description(grid),
                    }
                )
    train_collision = training_hashes & official_hashes
    eval_collision = synthetic_hashes & official_hashes
    if train_collision or eval_collision:
        raise RuntimeError(
            "synthetic/official grid collision: "
            f"train={sorted(train_collision)} eval={sorted(eval_collision)}"
        )
    return synthetic + official


def result_key(record: dict[str, Any]) -> tuple[str, str]:
    return record["split"], record["id"]


def verify_training_completions(
    config: dict[str, Any],
    runs_root: Path,
    expected_config_hash: str,
    expected_corpus_hash: str,
) -> dict[str, dict[str, Any]]:
    expected_modules = {"language_lora": 252, "multimodal_lora": 356}
    expected_trainable = {
        "language_lora": 33_030_144,
        "multimodal_lora": 40_271_872,
    }
    completions = {}
    for arm in config["arms"]:
        path = runs_root / arm / "completed.json"
        if not path.exists():
            raise RuntimeError(f"official evaluation is sealed until {path} exists")
        completion = json.loads(path.read_text())
        if completion["config_sha256"] != expected_config_hash:
            raise ValueError(f"{arm} config hash mismatch")
        if completion["corpus_sha256"] != expected_corpus_hash:
            raise ValueError(f"{arm} corpus hash mismatch")
        required = {
            "experiment": 28,
            "arm": arm,
            "pilot": False,
            "row_count": config["train_grid_count"],
            "seed": config["seed"],
            "model_revision": config["model_revision"],
            "resolved_module_count": expected_modules[arm],
            "trainable_parameters": expected_trainable[arm],
        }
        for key, value in required.items():
            if completion.get(key) != value:
                raise ValueError(f"{arm} completion mismatch: {key}")
        adapter_dir = runs_root / arm / "adapter"
        expected_files = {
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        if set(completion.get("adapter_sha256", {})) != expected_files:
            raise ValueError(f"{arm} adapter hash set mismatch")
        for name, expected_hash in completion["adapter_sha256"].items():
            target = adapter_dir / name
            if not target.exists() or file_sha256(target) != expected_hash:
                raise ValueError(f"{arm} adapter file mismatch: {name}")
        completions[arm] = completion
    return completions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--synthetic-eval", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--official-challenges", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("language_lora", "multimodal_lora"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    config = json.loads(args.config.read_text())
    expected_config_hash = file_sha256(args.config)
    expected_corpus_hash = file_sha256(args.train_corpus)
    if (
        json.loads(args.corpus_manifest.read_text())["config_sha256"]
        != expected_config_hash
    ):
        raise ValueError("corpus manifest config hash mismatch")
    completions = verify_training_completions(
        config, args.runs_root, expected_config_hash, expected_corpus_hash
    )

    rows = build_evaluation_rows(
        config,
        args.synthetic_eval,
        args.official_challenges,
        args.train_corpus,
        args.corpus_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.jsonl"
    manifest = {
        "experiment": 28,
        "arm": args.arm,
        "config_sha256": expected_config_hash,
        "train_corpus_sha256": expected_corpus_hash,
        "corpus_manifest_sha256": file_sha256(args.corpus_manifest),
        "synthetic_eval_sha256": file_sha256(args.synthetic_eval),
        "official_challenges_sha256": file_sha256(args.official_challenges),
        "adapter_completion_sha256": file_sha256(
            args.runs_root / args.arm / "completed.json"
        ),
        "evaluation_rows_sha256": canonical_hash(
            [
                {
                    "id": row["id"],
                    "split": row["split"],
                    "modality": row["modality"],
                    "grid_sha256": row["grid_sha256"],
                    "target": row["target"],
                }
                for row in rows
            ]
        ),
        "evaluator_code_sha256": file_sha256(Path(__file__)),
        "row_count": len(rows),
    }
    manifest_path = args.output_dir / "manifest.json"
    if result_path.exists() and not manifest_path.exists():
        raise ValueError("evaluation results exist without a manifest")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("evaluation resume manifest mismatch")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    expected_rows = {result_key(row): row for row in rows}
    if len(expected_rows) != len(rows):
        raise ValueError("evaluation row keys are not unique")
    completed_records: dict[tuple[str, str], dict[str, Any]] = {}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                key = result_key(record)
                if key in completed_records:
                    raise ValueError(f"duplicate result key: {key}")
                if key not in expected_rows:
                    raise ValueError(f"unknown resume result key: {key}")
                expected_row = expected_rows[key]
                for field in ("modality", "grid_sha256"):
                    if record.get(field) != expected_row[field]:
                        raise ValueError(f"resume result mismatch: {key} {field}")
                if record.get("expected") != expected_row["target"]:
                    raise ValueError(f"resume result target mismatch: {key}")
                if record.get("score") != score_description(
                    record.get("prediction"), expected_row["target"]
                ):
                    raise ValueError(f"resume result score mismatch: {key}")
                completed_records[key] = record

    model_source = args.model_path or config["model"]
    revision = None if args.model_path else config["model_revision"]
    local_only = bool(args.model_path)
    dtype = getattr(torch, config["compute_dtype"])
    processor = AutoProcessor.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=local_only,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map={"": 0},
        dtype=dtype,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        base, args.runs_root / args.arm / "adapter", is_trainable=False
    ).eval()
    model.config.use_cache = True

    with result_path.open("a") as output:
        for row in rows:
            key = (row["split"], row["id"])
            if key in completed_records:
                continue
            inputs = apply_template(
                processor,
                messages(row["grid"], row["modality"], config["image_cell_size"]),
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
            tokens = generated[0, prompt_tokens:]
            raw = processor.decode(tokens, skip_special_tokens=True)
            prediction = parse_json_object(raw)
            record = {
                "id": row["id"],
                "split": row["split"],
                "task_id": row.get("task_id"),
                "role": row.get("role"),
                "family": row.get("family"),
                "modality": row["modality"],
                "grid_sha256": row["grid_sha256"],
                "expected": row["target"],
                "prediction": prediction,
                "score": score_description(prediction, row["target"]),
                "prompt_tokens": prompt_tokens,
                "output_tokens": int(tokens.shape[0]),
                "elapsed_seconds": round(elapsed, 6),
                "raw": raw,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            completed_records[key] = record
            print(
                f"{len(completed_records):03d}/{len(rows)} {row['id']} "
                f"overall={record['score']['overall']:.4f} "
                f"exact={record['score']['exact']}",
                flush=True,
            )
    if len(completed_records) != len(rows):
        raise RuntimeError("evaluation did not materialize every expected row")
    (args.output_dir / "completed.json").write_text(
        json.dumps(
            {
                **manifest,
                "completed": True,
                "manifest_sha256": file_sha256(manifest_path),
                "results_sha256": file_sha256(result_path),
                "exact_count": sum(
                    record["score"]["exact"] for record in completed_records.values()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
