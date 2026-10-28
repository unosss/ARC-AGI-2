"""Label-free held-out inference for Experiment 042."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from model_io import (
    apply_template,
    build_messages,
    parse_grid,
    parse_program,
)
from program_sandbox import ProgramRejected, run_program, validate_program


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


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("direct", "program"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        set_seed,
    )

    config = json.loads(args.config.read_text())
    completion = json.loads((args.adapter_dir.parent / "completed.json").read_text())
    if completion["arm"] != args.arm:
        raise ValueError("adapter arm does not match inference arm")
    if completion["config_sha256"] != file_sha256(args.config):
        raise ValueError("adapter was trained with a different config")
    current_model_identity = resolved_model_identity(
        args.model_path, config["model"], config["model_revision"]
    )
    if completion["resolved_model"] != current_model_identity:
        raise ValueError("inference model does not match the training model")
    for name, expected in completion["adapter_sha256"].items():
        if file_sha256(args.adapter_dir / name) != expected:
            raise ValueError(f"adapter hash mismatch: {name}")

    rows = load_jsonl(args.challenges)
    if any("query_output" in row or "program" in row for row in rows):
        raise ValueError("challenge file contains sealed labels")
    set_seed(int(config["train_seed"]))
    model_source = args.model_path or config["model"]
    revision = None if args.model_path else config["model_revision"]
    local_only = bool(args.model_path)
    dtype = getattr(torch, config["compute_dtype"])
    processor = AutoProcessor.from_pretrained(
        args.adapter_dir, local_files_only=True, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=local_only,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=dtype,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidates.jsonl"
    manifest = {
        "schema_version": 1,
        "experiment": 42,
        "arm": args.arm,
        "budget": completion["budget"],
        "config_sha256": file_sha256(args.config),
        "challenges_sha256": file_sha256(args.challenges),
        "adapter_sha256": completion["adapter_sha256"],
        "resolved_model": current_model_identity,
    }
    manifest_path = args.output_dir / "inference_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing inference manifest does not match")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    existing = {}
    if candidate_path.exists():
        for row in load_jsonl(candidate_path):
            if row.get("id") in existing:
                raise ValueError(f"duplicate resumed candidate ID: {row.get('id')}")
            if row.get("arm") != args.arm:
                raise ValueError("resumed candidate arm mismatch")
            existing[row["id"]] = row
    challenge_by_id = {row["id"]: row for row in rows}
    if len(challenge_by_id) != len(rows):
        raise ValueError("challenge file contains duplicate IDs")
    if not set(existing) <= set(challenge_by_id):
        raise ValueError("resumed candidates contain an unknown challenge ID")
    for item_id, item in existing.items():
        if item.get("puzzle_key") != challenge_by_id[item_id]["puzzle_key"]:
            raise ValueError("resumed candidate puzzle key mismatch")
    ordered_results = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        if row["id"] in existing:
            ordered_results.append(existing[row["id"]])
            continue
        messages = build_messages(
            row,
            args.arm,
            config["modality"],
            int(config["image_cell_size"]),
            include_answer=False,
        )
        inputs = apply_template(processor, messages, add_generation_prompt=True)
        inputs = {
            key: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[1])
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(config["max_new_tokens"]),
                do_sample=False,
                use_cache=True,
            )
        generated_ids = generated[0, prompt_tokens:]
        raw = processor.decode(generated_ids, skip_special_tokens=True)
        prediction = None
        program = None
        replay_exact = None
        program_validated = None
        errors: list[str] = []
        if args.arm == "direct":
            prediction = parse_grid(raw)
            if prediction is None:
                errors.append("no valid grid parsed")
        else:
            program = parse_program(raw)
            if program is None:
                errors.append("no program parsed")
                replay_exact = False
                program_validated = False
            else:
                try:
                    validate_program(program)
                    program_validated = True
                except ProgramRejected as error:
                    program_validated = False
                    replay_exact = False
                    errors.append(str(error))
                if program_validated:
                    inputs_to_run = [pair["input"] for pair in row["demos"]]
                    inputs_to_run.append(row["query_input"])
                    executions = run_program(
                        program,
                        inputs_to_run,
                        timeout_seconds=float(config["program_timeout_seconds"]),
                        memory_bytes=int(config["program_memory_bytes"]),
                    )
                    for item in executions:
                        if "error" in item:
                            errors.append(item["error"])
                    replay_exact = all(
                        execution.get("output") == pair["output"]
                        for execution, pair in zip(executions[:-1], row["demos"])
                    )
                    if replay_exact and executions and "output" in executions[-1]:
                        prediction = executions[-1]["output"]
                    elif not replay_exact:
                        errors.append("program did not replay every demonstration")
        elapsed = time.monotonic() - started
        result = {
            "id": row["id"],
            "puzzle_key": row["puzzle_key"],
            "challenge_sha256": hashlib.sha256(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode()
            ).hexdigest(),
            "arm": args.arm,
            "prediction": prediction,
            "program": program,
            "program_validated": program_validated,
            "demo_replay_exact": replay_exact,
            "raw_generation": raw,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(generated_ids.shape[0]),
            "elapsed_seconds": elapsed,
            "errors": sorted(set(errors)),
        }
        ordered_results.append(result)
        write_atomic(
            candidate_path,
            "".join(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                for item in ordered_results
            ),
        )

    completion_payload = {
        **manifest,
        "candidate_count": len(ordered_results),
        "candidate_sha256": file_sha256(candidate_path),
        "complete": len(ordered_results) == len(rows),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }
    write_atomic(
        args.output_dir / "inference_completed.json",
        json.dumps(completion_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    print(json.dumps(completion_payload, indent=2))


if __name__ == "__main__":
    main()
