"""GPU worker for the leakage-audited single-task stage diagnosis."""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import time

import torch
from datasets import Dataset

from arc_loader import ArcDataset, QwenFormatter
from arc_solver.nvarc_stage_diagnosis import (
    PREFIX_8_ROWS_PLUS_1_CELL,
    TASK_ID,
    canonical_reply_tokens,
    forced_prefix_8_rows_plus_1_cell,
    forced_prefix_tokens,
    summarize_token_nll,
)
from nvarc_upstream_solver import (
    CFG,
    FastLanguageModel,
    QwenDataCollatorForCompletionOnlyLM,
    UnslothFixedTrainer,
    UnslothTrainingArguments,
    inference_turbo_dfs,
    make_view_batches,
    resolve_model_dir,
)


CONFIG = json.loads(os.environ["ARC_STAGE_CONFIG"])
PHASE = os.environ["ARC_STAGE_PHASE"]
RESULT_PATH = Path(os.environ["ARC_STAGE_RESULT"])
LIFECYCLE_PATH = Path(os.environ["ARC_STAGE_LIFECYCLE"])


def event(name: str, **fields) -> None:
    with LIFECYCLE_PATH.open("a") as handle:
        handle.write(
            json.dumps({"event": name, "time": time.time(), **fields}, separators=(",", ":"))
            + "\n"
        )


def train(model, tokenizer, task):
    formatter = QwenFormatter(tokenizer)
    puzzle = ArcDataset(queries={TASK_ID: task}, keys=[TASK_ID])
    augmented = puzzle.augment(
        n=CFG["n_train_aug"], shfl_keys=True, seed=CFG["train_aug_seed"]
    ).cut_to_len(formatter=formatter, name="text", max_len=8192)
    args = UnslothTrainingArguments(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=CFG["num_epochs"],
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=CFG["learning_rate"],
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=CFG["lora_seed"],
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )
    collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    started = time.time()
    model = FastLanguageModel.for_training(model)
    with io.StringIO() as buffer, redirect_stdout(buffer), redirect_stderr(buffer):
        trainer = UnslothFixedTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=Dataset.from_list(augmented.as_list(formatter)),
            dataset_text_field="text",
            max_seq_length=8192,
            args=args,
        )
        stats = trainer.train()
        model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        del trainer
    result = {
        "steps": int(stats.global_step),
        "loss": float(stats.training_loss),
        "seconds": round(time.time() - started, 3),
        "training_examples": len(augmented.keys),
    }
    if result["steps"] != CONFIG["optimizer_steps"]:
        raise AssertionError(f"expected 128 optimizer steps, got {result['steps']}")
    return FastLanguageModel.for_inference(model), formatter, result


def decode(
    model,
    tokenizer,
    formatter,
    train_pairs,
    test_input,
    *,
    prefix=None,
    budget_seconds=None,
    dfs_window_seconds=None,
):
    prompt = formatter.fmt_train(train_pairs) + formatter.fmt_query(
        [{"input": test_input}]
    )
    query_tokens = tokenizer.encode(prompt)
    seeded = list(prefix or [])
    budget = budget_seconds or CONFIG["condition_budget_seconds"]
    dfs_window = dfs_window_seconds or CONFIG["dfs_window_seconds"]
    deadline = time.time() + budget
    max_new = formatter.max_new_tokens() - len(seeded)
    with torch.inference_mode():
        result = inference_turbo_dfs(
            model,
            [query_tokens + seeded],
            max_new,
            -float(torch.log(torch.tensor(CONFIG["min_prob"]))),
            deadline,
            min(dfs_window, budget),
        )
    candidates = []
    for score, suffix in (result[0][1] if result else []):
        complete = seeded + list(suffix)
        array = formatter.convert_tokens_to_array(complete)
        if array is not None:
            grid = array.tolist()
            if grid not in [item["grid"] for item in candidates]:
                candidates.append(
                    {
                        "grid": grid,
                        "suffix_score": float(score),
                        "generated_suffix_tokens": len(suffix),
                        "suffix_mean_score": float(score) / max(1, len(suffix)),
                    }
                )
    return {
        "budget_seconds": budget,
        "dfs_window_seconds": dfs_window,
        "prefix_token_count": len(seeded),
        "score_scope": "generated_suffix_only",
        "score_comparability": (
            "scores are comparable only within the same prefix length; "
            "cross-prefix raw or mean scores are not compared"
        ),
        "candidates": candidates,
    }


def decode_exp015_augmented_baseline(model, tokenizer, formatter, task):
    """Decode the original 2 color permutations x 8 D4 inference views."""
    puzzle = ArcDataset(queries={TASK_ID: task}, keys=[TASK_ID]).split_multi_replies()
    eval_ds = puzzle.augment(
        n=CONFIG["baseline_v2_eval_augmentations"],
        seed=CONFIG["eval_seed"],
    )
    batches = make_view_batches(
        eval_ds,
        CONFIG["baseline_v2_eval_augmentations"],
        CONFIG["baseline_v2_decode_batch"],
    )
    budget = CONFIG["baseline_v2_budget_seconds"]
    deadline = time.time() + budget
    candidates = []
    decoded_views = 0
    with torch.inference_mode():
        for subkeys in batches:
            if time.time() >= deadline:
                break
            tokens = [
                tokenizer.encode(eval_ds.get(subkey, formatter)["input"])
                for subkey in subkeys
            ]
            result = inference_turbo_dfs(
                model,
                tokens,
                formatter.max_new_tokens(),
                -float(torch.log(torch.tensor(CONFIG["min_prob"]))),
                deadline,
                CONFIG["baseline_v2_dfs_window_seconds"],
            )
            decoded_views += len(subkeys)
            for subkey_id, scored_beams in result:
                subkey = subkeys[subkey_id]
                for score, suffix in scored_beams:
                    array = formatter.convert_tokens_to_array(suffix)
                    if array is None:
                        continue
                    grid = puzzle.invert_mod(array, subkey, inv_perm=True).tolist()
                    existing = next(
                        (item for item in candidates if item["grid"] == grid), None
                    )
                    provenance = {
                        "view": subkey,
                        "view_score": float(score),
                        "generated_tokens": len(suffix),
                    }
                    if existing is None:
                        candidates.append({"grid": grid, "provenance": [provenance]})
                    else:
                        existing["provenance"].append(provenance)
    return {
        "contract": "experiment_015_inference_augmentation",
        "eval_augmentations": CONFIG["baseline_v2_eval_augmentations"],
        "geometries": 8,
        "expected_views": CONFIG["baseline_v2_eval_augmentations"] * 8,
        "decoded_views": decoded_views,
        "decode_batch": CONFIG["baseline_v2_decode_batch"],
        "budget_seconds": budget,
        "dfs_window_seconds": CONFIG["baseline_v2_dfs_window_seconds"],
        "min_prob": CONFIG["min_prob"],
        "timed_out": decoded_views < CONFIG["baseline_v2_eval_augmentations"] * 8,
        "candidates": candidates,
    }


@torch.no_grad()
def token_nll(model, query_tokens, answer_tokens, mapping):
    ids = torch.tensor([query_tokens + answer_tokens], device=model.device)
    logits = model(input_ids=ids, return_dict=True, use_cache=False).logits[0].float()
    positions = torch.arange(
        len(query_tokens) - 1,
        len(query_tokens) - 1 + len(answer_tokens),
        device=model.device,
    )
    selected = logits[positions]
    targets = torch.tensor(answer_tokens, device=model.device)
    log_probs = torch.log_softmax(selected, dim=-1)
    probabilities = torch.softmax(selected, dim=-1)
    records = []
    for index, (token, meta) in enumerate(zip(answer_tokens, mapping)):
        target_logit = selected[index, token]
        records.append(
            {
                **meta,
                "token_id": token,
                "nll": float(-log_probs[index, token]),
                "correct_probability": float(probabilities[index, token]),
                "correct_rank": int((selected[index] > target_logit).sum()) + 1,
            }
        )
    return summarize_token_nll(records)


def add_truth_flags(result, truth):
    result["truth_present"] = any(item["grid"] == truth for item in result["candidates"])
    result["exact_candidate_count"] = sum(
        item["grid"] == truth for item in result["candidates"]
    )
    return result


def run() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    task = json.loads(Path(os.environ["ARC_STAGE_SOLVER_TASK"]).read_text())[TASK_ID]
    peft = dict(
        r=256,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
            "up_proj", "down_proj", "embed_tokens", "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=CFG["lora_seed"],
        use_rslora=True,
        loftq_config=None,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolve_model_dir(),
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=8192,
    )
    model = FastLanguageModel.get_peft_model(model, **peft)
    for _, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    model, formatter, training = train(model, tokenizer, task)

    if PHASE == "ordinary":
        event("ordinary_training_completed", optimizer_steps=training["steps"])
        baseline = decode(
            model, tokenizer, formatter, task["train"], task["test"][0]["input"]
        )
        baseline_exp015_comparable = decode_exp015_augmented_baseline(
            model, tokenizer, formatter, task
        )
        event("ordinary_generation_completed")
        result = {
            "phase": PHASE,
            "training": training,
            "baseline": baseline,
            "baseline_exp015_comparable": baseline_exp015_comparable,
            "labels_available": False,
        }
    elif PHASE == "oracle":
        event("oracle_training_completed", optimizer_steps=training["steps"])
        payload = json.loads(Path(os.environ["ARC_STAGE_ORACLE_PAYLOAD"]).read_text())
        event("oracle_payload_opened")
        truth = payload["truth"]
        conditions = {}
        for name, condition in payload["conditions"].items():
            conditions[name] = add_truth_flags(
                decode(
                    model,
                    tokenizer,
                    formatter,
                    condition["train"],
                    condition["test_input"],
                ),
                truth,
            )
            conditions[name]["metadata"] = {
                key: value
                for key, value in condition.items()
                if key not in {"train", "test_input"}
            }
        crops = []
        for crop in payload["correspondence_crops"]:
            decoded = add_truth_flags(
                decode(model, tokenizer, formatter, task["train"], crop["test_input"]),
                truth,
            )
            crops.append(
                {
                    "condition_id": crop["condition_id"],
                    "bbox": crop["bbox"],
                    "is_true_source": crop["is_true_source"],
                    **decoded,
                }
            )

        hinted_crops = []
        for index in range(5):
            condition = conditions[f"transpose_hint_crop_{index}"]
            hinted_crops.append(
                {
                    "condition_id": f"transpose_hint_crop_{index}",
                    "bbox": condition["metadata"]["bbox"],
                    "is_true_source": condition["metadata"]["is_true_source"],
                    **{
                        key: value
                        for key, value in condition.items()
                        if key != "metadata"
                    },
                }
            )

        full_prompt = formatter.fmt_train(task["train"]) + formatter.fmt_query(
            [{"input": task["test"][0]["input"]}]
        )
        query_tokens = tokenizer.encode(full_prompt)
        answer_tokens, mapping = canonical_reply_tokens(truth)
        expected = tokenizer.encode(formatter.fmt_reply([truth]))
        if answer_tokens != expected:
            raise AssertionError(
                "checkpoint tokenization no longer matches one-token ARC mapping"
            )
        with model.disable_adapter():
            base_nll = token_nll(model, query_tokens, answer_tokens, mapping)
        rslora_nll = token_nll(model, query_tokens, answer_tokens, mapping)

        prefix_results = []
        for rows in payload["prefix_rows"]:
            prefix = forced_prefix_tokens(truth, rows)
            decoded = decode(
                model,
                tokenizer,
                formatter,
                task["train"],
                task["test"][0]["input"],
                prefix=prefix,
            )
            prefix_results.append(
                {
                    "prefix_rows": rows,
                    "exact_completion": any(
                        item["grid"] == truth for item in decoded["candidates"]
                    ),
                    **decoded,
                }
            )
        plus_one_prefix = forced_prefix_8_rows_plus_1_cell(truth)
        plus_one_decoded = decode(
            model,
            tokenizer,
            formatter,
            task["train"],
            task["test"][0]["input"],
            prefix=plus_one_prefix,
        )
        prefix_results.append(
            {
                "condition_id": PREFIX_8_ROWS_PLUS_1_CELL,
                "prefix_rows": None,
                "forced_full_rows": 8,
                "forced_additional_cells": [{"row": 8, "column": 0}],
                "exact_completion": any(
                    item["grid"] == truth for item in plus_one_decoded["candidates"]
                ),
                **plus_one_decoded,
            }
        )
        event("oracle_diagnostics_completed")
        result = {
            "phase": PHASE,
            "training": training,
            "conditions": conditions,
            "execution": {
                "transpose_probe": conditions["transpose_demonstration_probe"],
                "can_transpose_oracle_crop_with_hint": conditions[
                    "transpose_demonstration_probe"
                ]["truth_present"],
            },
            "correspondence": {
                "identical_budget_seconds": CONFIG["condition_budget_seconds"],
                "crops": crops,
            },
            "hinted_correspondence": {
                "identical_budget_seconds": CONFIG["condition_budget_seconds"],
                "crops": hinted_crops,
            },
            "nll": {
                "prompt_condition": "baseline_full_30x30",
                "base": base_nll,
                "rslora": rslora_nll,
                "delta_per_token_nll": (
                    rslora_nll["per_token_nll"] - base_nll["per_token_nll"]
                ),
            },
            "forced_prefix": {
                "prompt_condition": "baseline_full_30x30",
                "normalization": (
                    "DFS scores cover generated suffix tokens only. No score is "
                    "compared across prefix lengths; exact completion is the endpoint."
                ),
                "conditions": prefix_results,
            },
        }
    else:
        raise ValueError(f"unknown phase {PHASE!r}")
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
