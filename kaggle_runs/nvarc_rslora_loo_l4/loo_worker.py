"""Dedicated paired LOO worker built on the pinned NVARC implementation.

The upstream module is imported unchanged. This file owns every experimental
change: adapter-disabled Base inference, RSLoRA inference, and label-gated NLL.
"""

from __future__ import annotations

import bz2
import gc
import io
import json
import os
import pickle
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from arc_loader import ArcDataset, QwenFormatter
from nvarc_upstream_solver import (
    CFG,
    PAD_ID,
    QwenDataCollatorForCompletionOnlyLM,
    UnslothFixedTrainer,
    UnslothTrainingArguments,
    UnslothTrainer,
    FastLanguageModel,
    calc_scores,
    inference_turbo_dfs,
    make_view_batches,
    resolve_model_dir,
    stable_seed,
)


def _append_jsonl(path: str, record: dict) -> None:
    if path:
        with open(path, "a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _train(model, tokenizer, collator, formatter, puzzle_ds, max_seq_length):
    train_ds = puzzle_ds.augment(
        n=CFG["n_train_aug"], shfl_keys=True, seed=CFG["train_aug_seed"]
    )
    train_ds = train_ds.cut_to_len(
        formatter=formatter, name="text", max_len=max_seq_length
    )
    args = dict(
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
    started = time.time()
    model = FastLanguageModel.for_training(model)
    with io.StringIO() as buffer, redirect_stdout(buffer), redirect_stderr(buffer):
        trainer = UnslothFixedTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=UnslothTrainingArguments(**args),
        )
        stats = trainer.train()
        model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        del trainer
    elapsed = time.time() - started
    return model, {
        "training_steps": int(getattr(stats, "global_step", 0)),
        "training_loss": float(getattr(stats, "training_loss", float("nan"))),
        "training_seconds": round(elapsed, 3),
    }


def _decode(
    rank,
    key,
    model,
    tokenizer,
    formatter,
    puzzle_ds,
    out_dir,
    global_end_time,
):
    max_new_tokens = formatter.max_new_tokens()
    max_score = -np.log(CFG["min_prob"])
    puzzle_ds_multi = puzzle_ds.split_multi_replies()
    eval_ds = puzzle_ds_multi.augment(
        n=CFG["n_eval_aug"], seed=CFG["eval_aug_seed"]
    )
    eval_ds = eval_ds.cut_to_len(
        formatter=formatter,
        name="input",
        max_len=8192 - max_new_tokens,
    )
    batches = make_view_batches(eval_ds, CFG["n_eval_aug"], CFG["decode_batch"])
    decode_started = time.time()
    decode_deadline = min(global_end_time, decode_started + CFG["task_cap"])
    hit_cap = False
    known_scores = {}
    with torch.inference_mode():
        for subkeys in batches:
            if time.time() >= decode_deadline:
                hit_cap = True
                break
            print(f"[Rank {rank}] decoding {key}: {subkeys}", flush=True)
            tokens = [
                tokenizer.encode(eval_ds.get(subkey, formatter)["input"])
                for subkey in subkeys
            ]
            dfs_result = inference_turbo_dfs(
                model,
                tokens,
                max_new_tokens,
                max_score,
                decode_deadline,
                CFG["dfs_window"],
            )
            for subkey_id, scored_beams in dfs_result:
                subkey = subkeys[subkey_id]
                base_key = subkey.split(".")[0]
                decoded_result = []
                for beam_score, output_tokens in scored_beams:
                    if time.time() >= decode_deadline:
                        hit_cap = True
                        break
                    array = formatter.convert_tokens_to_array(output_tokens)
                    if array is None:
                        continue
                    solution = puzzle_ds_multi.invert_mod(
                        array, subkey, inv_perm=True
                    )
                    grid_id = (base_key, tuple(map(tuple, solution)))
                    if grid_id not in known_scores:
                        augmented = ArcDataset(
                            keys=[base_key],
                            queries={base_key: puzzle_ds_multi.queries[base_key]},
                            replies={base_key: [solution.tolist()]},
                        )
                        augmented = augmented.augment(
                            seed=stable_seed(base_key, CFG["score_seed_off"])
                        )
                        augmented = augmented.cut_to_len(
                            formatter=formatter,
                            name="input",
                            max_len=8192 - max_new_tokens,
                        )
                        rows = augmented.as_list(formatter)
                        known_scores[grid_id] = calc_scores(
                            [row["input"] for row in rows[:4]],
                            [row["reply"] for row in rows[:4]],
                            tokenizer,
                            model,
                        ) + calc_scores(
                            [row["input"] for row in rows[4:]],
                            [row["reply"] for row in rows[4:]],
                            tokenizer,
                            model,
                        )
                    decoded_result.append(
                        {
                            "beam_score": beam_score,
                            "score_aug": known_scores[grid_id],
                            "solution": solution,
                        }
                    )
                if decoded_result:
                    with bz2.BZ2File(os.path.join(out_dir, subkey), "w") as handle:
                        pickle.dump(decoded_result, handle)
            if time.time() >= decode_deadline:
                hit_cap = True
                break
    elapsed = time.time() - decode_started
    return {
        "decode_seconds": round(elapsed, 3),
        "decode_cap_seconds": CFG["task_cap"],
        "hit_decode_cap": hit_cap,
    }


def _score_rows(rows, tokenizer, model):
    scores = []
    lengths = []
    for offset in range(0, len(rows), CFG["decode_batch"]):
        batch = rows[offset : offset + CFG["decode_batch"]]
        scores.extend(
            calc_scores(
                [row["input"] for row in batch],
                [row["reply"] for row in batch],
                tokenizer,
                model,
            )
        )
        lengths.extend(len(tokenizer.encode(row["reply"])) for row in batch)
    return scores, lengths


def _summarize_nll(pre_scores, post_scores, lengths):
    if not (len(pre_scores) == len(post_scores) == len(lengths)):
        raise AssertionError("pre/post NLL views or token lengths differ")
    tokens = sum(lengths)
    pre_sum = sum(pre_scores)
    post_sum = sum(post_scores)
    return {
        "view_count": len(lengths),
        "token_count": tokens,
        "pre_sum_nll": pre_sum,
        "post_sum_nll": post_sum,
        "delta_sum_nll": post_sum - pre_sum,
        "pre_per_token_nll": pre_sum / tokens,
        "post_per_token_nll": post_sum / tokens,
        "delta_per_token_nll": (post_sum - pre_sum) / tokens,
        "token_lengths": lengths,
    }


def _run_nll(
    key,
    model,
    tokenizer,
    formatter,
    puzzle_ds,
    trained_weights,
    training,
):
    lifecycle_path = os.getenv("ARC_LIFECYCLE_FILE", "")
    _append_jsonl(
        lifecycle_path,
        {"fold_key": key, "event": "training_completed", "time": time.time()},
    )
    # This is the first label read in this process for this fold. There are no
    # optimizer or DFS calls below this point.
    label_path = Path(os.environ["ARC_LABEL_DIR"]) / f"{key}.json"
    truth_list = json.loads(label_path.read_text())
    _append_jsonl(
        lifecycle_path,
        {"fold_key": key, "event": "label_opened", "time": time.time()},
    )
    labeled = ArcDataset(
        queries={key: puzzle_ds.queries[key]},
        replies={key: truth_list},
        keys=[key],
    ).split_multi_replies()
    canonical = labeled.as_list(formatter)
    views = labeled.augment(
        n=CFG["n_eval_aug"], seed=CFG["eval_aug_seed"]
    ).as_list(formatter)
    model = FastLanguageModel.for_inference(model)
    with model.disable_adapter():
        canonical_pre, canonical_lengths = _score_rows(
            canonical, tokenizer, model
        )
        views_pre, view_lengths = _score_rows(views, tokenizer, model)
    set_peft_model_state_dict(model, trained_weights, adapter_name="default")
    canonical_post, canonical_post_lengths = _score_rows(
        canonical, tokenizer, model
    )
    views_post, view_post_lengths = _score_rows(views, tokenizer, model)
    if canonical_lengths != canonical_post_lengths or view_lengths != view_post_lengths:
        raise AssertionError("pre/post teacher-forced token lengths changed")
    result = {
        "fold_key": key,
        "training": training,
        "canonical": _summarize_nll(
            canonical_pre, canonical_post, canonical_lengths
        ),
        "multi_view": _summarize_nll(views_pre, views_post, view_lengths),
        "lifecycle": [
            "training_completed",
            "label_opened",
            "base_nll",
            "trained_rslora_nll",
            "nll_completed",
        ],
    }
    _append_jsonl(os.environ["ARC_NLL_FILE"], result)
    _append_jsonl(
        lifecycle_path,
        {"fold_key": key, "event": "nll_completed", "time": time.time()},
    )


def worker(rank, queue, end_time, test_path=None):
    if os.getenv("ARC_DETERMINISTIC", "1") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
    mode = os.environ["ARC_LOO_MODE"]
    if mode not in {"base_candidates", "rslora_candidates", "nll"}:
        raise ValueError(f"unknown ARC_LOO_MODE {mode!r}")
    peft_params = dict(
        r=256,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
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
    model = FastLanguageModel.get_peft_model(model, **peft_params)
    for _, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    checkpoint = {
        name: value.clone().detach()
        for name, value in get_peft_model_state_dict(
            model, adapter_name="default"
        ).items()
    }
    collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    formatter = QwenFormatter(tokenizer=tokenizer)
    arc_set = ArcDataset.from_file(test_path)
    out_dir = CFG["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    while time.time() < end_time:
        key = queue.get()
        if key is None:
            break
        training = {
            "training_steps": 0,
            "training_loss": None,
            "training_seconds": 0.0,
        }
        try:
            set_peft_model_state_dict(
                model, checkpoint.copy(), adapter_name="default"
            )
            puzzle_ds = arc_set.change_keys([key])
            if mode != "base_candidates":
                model, training = _train(
                    model,
                    tokenizer,
                    collator,
                    formatter,
                    puzzle_ds,
                    8192,
                )
            model = FastLanguageModel.for_inference(model)
            gc.collect()
            torch.cuda.empty_cache()
            if mode == "nll":
                trained_weights = {
                    name: value.clone().detach()
                    for name, value in get_peft_model_state_dict(
                        model, adapter_name="default"
                    ).items()
                }
                _run_nll(
                    key,
                    model,
                    tokenizer,
                    formatter,
                    puzzle_ds,
                    trained_weights,
                    training,
                )
            else:
                if mode == "base_candidates":
                    with model.disable_adapter():
                        decoding = _decode(
                            rank,
                            key,
                            model,
                            tokenizer,
                            formatter,
                            puzzle_ds,
                            out_dir,
                            end_time,
                        )
                else:
                    decoding = _decode(
                        rank,
                        key,
                        model,
                        tokenizer,
                        formatter,
                        puzzle_ds,
                        out_dir,
                        end_time,
                    )
                _append_jsonl(
                    os.getenv("ARC_TELEMETRY_FILE", ""),
                    {
                        "fold_key": key,
                        "arm": "base"
                        if mode == "base_candidates"
                        else "rslora",
                        **training,
                        **decoding,
                    },
                )
        except Exception as exc:
            print(
                f"[Rank {rank}] ERROR {mode} {key}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
        print(f"[Rank {rank}] finished {mode} {key}", flush=True)
