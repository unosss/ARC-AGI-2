from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import zlib
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle
import traceback

logging.disable(logging.WARNING)

# ---------------------------------------------------------------------------
# Run configuration (all overridable through environment variables so the
# same solver can run the LB-proven primary pass and a wider secondary pass).
# Defaults == the public LB 33.89 NVARC configuration.
# ---------------------------------------------------------------------------
def _env(name, default, cast):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default

CFG = dict(
    model_path      = _env("ARC_MODEL_PATH", "", str),
    out_dir         = _env("ARC_OUT_DIR", "/kaggle/inference_outputs", str),
    lora_seed       = _env("ARC_LORA_SEED", 42, int),
    train_aug_seed  = _env("ARC_TRAIN_AUG_SEED", 1, int),
    n_train_aug     = _env("ARC_N_TRAIN_AUG", 16, int),     # x8 geometries = 128 sequences
    num_epochs      = _env("ARC_EPOCHS", 1, int),
    learning_rate   = _env("ARC_LR", 5e-5, float),
    eval_aug_seed   = _env("ARC_EVAL_AUG_SEED", 2, int),
    n_eval_aug      = _env("ARC_N_EVAL_AUG", 2, int),       # x8 geometries = 16 decoded views
    min_prob        = _env("ARC_MIN_PROB", 0.2, float),     # DFS cumulative prob threshold
    dfs_window      = _env("ARC_DFS_WINDOW", 540.0, float), # seconds per DFS call
    task_cap        = _env("ARC_TASK_CAP", 1200.0, float),  # seconds per task (decode phase)
    score_seed_off  = _env("ARC_SCORE_SEED_OFFSET", 0, int),
    decode_batch    = _env("ARC_DECODE_BATCH", 4, int),
)
# ---------------------------------------------------------------------------
# v5 (theory v32 landing, revision 2): per-task configuration overrides +
# run telemetry + model-side intervention keys.
# The controller cell writes {task_key: {cfg_key: value}} to ARC_TASK_CONFIG_FILE;
# each worker applies the overrides for the task it is about to run and restores
# the process defaults afterwards.  Telemetry (one JSON line per task) records
# elapsed time, whether the task hit its cap (the search-pressure signal), the
# shard count and the applied overrides -- the real-ARC diagnostic source.
#
# v5: train_aug_seed / n_train_aug join the overridable keys.  These are read
# inside the task loop (puzzle_ds.augment(..., seed=CFG["train_aug_seed"])),
# so per-task overrides take effect: an empty candidate pool (express == 0)
# can be given a DIFFERENT trained model, not just more decoding effort --
# the footprint-matched intervention for a model-side bottleneck.
# ---------------------------------------------------------------------------
_TASK_CFG_KEYS = ("task_cap", "n_eval_aug", "min_prob", "dfs_window",
                  "score_seed_off", "eval_aug_seed", "train_aug_seed",
                  "n_train_aug")

def _load_task_configs():
    path = os.getenv("ARC_TASK_CONFIG_FILE", "")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return {}
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if kk in _TASK_CFG_KEYS}
    return out

_TASK_CFG = _load_task_configs()

def _apply_task_overrides(key):
    ov = _TASK_CFG.get(key)
    if not ov:
        return {}
    applied = {}
    for k in _TASK_CFG_KEYS:
        if k in ov:
            applied[k] = CFG[k]
            CFG[k] = ov[k]
    return applied

def _restore_task_overrides(applied):
    for k, v in applied.items():
        CFG[k] = v

_TEL_PATH = os.getenv("ARC_TELEMETRY_FILE", "")

def _tel(rec):
    if not _TEL_PATH:
        return
    try:
        with open(_TEL_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


def resolve_model_dir():
    if CFG["model_path"] and os.path.isdir(CFG["model_path"]):
        return CFG["model_path"]
    candidates = [
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    import glob
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if "grids15" in d and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise FileNotFoundError("qwen3_4b_grids15_sft139 model directory not found under /kaggle/input")


def stable_seed(key, offset=0):
    return (zlib.crc32(key.encode("utf-8")) + offset) % (1024 ** 2)


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(loss, "clone"):
            loss = loss.clone()
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


# Throughput patch (algebraically identical to the baseline): only the 12
# ARC-token NLL values are moved to the CPU instead of the whole vocabulary.
_ARC_TOKEN_ID_CACHE = {}


def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, dfs_window) -> dict:

    n = logits.size(0)

    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0])

    while time.time() - start_time < dfs_window and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
            dfs_window=dfs_window,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, dfs_window):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
        dfs_window=dfs_window,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
    return result


def make_view_batches(eval_ds, n_perm, batch_size):
    """Group decoded views into batches. With n_perm=2 / batch_size=4 this is
    byte-for-byte the baseline batching ([id,rot180],[rot90,rot270],[T,T.rot180],[T.rot90,T.rot270])."""
    test_id_to_subkeys = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        test_id_to_subkeys[test_id].append(subkey)
    # sorted geometry order: 0 id, 1 rot90, 2 rot180, 3 rot270, 4 T, 5 T.rot90, 6 T.rot180, 7 T.rot270
    groups_a = [0, 2, 1, 3]          # untransposed geometries
    groups_b = [4, 6, 5, 7]          # transposed geometries
    batches = []
    for geos in (groups_a, groups_b):
        for test_id, subkeys in test_id_to_subkeys.items():
            if n_perm == 2 and batch_size == 4:
                # exact baseline batches: [id,rot180], [rot90,rot270], ...
                for a, b in ((geos[0], geos[1]), (geos[2], geos[3])):
                    batches.append(subkeys[a*n_perm:(a+1)*n_perm] + subkeys[b*n_perm:(b+1)*n_perm])
            else:
                # Batched DFS needs equal token lengths in a batch. Within a geometry group the
                # shape classes are: rot0/rot180 = H x W, rot90/rot270 = W x H, so pair them.
                for a, b in ((geos[0], geos[1]), (geos[2], geos[3])):
                    views = subkeys[a*n_perm:(a+1)*n_perm] + subkeys[b*n_perm:(b+1)*n_perm]
                    for i in range(0, len(views), batch_size):
                        batches.append(views[i:i+batch_size])
    return batches


def worker(rank, queue, end_time, test_path=None):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if os.getenv("ARC_DETERMINISTIC", "1") == "1":
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=CFG["lora_seed"],
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
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

    max_seq_length = 8192

    model_dir = resolve_model_dir()
    print(f"[Rank {rank}] model dir: {model_dir}")
    print(f"[Rank {rank}] config: {CFG}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    if test_path is None:
        if rerun_mode:
            test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        else:
            test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = CFG["out_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    while True:

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()

        applied = _apply_task_overrides(key)
        max_score = -np.log(CFG["min_prob"])
        hit_cap = False

        try:
            torch.cuda.reset_peak_memory_stats()

            set_peft_model_state_dict(
                model,
                default_weights.copy(),
                adapter_name="default",
            )

            model = FastLanguageModel.for_training(model)

            puzzle_ds = arc_test_set.change_keys([key])

            train_ds = puzzle_ds.augment(n=CFG["n_train_aug"], shfl_keys=True, seed=CFG["train_aug_seed"])
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):

                trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )

                stats = trainer.train()

                model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

                del trainer

            model = FastLanguageModel.for_inference(model)

            gc.collect()
            torch.cuda.empty_cache()

            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

            torch.cuda.reset_peak_memory_stats()

            print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

            puzzle_ds_multi = puzzle_ds.split_multi_replies()

            eval_ds = puzzle_ds_multi.augment(n=CFG["n_eval_aug"], seed=CFG["eval_aug_seed"])
            eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

            batches = make_view_batches(eval_ds, CFG["n_eval_aug"], CFG["decode_batch"])

            with torch.inference_mode():

                known_scores = {}

                for subkeys in batches:

                    spend_time = time.time() - start_time
                    if spend_time > CFG["task_cap"] or time.time() > end_time:
                        print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                        hit_cap = True
                        break

                    print(f"[Rank {rank}] decoding {subkeys}")

                    tokens = []
                    for subkey in subkeys:
                        data = eval_ds.get(subkey, formatter)
                        tokens.append(tokenizer.encode(data["input"]))

                    dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time, CFG["dfs_window"])

                    for subkey_id, scored_beams in dfs_result:

                        subkey = subkeys[subkey_id]
                        bk = subkey.split(".")[0]
                        decoded_result = []

                        for beam_score, tokens_ in scored_beams:

                            array = formatter.convert_tokens_to_array(tokens_)
                            if array is None:
                                continue

                            solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                            grid_id = (bk, tuple(map(tuple, solution)))

                            if grid_id in known_scores:
                                augmented_scores = known_scores[grid_id]
                            else:
                                print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                                aug_dataset = ArcDataset(
                                    keys=[bk],
                                    queries={bk: puzzle_ds_multi.queries.get(bk)},
                                    replies={bk: [solution.tolist()]},
                                )
                                aug_dataset = aug_dataset.augment(seed=stable_seed(bk, CFG["score_seed_off"]))
                                aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                                aug_queries = []
                                aug_answers = []
                                for augmented_sample in aug_dataset.as_list(formatter):
                                    aug_queries.append(augmented_sample["input"])
                                    aug_answers.append(augmented_sample["reply"])
                                augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                                augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                                augmented_scores = augmented_scores1 + augmented_scores2
                                known_scores[grid_id] = augmented_scores

                            decoded_result.append({
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                            })

                        if len(decoded_result):
                            with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                                pickle.dump(decoded_result, f)

            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")

        except Exception as e:
            print(f"[Rank {rank}] ERROR on puzzle {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                model = FastLanguageModel.for_inference(model)
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.synchronize()

        spend_time = time.time() - start_time
        _restore_task_overrides(applied)
        try:
            n_shards = len([f for f in os.listdir(dir_outputs) if f.startswith(key + "_")])
        except Exception:
            n_shards = -1
        _tel({"task": key, "pass": os.getenv("ARC_PASS_NAME", "?"),
              "elapsed": round(spend_time, 1), "hit_cap": hit_cap,
              "n_shards": n_shards, "overrides": _TASK_CFG.get(key, {})})
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")

