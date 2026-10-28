"""Frozen protocol, sealing barrier, and parent-only grading for Experiment 036."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

ARMS = ("base", "diarc")
DEFAULT_HARD_TASK = "195c6913"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(value) + "\n")
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> None:
    frozen = {
        "experiment": 36,
        "checkpoint": "sorokin/qwen3_4b_grids15_sft139",
        "diarc_adapter": "yyxdnmd/DiARC-adapters/qwen3-4b/arc-agi-2",
        "arms": list(ARMS),
        "structure": {"arm": "text_control", "scale": 0.0, "trainable": False},
    }
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise ValueError(f"config.{key} changed from frozen value")
    local = config.get("task_local", {})
    required_local = {
        "type": "RSLoRA",
        "rank": 256,
        "alpha": 32,
        "dropout": 0.0,
        "steps": 128,
        "train_color_permutations": 16,
        "train_geometries": 8,
        "decode_color_permutations": 1,
        "decode_geometries": 8,
        "rescore_views": 8,
        "rescore_batch_size": 1,
        "merge_for_inference": False,
    }
    for key, expected in required_local.items():
        if local.get(key) != expected:
            raise ValueError(f"config.task_local.{key} changed from frozen value")
    expected_dfs = {
        "batch_size": 4,
        "max_nodes_per_view": 4096,
        "max_model_forward_tokens_per_view": 12288,
        "max_candidates_per_view": 128,
        "max_new_tokens": 931,
        "emergency_seconds_per_view": 900,
        "min_prob": 0.2,
    }
    if config.get("dfs") != expected_dfs:
        raise ValueError("config.dfs changed from frozen production budget")
    expected_diarc = config.get("diarc_expected", {})
    if (
        expected_diarc.get("rank") != 256
        or expected_diarc.get("use_rslora") is not False
        or tuple(expected_diarc.get("target_modules", ())) != TARGET_MODULES
    ):
        raise ValueError("released DiARC adapter contract changed")


def mode_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a copy; only calibration may reduce registered compute budgets."""
    validate_config(config)
    result = copy.deepcopy(config)
    if mode in {"smoke", "hard-exploratory"}:
        return result
    if mode != "calibrate":
        raise ValueError(f"unknown mode: {mode}")
    calibration = result["calibration"]
    local = result["task_local"]
    for key in (
        "task_steps",
        "train_color_permutations",
        "train_geometries",
        "decode_color_permutations",
        "decode_geometries",
        "rescore_views",
    ):
        destination = "steps" if key == "task_steps" else key
        local[destination] = calibration[key]
    result["dfs"] = copy.deepcopy(calibration["dfs"])
    return result


def reject_test_outputs(challenges: dict[str, Any]) -> None:
    if any(
        "output" in item
        for task in challenges.values()
        for item in task.get("test", [])
    ):
        raise ValueError("label-blind worker refuses challenge test outputs")


def candidate_path(work: Path, arm: str, task_id: str) -> Path:
    return work / "arms" / arm / "candidates" / f"{task_id}.json"


def seal_candidate(path: Path, payload: dict[str, Any]) -> str:
    atomic_json(path, payload)
    digest = sha256_file(path)
    atomic_json(path.with_suffix(".sha256"), {"sha256": digest, "immutable": True})
    return digest


def verify_hash_barrier(work: Path, task_ids: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for arm in ARMS:
        for task_id in task_ids:
            path = candidate_path(work, arm, task_id)
            marker = path.with_suffix(".sha256")
            if not path.is_file() or not marker.is_file():
                raise RuntimeError(f"labels sealed; candidate hash missing: {arm}/{task_id}")
            recorded = json.loads(marker.read_text())
            actual = sha256_file(path)
            if recorded != {"sha256": actual, "immutable": True}:
                raise RuntimeError(f"labels sealed; candidate hash mismatch: {arm}/{task_id}")
            hashes[f"{arm}/{task_id}"] = actual
    atomic_json(work / "candidate_hash_barrier.json", hashes)
    return hashes


def cell_accuracy(candidate: list[list[int]], truth: list[list[int]]) -> float:
    if len(candidate) != len(truth) or any(
        len(left) != len(right) for left, right in zip(candidate, truth)
    ):
        return 0.0
    count = sum(map(len, truth))
    if count == 0:
        return 1.0
    correct = sum(
        left == right
        for candidate_row, truth_row in zip(candidate, truth)
        for left, right in zip(candidate_row, truth_row)
    )
    return correct / count


def grade_item(candidates: list[dict[str, Any]], truth: list[list[int]]) -> dict[str, Any]:
    grids = [candidate["grid"] for candidate in candidates]
    rank = grids.index(truth) + 1 if truth in grids else None
    return {
        "top1_exact": rank == 1,
        "top2_exact": rank is not None and rank <= 2,
        "truth_in_pool": rank is not None,
        "truth_rank": rank,
        "generation_miss": rank is None,
        "selection_miss": rank is not None and rank > 2,
        "best_top2_cell_accuracy": max(
            (cell_accuracy(grid, truth) for grid in grids[:2]), default=0.0
        ),
        "ranked_votes": [
            {"rank": index + 1, "votes": candidate.get("votes", 0)}
            for index, candidate in enumerate(candidates)
        ],
    }


def grade(
    work: Path, task_ids: list[str], solutions: dict[str, Any]
) -> dict[str, Any]:
    hashes = verify_hash_barrier(work, task_ids)
    if any(task_id not in solutions for task_id in task_ids):
        raise ValueError("solutions do not contain every sealed task")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        tasks = []
        for task_id in task_ids:
            payload = json.loads(candidate_path(work, arm, task_id).read_text())
            truths = solutions[task_id]
            if len(payload["ranked_candidates"]) != len(truths):
                raise ValueError(f"test item count mismatch: {arm}/{task_id}")
            items = [
                grade_item(candidates, truth)
                for candidates, truth in zip(payload["ranked_candidates"], truths)
            ]
            tasks.append(
                {
                    "task_id": task_id,
                    "task_top2_exact": all(item["top2_exact"] for item in items),
                    "items": items,
                    "pre_task_loo_nll": payload["pre_task_loo_nll"],
                    "post_task_loo_nll": payload["post_task_loo_nll"],
                    "timing": payload["timing"],
                    "peak_memory_bytes": payload["peak_memory_bytes"],
                }
            )
        flat = [item for task in tasks for item in task["items"]]
        arms[arm] = {
            "tasks": tasks,
            "aggregate": {
                "task_top2_exact": sum(task["task_top2_exact"] for task in tasks),
                "item_top2_exact": sum(item["top2_exact"] for item in flat),
                "truth_in_pool": sum(item["truth_in_pool"] for item in flat),
                "generation_misses": sum(item["generation_miss"] for item in flat),
                "selection_misses": sum(item["selection_miss"] for item in flat),
                "mean_best_top2_cell_accuracy": (
                    sum(item["best_top2_cell_accuracy"] for item in flat) / len(flat)
                    if flat
                    else 0.0
                ),
            },
        }
    deltas = {}
    for metric in (
        "task_top2_exact",
        "item_top2_exact",
        "truth_in_pool",
        "generation_misses",
        "selection_misses",
        "mean_best_top2_cell_accuracy",
    ):
        deltas[metric] = (
            arms["diarc"]["aggregate"][metric] - arms["base"]["aggregate"][metric]
        )
    return {
        "schema_version": 1,
        "experiment": 36,
        "candidate_hashes": hashes,
        "labels_opened_after_hash_barrier": True,
        "arms": arms,
        "pairwise_diarc_minus_base": deltas,
    }
