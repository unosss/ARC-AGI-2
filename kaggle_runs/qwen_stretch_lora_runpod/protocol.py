"""Frozen paired protocol and label barrier for Experiment 037."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from kaggle_runs.qwen_diarc_paired_runpod.protocol import (
    atomic_json,
    grade_item,
    sha256_file,
)


ARMS = ("control", "stretch")
DEFAULT_HARD_TASK = "195c6913"
STRETCH_MODES = ("identity", "rows", "cols", "both")


def validate_config(config: dict[str, Any]) -> None:
    frozen = {
        "experiment": 37,
        "checkpoint": "sorokin/qwen3_4b_grids15_sft139",
        "diarc_adapter": "yyxdnmd/DiARC-adapters/qwen3-4b/arc-agi-2",
        "arms": list(ARMS),
        "global_max_sequence_length": 8192,
    }
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise ValueError(f"config.{key} changed from frozen value")
    local = config.get("task_local", {})
    required = {
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
        "stretch_modes": list(STRETCH_MODES),
        "stretch_max_dim": 30,
    }
    for key, expected in required.items():
        if local.get(key) != expected:
            raise ValueError(f"config.task_local.{key} changed from frozen value")
    if local["steps"] != (
        local["train_color_permutations"] * local["train_geometries"]
    ):
        raise ValueError("task-local steps must equal color x geometry views")


def mode_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
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
        local["steps" if key == "task_steps" else key] = calibration[key]
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
    hashes = {}
    for arm in ARMS:
        for task_id in task_ids:
            path = candidate_path(work, arm, task_id)
            marker = path.with_suffix(".sha256")
            if not path.is_file() or not marker.is_file():
                raise RuntimeError(f"candidate hash missing: {arm}/{task_id}")
            recorded = json.loads(marker.read_text())
            actual = sha256_file(path)
            if recorded != {"sha256": actual, "immutable": True}:
                raise RuntimeError(f"candidate hash mismatch: {arm}/{task_id}")
            hashes[f"{arm}/{task_id}"] = actual
    atomic_json(work / "candidate_hash_barrier.json", hashes)
    return hashes


def grade(
    work: Path, task_ids: list[str], solutions: dict[str, Any]
) -> dict[str, Any]:
    hashes = verify_hash_barrier(work, task_ids)
    arms = {}
    for arm in ARMS:
        tasks = []
        for task_id in task_ids:
            payload = json.loads(candidate_path(work, arm, task_id).read_text())
            truths = solutions[task_id]
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
                "mean_best_top2_cell_accuracy": (
                    sum(item["best_top2_cell_accuracy"] for item in flat) / len(flat)
                ),
            },
        }
    metrics = (
        "task_top2_exact",
        "item_top2_exact",
        "truth_in_pool",
        "mean_best_top2_cell_accuracy",
    )
    return {
        "schema_version": 1,
        "experiment": 37,
        "candidate_hashes": hashes,
        "labels_opened_after_hash_barrier": True,
        "arms": arms,
        "pairwise_stretch_minus_control": {
            metric: arms["stretch"]["aggregate"][metric]
            - arms["control"]["aggregate"][metric]
            for metric in metrics
        },
    }
