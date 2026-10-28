"""CPU-only helpers for leakage-safe paired NVARC leave-one-out evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from arc_solver.ttt_eval import is_valid_grid


FOLD_SEPARATOR = "~loo~"


def make_fold_key(task_id: str, train_index: int) -> str:
    """Return an unambiguous, filename-safe fold key."""
    if not task_id or FOLD_SEPARATOR in task_id or "_" in task_id:
        raise ValueError(f"unsupported task id: {task_id!r}")
    if train_index < 0:
        raise ValueError("train index must be non-negative")
    return f"{task_id}{FOLD_SEPARATOR}{train_index:03d}"


def parse_fold_key(fold_key: str) -> tuple[str, int]:
    parts = fold_key.split(FOLD_SEPARATOR)
    if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
        raise ValueError(f"invalid fold key: {fold_key!r}")
    return parts[0], int(parts[1])


def build_loo_bundle(
    challenges: dict[str, Any],
    task_ids: list[str],
    *,
    minimum_demos: int = 2,
) -> tuple[dict[str, Any], dict[str, list[list[list[int]]]], dict[str, Any]]:
    """Build pseudo challenges, held labels, and fold metadata separately.

    Every original training pair becomes one fold. Its input is the only test
    item; all other complete input/output pairs remain demonstrations.
    """
    pseudo: dict[str, Any] = {}
    labels: dict[str, list[list[list[int]]]] = {}
    folds: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        if task_id not in challenges:
            raise KeyError(f"missing challenge for {task_id}")
        train = challenges[task_id].get("train")
        if not isinstance(train, list) or len(train) <= minimum_demos:
            raise ValueError(
                f"{task_id} needs at least {minimum_demos + 1} train pairs"
            )
        for train_index, held in enumerate(train):
            if (
                not isinstance(held, dict)
                or not is_valid_grid(held.get("input"))
                or not is_valid_grid(held.get("output"))
            ):
                raise ValueError(f"invalid train pair {task_id}[{train_index}]")
            fold_key = make_fold_key(task_id, train_index)
            remaining = [
                copy.deepcopy(pair)
                for index, pair in enumerate(train)
                if index != train_index
            ]
            if len(remaining) < minimum_demos or any(
                not is_valid_grid(pair.get("input"))
                or not is_valid_grid(pair.get("output"))
                for pair in remaining
            ):
                raise ValueError(f"invalid remaining demos for {fold_key}")
            pseudo[fold_key] = {
                "train": remaining,
                "test": [{"input": copy.deepcopy(held["input"])}],
            }
            labels[fold_key] = [copy.deepcopy(held["output"])]
            folds.append(
                {
                    "fold_key": fold_key,
                    "task_id": task_id,
                    "held_train_index": train_index,
                    "remaining_train_pairs": len(remaining),
                }
            )
    return pseudo, labels, {"fold_count": len(folds), "folds": folds}


def build_loo_solver_challenges(
    challenges: dict[str, Any],
    task_ids: list[str],
    *,
    minimum_demos: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build label-free folds without reading held outputs into a label map."""
    pseudo: dict[str, Any] = {}
    folds = []
    for task_id in sorted(task_ids):
        if task_id not in challenges:
            raise KeyError(f"missing challenge for {task_id}")
        train = challenges[task_id].get("train")
        if not isinstance(train, list) or len(train) <= minimum_demos:
            raise ValueError(
                f"{task_id} needs at least {minimum_demos + 1} train pairs"
            )
        for train_index, held in enumerate(train):
            if not isinstance(held, dict) or not is_valid_grid(held.get("input")):
                raise ValueError(f"invalid train input {task_id}[{train_index}]")
            fold_key = make_fold_key(task_id, train_index)
            remaining = [
                copy.deepcopy(pair)
                for index, pair in enumerate(train)
                if index != train_index
            ]
            if len(remaining) < minimum_demos or any(
                not is_valid_grid(pair.get("input"))
                or not is_valid_grid(pair.get("output"))
                for pair in remaining
            ):
                raise ValueError(f"invalid remaining demos for {fold_key}")
            pseudo[fold_key] = {
                "train": remaining,
                "test": [{"input": copy.deepcopy(held["input"])}],
            }
            folds.append(
                {
                    "fold_key": fold_key,
                    "task_id": task_id,
                    "held_train_index": train_index,
                    "remaining_train_pairs": len(remaining),
                }
            )
    return pseudo, {"fold_count": len(folds), "folds": folds}


def load_loo_labels(
    challenges: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, list[list[list[int]]]]:
    """Load held outputs for parent-only finalization after candidates exit."""
    labels = {}
    for fold in metadata["folds"]:
        output = challenges[fold["task_id"]]["train"][
            fold["held_train_index"]
        ].get("output")
        if not is_valid_grid(output):
            raise ValueError(f"invalid held output for {fold['fold_key']}")
        labels[fold["fold_key"]] = [copy.deepcopy(output)]
    return labels


def write_loo_bundle(
    challenges_path: Path,
    task_ids: list[str],
    solver_dir: Path,
    parent_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write solver JSON separately from parent-only labels and metadata."""
    challenges = json.loads(challenges_path.read_text())
    pseudo, labels, metadata = build_loo_bundle(challenges, task_ids)
    solver_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)
    challenge_path = solver_dir / "loo_challenges.json"
    labels_path = parent_dir / "loo_labels.json"
    metadata_path = parent_dir / "loo_metadata.json"
    challenge_path.write_text(json.dumps(pseudo, separators=(",", ":")))
    labels_path.write_text(json.dumps(labels, separators=(",", ":")))
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")))
    return challenge_path, labels_path, metadata_path


def write_per_fold_labels(labels: dict[str, Any], label_dir: Path) -> None:
    """Materialize one label file per fold for lifecycle-safe NLL workers."""
    label_dir.mkdir(parents=True, exist_ok=True)
    for fold_key in sorted(labels):
        (label_dir / f"{fold_key}.json").write_text(
            json.dumps(labels[fold_key], separators=(",", ":"))
        )


def _items_by_fold(oracle_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for task in oracle_report.get("tasks", []):
        task_items = task.get("items", [])
        if len(task_items) != 1:
            raise ValueError(f"LOO fold {task.get('task_id')} must have one item")
        items[task["task_id"]] = task_items[0]
    return items


def aggregate_paired_report(
    metadata: dict[str, Any],
    arm_oracles: dict[str, dict[str, Any]],
    submissions: dict[str, dict[str, Any]],
    telemetry: dict[str, dict[str, dict[str, Any]]],
    nll_records: list[dict[str, Any]],
    *,
    nll_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Join per-arm candidates and NLL records into paired fold outcomes."""
    if set(arm_oracles) != {"base", "rslora"}:
        raise ValueError("base and rslora oracle reports are required")
    arm_items = {arm: _items_by_fold(report) for arm, report in arm_oracles.items()}
    nll_by_fold = {record["fold_key"]: record for record in nll_records}
    folds = []
    outcomes = {"rslora_win": 0, "base_win": 0, "tie": 0}
    nll_outcomes = {
        view: {"improve": 0, "tie": 0, "worsen": 0}
        for view in ("canonical", "multi_view")
    }
    for fold_meta in metadata["folds"]:
        fold_key = fold_meta["fold_key"]
        arms = {}
        top2 = {}
        for arm in ("base", "rslora"):
            item = arm_items[arm][fold_key]
            attempt = submissions[arm][fold_key][0]
            tel = telemetry.get(arm, {}).get(fold_key, {})
            top2[arm] = bool(item["score_kgmon_top2"])
            arms[arm] = {
                "truth_present": item["truth_present"],
                "truth_rank": item["score_kgmon_truth_rank"],
                "truth_ranks": {
                    "score_kgmon": item["score_kgmon_truth_rank"],
                    "score_full_probmul_3": item[
                        "score_full_probmul_3_truth_rank"
                    ],
                },
                "top2": top2[arm],
                "top2_by_scorer": {
                    "score_kgmon": item["score_kgmon_top2"],
                    "score_full_probmul_3": item[
                        "score_full_probmul_3_top2"
                    ],
                },
                "best_candidate_cell_accuracy": item[
                    "best_candidate_cell_accuracy"
                ],
                "unique_candidate_count": item["unique_candidate_count"],
                "top2_outputs": [attempt["attempt_1"], attempt["attempt_2"]],
                "decode_seconds": tel.get("decode_seconds"),
                "decode_cap_seconds": tel.get("decode_cap_seconds"),
                "hit_decode_cap": tel.get("hit_decode_cap"),
                "training_steps": tel.get("training_steps", 0),
                "training_loss": tel.get("training_loss"),
                "training_seconds": tel.get("training_seconds", 0.0),
            }
        outcome = (
            "rslora_win"
            if top2["rslora"] and not top2["base"]
            else "base_win"
            if top2["base"] and not top2["rslora"]
            else "tie"
        )
        outcomes[outcome] += 1
        nll = nll_by_fold.get(fold_key, {})
        fold_nll_outcomes = {}
        for view in ("canonical", "multi_view"):
            delta = nll.get(view, {}).get("delta_sum_nll")
            result = (
                None
                if delta is None
                else "improve"
                if delta < -nll_tolerance
                else "worsen"
                if delta > nll_tolerance
                else "tie"
            )
            fold_nll_outcomes[view] = result
            if result is not None:
                nll_outcomes[view][result] += 1
        folds.append(
            {
                **fold_meta,
                "base": arms["base"],
                "rslora": arms["rslora"],
                "paired_top2_outcome": outcome,
                "nll": nll,
                "nll_outcomes": fold_nll_outcomes,
            }
        )
    return {
        "fold_count": len(folds),
        "folds": folds,
        "aggregate": {
            "paired_top2": outcomes,
            "nll": nll_outcomes,
            "base_truth_coverage": sum(f["base"]["truth_present"] for f in folds),
            "rslora_truth_coverage": sum(
                f["rslora"]["truth_present"] for f in folds
            ),
            "base_top2": sum(f["base"]["top2"] for f in folds),
            "rslora_top2": sum(f["rslora"]["top2"] for f in folds),
        },
    }
