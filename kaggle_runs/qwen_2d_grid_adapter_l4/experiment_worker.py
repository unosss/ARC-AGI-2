"""Four-GPU arm scheduler and parent-only finalizer for Experiment 021."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from arc_solver.qwen2d_experiment import LabelLifecycle, paired_contrast, sha256_json
try:
    from orchestration_safety import validate_final_report
except ModuleNotFoundError:  # Local package import; archive imports top-level.
    from .orchestration_safety import validate_final_report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _score_item(candidates: list[dict[str, Any]], truth: list[list[int]]) -> dict[str, Any]:
    grids = [candidate["grid"] for candidate in candidates]
    rank = grids.index(truth) + 1 if truth in grids else None
    best_cell = 0.0
    shape_hit = False
    for grid in grids:
        same_shape = len(grid) == len(truth) and all(
            len(left) == len(right) for left, right in zip(grid, truth)
        )
        shape_hit |= same_shape
        if same_shape:
            total = sum(map(len, truth))
            correct = sum(
                a == b
                for left, right in zip(grid, truth)
                for a, b in zip(left, right)
            )
            best_cell = max(best_cell, correct / total)
    return {
        "top2_exact": rank is not None and rank <= 2,
        "truth_in_pool": rank is not None,
        "truth_rank": rank,
        "generation_miss": rank is None,
        "selection_miss": rank is not None and rank > 2,
        "shape_accuracy": float(shape_hit),
        "cell_accuracy": best_cell,
        "unique_candidates": len(grids),
    }


def _run_truth_nll(
    config: dict[str, Any],
    work: Path,
    model_dir: Path,
    prepared: dict[str, Any],
    label_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    arms = config["arms"] + config["negative_controls"]
    script = Path(__file__).with_name("truth_nll_worker.py")
    runtime_config = work / "runtime-config.json"
    prepared_path = work / "manifests" / "prepared.json"
    for offset in range(0, len(arms), config["gpu_count"]):
        processes = []
        for gpu, arm in enumerate(arms[offset : offset + config["gpu_count"]]):
            log = (work / "arms" / arm / "truth_nll.log").open("a")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(script),
                "--arm", arm,
                "--config", str(runtime_config),
                "--prepared", str(prepared_path),
                "--model-dir", str(model_dir),
                "--work", str(work),
                "--label-dir", str(label_dir),
            ]
            processes.append((arm, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for arm, process, log in processes:
            code = process.wait()
            log.close()
            if code:
                raise RuntimeError(f"truth NLL arm {arm} failed with exit code {code}")
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        rows = _read_jsonl(work / "arms" / arm / "truth_nll.jsonl")
        if len(rows) != len({row["task_id"] for row in rows}):
            raise AssertionError(f"duplicate truth-NLL task records for {arm}")
        records[arm] = {row["task_id"]: row for row in rows}
        expected_ids = (
            set(prepared["manifest"]["negative_control_ids"])
            if arm in config["negative_controls"]
            else set(prepared["manifest"]["primary_ids"])
        )
        if set(records[arm]) != expected_ids:
            raise AssertionError(
                f"truth-NLL task alignment mismatch for {arm}: "
                f"expected {sorted(expected_ids)}, got {sorted(records[arm])}"
            )
        lifecycle = _read_jsonl(work / "arms" / arm / "truth_nll_lifecycle.jsonl")
        for task_id in records[arm]:
            events = [
                row["event"] for row in lifecycle if row["task_id"] == task_id
            ]
            if events[-3:] != ["training_completed", "label_opened", "nll_completed"]:
                raise AssertionError(f"invalid truth-NLL lifecycle {arm}/{task_id}: {events}")
            if records[arm][task_id]["optimizer_steps_after_label"] != 0:
                raise AssertionError("optimizer ran after held-out label open")
            if records[arm][task_id]["dfs_calls_after_label"] != 0:
                raise AssertionError("DFS ran after held-out label open")
    return records


def _finalize(
    config: dict[str, Any],
    work: Path,
    prepared: dict[str, Any],
    model_dir: Path,
) -> None:
    manifest = prepared["manifest"]
    evaluated_arms = config["arms"] + config["negative_controls"]
    primary_ids = manifest["primary_ids"]
    lifecycle = LabelLifecycle(work, evaluated_arms, primary_ids)
    for arm in evaluated_arms:
        for task_id in primary_ids:
            candidate = work / "arms" / arm / "candidates" / f"{task_id}.json"
            if arm in config["negative_controls"] and task_id not in manifest["negative_control_ids"]:
                # Negative controls are preregistered only on their subset; a
                # signed explicit skip artifact prevents accidental expansion.
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_text(
                    json.dumps({"task_id": task_id, "arm": arm, "not_evaluated": True})
                )
            lifecycle.record_candidates(arm, task_id, candidate)
    # This is the first read of held-out outputs in the process tree.
    input_root = Path("/kaggle/input")
    solutions_paths = sorted(
        path for path in input_root.rglob("*.json") if "training_solutions" in path.name
    )
    if len(solutions_paths) != 1:
        raise RuntimeError(f"expected one sealed training solution file: {solutions_paths}")
    all_labels = json.loads(solutions_paths[0].read_text())
    labels = {task_id: all_labels[task_id] for task_id in primary_ids}
    lifecycle.materialize_labels(labels)
    label_dir = work / "parent_evaluator_only" / "per_task_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for task_id, value in labels.items():
        (label_dir / f"{task_id}.json").write_text(
            json.dumps(value, separators=(",", ":"))
        )
    truth_nll = _run_truth_nll(config, work, model_dir, prepared, label_dir)

    arms: dict[str, Any] = {}
    task_exact: dict[str, dict[str, float]] = {}
    for arm in evaluated_arms:
        task_reports = []
        for task_id in primary_ids:
            path = work / "arms" / arm / "candidates" / f"{task_id}.json"
            payload = json.loads(path.read_text())
            if payload.get("not_evaluated"):
                continue
            nll_items = truth_nll[arm][task_id]["items"]
            item_groups = (
                payload.get("ranked_candidates"),
                labels[task_id],
                payload.get("item_diagnostics"),
                nll_items,
            )
            if not all(isinstance(group, list) and group for group in item_groups):
                raise AssertionError(f"empty item data for {arm}/{task_id}")
            counts = [len(group) for group in item_groups]
            if len(set(counts)) != 1:
                raise AssertionError(
                    f"item count mismatch for {arm}/{task_id}: "
                    f"candidates/truth/diagnostics/nll={counts}"
                )
            items = []
            for item_index in range(counts[0]):
                candidates = payload["ranked_candidates"][item_index]
                truth = labels[task_id][item_index]
                diagnostic = payload["item_diagnostics"][item_index]
                nll_diagnostic = nll_items[item_index]
                if not isinstance(candidates, list):
                    raise AssertionError(
                        f"candidate item index mismatch for {arm}/{task_id}/{item_index}"
                    )
                items.append({
                    "item_index": item_index,
                    **_score_item(candidates, truth),
                    **diagnostic,
                    **nll_diagnostic,
                })
            task_reports.append(
                {
                    "task_id": task_id,
                    "items": items,
                    "task_top2_exact": all(item["top2_exact"] for item in items),
                    "nodes_used": payload["nodes_used"],
                    "model_forward_tokens": payload["model_forward_tokens"],
                }
            )
        task_exact[arm] = {
            task["task_id"]: float(task["task_top2_exact"]) for task in task_reports
        }
        probe_path = work / "arms" / arm / "probe_metrics.json"
        arms[arm] = {
            "probe_metrics": json.loads(probe_path.read_text()),
            "tasks": task_reports,
            "aggregate": {
                "task_count": len(task_reports),
                "task_top2_exact": sum(task["task_top2_exact"] for task in task_reports),
                "item_top2_exact": sum(
                    item["top2_exact"] for task in task_reports for item in task["items"]
                ),
                "truth_in_pool": sum(
                    item["truth_in_pool"] for task in task_reports for item in task["items"]
                ),
            },
        }
    negative_ids = set(manifest["negative_control_ids"])
    negative_pairs = [
        ("shuffled_rowcol", "rowcol"),
        ("shuffled_component", "rowcol_component"),
    ]
    negative_contrasts = []
    causal_signals = []
    for shuffled, true_arm in negative_pairs:
        shuffled_ids = set(task_exact[shuffled])
        true_subset = {
            task_id: task_exact[true_arm][task_id] for task_id in negative_ids
        }
        if shuffled_ids != negative_ids or set(true_subset) != negative_ids:
            raise AssertionError(
                f"negative contrast misaligned for {true_arm}/{shuffled}"
            )
        metrics = {shuffled: task_exact[shuffled], true_arm: true_subset}
        contrast = paired_contrast(metrics, shuffled, true_arm)
        negative_contrasts.append(contrast)
        difference = contrast["mean_difference"]
        causal_signals.append(
            {
                "true_arm": true_arm,
                "shuffled_arm": shuffled,
                "n_tasks": contrast["n_tasks"],
                "true_minus_shuffled_mean": difference,
                "shuffled_matches_or_exceeds_true": difference is None or difference <= 0,
            }
        )
    report = {
        "schema_version": 1,
        "config_sha256": sha256_json(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "contamination_note": (
            "The upstream checkpoint used ARC-AGI-2 evaluation-derived synthesis; "
            "absolute scores are contaminated and paired contrasts isolate the "
            "metadata intervention only imperfectly."
        ),
        "arms": {arm: arms[arm] for arm in config["arms"]},
        "contrasts": [
            paired_contrast(task_exact, "text_control", "rowcol"),
            paired_contrast(task_exact, "rowcol", "rowcol_component"),
        ],
        "negative_control_contrasts": negative_contrasts,
        "causal_interpretation": {
            "supported": all(
                not signal["shuffled_matches_or_exceeds_true"]
                for signal in causal_signals
            ),
            "scope": "negative-control specificity signal only; not standalone proof of causality",
            "status": (
                "not_blocked_by_negative_controls"
                if all(
                    not signal["shuffled_matches_or_exceeds_true"]
                    for signal in causal_signals
                )
                else "blocked_by_negative_control"
            ),
            "rule": (
                "Blocked if either shuffled control matches or exceeds its "
                "corresponding true-metadata arm on the exact negative subset."
            ),
            "signals": causal_signals,
        },
        "negative_controls": {arm: arms[arm] for arm in config["negative_controls"]},
        "throughput_decision": prepared.get("throughput_decision"),
        "replays": {
            arm: json.loads((work / "arms" / arm / "replays.json").read_text())
            for arm in config["arms"]
        },
    }
    version = 1
    while (Path("/kaggle/working") / f"output_v{version:03d}").exists():
        version += 1
    output_root = Path("/kaggle/working") / f"output_v{version:03d}"
    report["validation"] = {
        "method": validate_final_report(
            report, Path(__file__).with_name("report_schema.json")
        )
    }
    staging_root = output_root.with_name(output_root.name + ".tmp")
    shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir()
    (staging_root / "experiment_021_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    shutil.copytree(work / "manifests", staging_root / "manifests")
    shutil.copytree(work / "candidate_hashes", staging_root / "candidate_hashes")
    shutil.copy2(work / "runtime-config.json", staging_root / "runtime-config.json")
    diagnostics_root = staging_root / "arm_diagnostics"
    for arm in evaluated_arms:
        source = work / "arms" / arm
        destination = diagnostics_root / arm
        destination.mkdir(parents=True)
        for name in (
            "worker.log", "truth_nll.log", "probe_metrics.json", "replays.json",
            "ARM_COMPLETE.sha256", "truth_nll.jsonl", "truth_nll_lifecycle.jsonl",
        ):
            path = source / name
            if path.is_file():
                shutil.copy2(path, destination / name)
        candidate_dir = source / "candidates"
        if candidate_dir.is_dir():
            shutil.copytree(candidate_dir, destination / "candidates")
    os.replace(staging_root, output_root)


def run_all_phases(
    config: dict[str, Any], work: Path, model_dir: Path, prepared: dict[str, Any]
) -> None:
    arms = config["arms"] + config["negative_controls"]
    pending = [
        arm
        for arm in arms
        if not (work / "arms" / arm / "ARM_COMPLETE.sha256").is_file()
    ]
    script = Path(__file__).with_name("gpu_arm_worker.py")
    for offset in range(0, len(pending), config["gpu_count"]):
        processes = []
        for gpu, arm in enumerate(pending[offset : offset + config["gpu_count"]]):
            arm_root = work / "arms" / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            log = (arm_root / "worker.log").open("a")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(script),
                "--arm",
                arm,
                "--config",
                str(work / "runtime-config.json"),
                "--prepared",
                str(work / "manifests" / "prepared.json"),
                "--model-dir",
                str(model_dir),
                "--work",
                str(work),
            ]
            processes.append((arm, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for arm, process, log in processes:
            code = process.wait()
            log.close()
            if code:
                raise RuntimeError(f"arm {arm} failed with exit code {code}")
    _finalize(config, work, prepared, model_dir)
