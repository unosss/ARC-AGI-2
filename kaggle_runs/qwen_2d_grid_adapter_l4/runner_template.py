"""Monolithic Kaggle entrypoint for Experiment 021.

The archive is injected by build_kernel.py.  This entrypoint owns split/corpus
materialization and strict preflight. GPU phase code imports the same packaged
runtime functions; a phase is never marked complete until its atomic artifact
and hash exist.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


SOURCE_ARCHIVE = "__SOURCE_ARCHIVE__"
CONFIG = json.loads(r"""__RUN_CONFIG__""")
WORK = Path("/kaggle/working/experiment_021")
INPUT = Path("/kaggle/input")


def extract_sources() -> Path:
    root = Path("/tmp/experiment_021_source")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
        archive.extractall(root)
    return root


def find_unique(fragment: str) -> Path:
    matches = sorted(path for path in INPUT.rglob("*.json") if fragment in path.name)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {fragment!r} JSON, found {matches}")
    return matches[0]


def exact_model_dir() -> Path:
    from orchestration_safety import find_exact_model_dir

    return find_exact_model_dir(INPUT)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, path)


def prepare(source_root: Path) -> dict:
    from arc_solver.qwen2d_experiment import (
        collision_audit,
        fixed_token_budget,
        make_arc_loo_rows,
        make_synthetic_rows,
        make_task_manifest,
        sha256_json,
    )

    from orchestration_safety import validate_preregistered_manifest
    from arc_solver.qwen2d_runtime import serialize_chatml_bounded

    prereg = json.loads((source_root / "preregistered_manifest.json").read_text())
    validate_preregistered_manifest(prereg, CONFIG)
    challenges = json.loads(find_unique("training_challenges").read_text())
    excluded = prereg["reviewed_10_excluded"] + prereg["phase1_20_excluded"]
    primary_count = CONFIG["primary_tasks"]
    manifest = make_task_manifest(
        challenges,
        excluded,
        seed=CONFIG["split_seed"],
        reserve_count=CONFIG["reserve_tasks"],
        primary_count=primary_count,
        negative_count=CONFIG["negative_control_tasks"],
        replay_count=CONFIG["replay_tasks"],
    )
    minimum_manifest = make_task_manifest(
        challenges,
        excluded,
        seed=CONFIG["split_seed"],
        reserve_count=CONFIG["reserve_tasks"],
        primary_count=CONFIG["minimum_primary_tasks"],
        negative_count=CONFIG["negative_control_tasks"],
        replay_count=CONFIG["replay_tasks"],
    )
    for key in ("global_train_ids", "reserved_ids", "negative_control_ids", "replay_ids"):
        if minimum_manifest[key] != manifest[key]:
            raise AssertionError(f"preregistered 32/24 manifests disagree on {key}")
    arc_rows = make_arc_loo_rows(challenges, manifest["global_train_ids"])
    synthetic, probes = make_synthetic_rows(
        seeds_per_family=CONFIG["synthetic_seeds_per_family"],
        seed=CONFIG["seed"],
        heldout_modulus=CONFIG["synthetic_holdout_modulus"],
    )
    bounded_rows = []
    for row in arc_rows + synthetic:
        _, _, dropped = serialize_chatml_bounded(
            row.get("train", []),
            row["query"],
            row["target"],
            max_length=CONFIG["global_max_sequence_length"],
        )
        bounded = dict(row)
        bounded["train"] = row.get("train", [])[dropped:]
        bounded["dropped_oldest_demos"] = dropped
        bounded_rows.append(bounded)
    corpus, corpus_manifest = fixed_token_budget(
        bounded_rows, CONFIG["global_loss_token_budget_per_arm"]
    )
    audit = collision_audit(
        corpus,
        probes,
        manifest["global_train_ids"],
        manifest["reserved_ids"],
    )
    prepared = {
        "config_sha256": sha256_json(CONFIG),
        "manifest": manifest,
        "minimum_manifest": minimum_manifest,
        "corpus_manifest": corpus_manifest,
        "collision_audit": audit,
        "probe_count": len(probes),
        "arm_row_hashes": {
            arm: corpus_manifest["rows_sha256"]
            for arm in CONFIG["arms"] + CONFIG["negative_controls"]
        },
        "arm_loss_tokens": {
            arm: corpus_manifest["loss_tokens"]
            for arm in CONFIG["arms"] + CONFIG["negative_controls"]
        },
    }
    atomic_json(WORK / "manifests" / "prepared.json", prepared)
    atomic_json(WORK / "manifests" / "task_manifest.json", manifest)
    atomic_json(WORK / "manifests" / "corpus_manifest.json", corpus_manifest)
    # JSONL is deterministic and shared read-only by every arm.
    corpus_path = WORK / "manifests" / "global_corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus))
    atomic_json(WORK / "manifests" / "probe_rows.json", probes)
    return prepared


def preflight(source_root: Path, prepared: dict) -> None:
    import torch
    from transformers import AutoTokenizer
    from arc_solver.qwen2d_metadata import assert_grids15_tokenizer
    from arc_solver.qwen2d_experiment import sha256_json
    from orchestration_safety import validate_generated_manifest

    if CONFIG["profile"] == "full_preregistered" and torch.cuda.device_count() != 4:
        raise RuntimeError("full profile requires exactly four visible L4 GPUs")
    model_dir = exact_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    assert_grids15_tokenizer(tokenizer)
    validate_generated_manifest(prepared["manifest"], CONFIG)
    validate_generated_manifest(prepared["minimum_manifest"], CONFIG)
    for manifest in (prepared["manifest"], prepared["minimum_manifest"]):
        unhashed = dict(manifest)
        expected = unhashed.pop("manifest_sha256")
        if sha256_json(unhashed) != expected:
            raise AssertionError("generated manifest hash mismatch")
    if len(set(prepared["arm_row_hashes"].values())) != 1:
        raise AssertionError("arm corpus rows differ")
    if len(set(prepared["arm_loss_tokens"].values())) != 1:
        raise AssertionError("arm token budgets differ")


def calibrate(source_root: Path, model_dir: Path, prepared: dict) -> dict:
    if "throughput_calibration" not in CONFIG:
        return prepared
    command = [
        sys.executable,
        str(source_root / "calibrate.py"),
        "--config", str(WORK / "runtime-config.json"),
        "--prepared", str(WORK / "manifests" / "prepared.json"),
        "--model-dir", str(model_dir),
        "--work", str(WORK),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(command, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"throughput calibration failed: {result.returncode}")
    report = json.loads(
        (WORK / "manifests" / "throughput_calibration.json").read_text()
    )
    deadline = float(os.environ["ARC021_DEADLINE"])
    remaining = deadline - time.time()
    if report["projected_minimum_seconds"] > remaining:
        raise RuntimeError(
            "calibration hard abort: preregistered 24-task workload "
            f"projects {report['projected_minimum_seconds']:.1f}s with "
            f"only {remaining:.1f}s remaining"
        )
    full_limit = min(
        CONFIG["throughput_calibration"]["promote_to_32_if_projected_seconds_lte"],
        remaining,
    )
    if report["projected_full_seconds"] > full_limit:
        prepared["manifest"] = prepared["minimum_manifest"]
        prepared["throughput_decision"] = {
            "selected_primary_count": CONFIG["minimum_primary_tasks"],
            "reason": "projected_full_workload_exceeds_gate",
            "full_limit_seconds": full_limit,
            "negative_control_ids_preserved": True,
            "replay_ids_preserved": True,
        }
        atomic_json(
            WORK / "manifests" / "task_manifest.json", prepared["manifest"]
        )
        atomic_json(WORK / "manifests" / "prepared.json", prepared)
    else:
        prepared["throughput_decision"] = {
            "selected_primary_count": CONFIG["primary_tasks"],
            "reason": "projected_full_workload_within_gate",
            "full_limit_seconds": full_limit,
            "negative_control_ids_preserved": True,
            "replay_ids_preserved": True,
        }
        atomic_json(WORK / "manifests" / "prepared.json", prepared)
    return prepared


def main() -> None:
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    if "throughput_calibration" in CONFIG:
        os.environ.setdefault(
            "ARC021_DEADLINE",
            str(time.time() + CONFIG["throughput_calibration"]["hard_run_cap_seconds"]),
        )
    WORK.mkdir(parents=True, exist_ok=True)
    source_root = extract_sources()
    sys.path.insert(0, str(source_root))
    from orchestration_safety import (
        validate_report_schema_document,
        validate_runtime_config,
    )

    validate_runtime_config(CONFIG)
    validate_report_schema_document(source_root / "report_schema.json")
    atomic_json(WORK / "runtime-config.json", CONFIG)
    prepared = prepare(source_root)
    preflight(source_root, prepared)
    prepared = calibrate(source_root, exact_model_dir(), prepared)
    # Imported only after preflight so an incompatible checkpoint cannot start
    # an expensive phase or create a misleading result file.
    from experiment_worker import run_all_phases

    run_all_phases(CONFIG, WORK, exact_model_dir(), prepared)


if __name__ == "__main__":
    main()
