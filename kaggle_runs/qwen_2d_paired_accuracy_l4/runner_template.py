"""Offline two-arm paired accuracy runner for Experiment 023."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

SOURCE_ARCHIVE = "__SOURCE_ARCHIVE__"
CONFIG = json.loads(r"""__RUN_CONFIG__""")
INPUT = Path("/kaggle/input")
WORK = Path(f"/kaggle/working/experiment_{CONFIG['experiment']:03d}")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, path)


def find_unique(fragment: str) -> Path:
    matches = sorted(path for path in INPUT.rglob("*.json") if fragment in path.name)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {fragment!r} JSON, found {matches}")
    return matches[0]


def model_dir() -> Path:
    suffix = ("qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
    matches = sorted(
        path.parent
        for path in INPUT.rglob("config.json")
        if len(path.parent.parts) >= 4
        and (
            path.parent.parts[-4],
            path.parent.parts[-3].lower(),
            path.parent.parts[-2],
            path.parent.parts[-1],
        )
        == suffix
    )
    if len(matches) != 1:
        raise RuntimeError(f"exact checkpoint unavailable: {matches}")
    return matches[0]


def prepare(source: Path) -> dict:
    from arc_solver.qwen2d_experiment import (
        fixed_token_budget,
        make_arc_loo_rows,
        make_synthetic_rows,
        make_task_manifest,
    )
    from arc_solver.qwen2d_runtime import serialize_chatml_bounded

    prereg = json.loads((source / "preregistered_manifest.json").read_text())
    exclusions = json.loads((source / "exclusions.json").read_text())
    challenges = json.loads(find_unique("training_challenges").read_text())
    manifest = make_task_manifest(
        challenges,
        exclusions,
        seed=CONFIG["split_seed"],
        reserve_count=CONFIG["reserve_tasks"],
        # Recreate the already-fixed Experiment 021 selection before taking
        # the Experiment 023 prefix. The round-robin selector depends on the
        # requested count, so asking it directly for 2 or 8 chooses other IDs.
        primary_count=32,
        negative_count=12,
        replay_count=4,
    )
    expected = prereg["primary_ids"][: CONFIG["primary_tasks"]]
    if (
        CONFIG.get("primary_selection", "experiment021_prefix")
        == "experiment021_prefix"
        and manifest["primary_ids"][: CONFIG["primary_tasks"]] != expected
    ):
        raise AssertionError(
            f"fixed task mismatch: {manifest['primary_ids'][:CONFIG['primary_tasks']]} "
            f"!= {expected}"
        )
    manifest["primary_ids"] = expected
    manifest["negative_control_ids"] = []
    manifest["replay_ids"] = []
    # A fixed difficult set may come from outside the original reserved pool.
    # Its train pairs must not enter global training for either arm.
    target_ids = set(expected)
    manifest["global_train_ids"] = [
        task_id for task_id in manifest["global_train_ids"]
        if task_id not in target_ids
    ]
    manifest["reserved_ids"] = sorted(set(manifest["reserved_ids"]) | target_ids)
    arc_rows = make_arc_loo_rows(challenges, manifest["global_train_ids"])
    synthetic, _ = make_synthetic_rows(
        seeds_per_family=CONFIG["synthetic_seeds_per_family"],
        seed=CONFIG["seed"],
        heldout_modulus=CONFIG["synthetic_holdout_modulus"],
    )
    bounded = []
    for row in arc_rows + synthetic:
        _, _, dropped = serialize_chatml_bounded(
            row.get("train", []),
            row["query"],
            row["target"],
            max_length=CONFIG["global_max_sequence_length"],
        )
        item = dict(row)
        item["train"] = row.get("train", [])[dropped:]
        item["dropped_oldest_demos"] = dropped
        bounded.append(item)
    corpus, corpus_manifest = fixed_token_budget(
        bounded, CONFIG["global_loss_token_budget_per_arm"]
    )
    manifests = WORK / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    atomic_json(manifests / "prepared.json", {"manifest": manifest})
    atomic_json(manifests / "task_manifest.json", manifest)
    atomic_json(manifests / "corpus_manifest.json", corpus_manifest)
    atomic_json(manifests / "probe_rows.json", [])
    (manifests / "global_corpus.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus)
    )
    return manifest


def score_item(candidates, truth):
    grids = [candidate["grid"] for candidate in candidates]
    rank = grids.index(truth) + 1 if truth in grids else None
    return {
        "top2_exact": rank is not None and rank <= 2,
        "truth_in_pool": rank is not None,
        "truth_rank": rank,
        "generation_miss": rank is None,
        "selection_miss": rank is not None and rank > 2,
        "unique_candidates": len(grids),
    }


def run_workers(source: Path, manifest: dict, started: float) -> None:
    processes = []
    for gpu, arm in enumerate(CONFIG["arms"]):
        arm_root = WORK / "arms" / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        log = (arm_root / "worker.log").open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["ARC021_DEADLINE"] = str(started + CONFIG["hard_run_cap_seconds"])
        command = [
            sys.executable,
            str(source / "gpu_arm_worker.py"),
            "--arm", arm,
            "--config", str(WORK / "runtime-config.json"),
            "--prepared", str(WORK / "manifests" / "prepared.json"),
            "--model-dir", str(model_dir()),
            "--work", str(WORK),
        ]
        processes.append((arm, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    for arm, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            raise RuntimeError(f"arm {arm} failed with exit code {code}")
    for arm in CONFIG["arms"]:
        for task_id in manifest["primary_ids"]:
            if not (WORK / "arms" / arm / "candidates" / f"{task_id}.json").is_file():
                raise AssertionError(f"missing sealed candidates {arm}/{task_id}")


def evaluate(manifest: dict, started: float) -> None:
    # First solution-file read occurs only after every arm/task candidate exists.
    solutions = json.loads(find_unique("training_solutions").read_text())
    arms = {}
    task_exact = {}
    for arm in CONFIG["arms"]:
        tasks = []
        for task_id in manifest["primary_ids"]:
            payload = json.loads(
                (WORK / "arms" / arm / "candidates" / f"{task_id}.json").read_text()
            )
            labels = solutions[task_id]
            if len(payload["ranked_candidates"]) != len(labels):
                raise AssertionError(f"item mismatch {arm}/{task_id}")
            items = [
                {"item_index": index, **score_item(candidates, truth)}
                for index, (candidates, truth) in enumerate(
                    zip(payload["ranked_candidates"], labels)
                )
            ]
            tasks.append(
                {
                    "task_id": task_id,
                    "task_top2_exact": all(item["top2_exact"] for item in items),
                    "items": items,
                    "nodes_used": payload["nodes_used"],
                    "model_forward_tokens": payload["model_forward_tokens"],
                }
            )
        task_exact[arm] = {
            task["task_id"]: task["task_top2_exact"] for task in tasks
        }
        arms[arm] = {
            "tasks": tasks,
            "aggregate": {
                "task_count": len(tasks),
                "task_top2_exact": sum(task["task_top2_exact"] for task in tasks),
                "item_top2_exact": sum(
                    item["top2_exact"] for task in tasks for item in task["items"]
                ),
                "truth_in_pool": sum(
                    item["truth_in_pool"] for task in tasks for item in task["items"]
                ),
            },
        }
    control, metadata = CONFIG["arms"]
    wins = [
        task_id for task_id in manifest["primary_ids"]
        if task_exact[metadata][task_id] and not task_exact[control][task_id]
    ]
    losses = [
        task_id for task_id in manifest["primary_ids"]
        if task_exact[control][task_id] and not task_exact[metadata][task_id]
    ]
    report = {
        "experiment": CONFIG["experiment"],
        "profile": CONFIG["profile"],
        "elapsed_seconds": time.time() - started,
        "labels_opened_after_all_candidates": True,
        "primary_metric": "task_top2_exact",
        "arms": arms,
        "paired": {
            "metadata_minus_control": (
                arms[metadata]["aggregate"]["task_top2_exact"]
                - arms[control]["aggregate"]["task_top2_exact"]
            ),
            "wins": wins,
            "losses": losses,
            "ties": [
                task_id for task_id in manifest["primary_ids"]
                if task_exact[control][task_id] == task_exact[metadata][task_id]
            ],
            "directionally_better": len(wins) > len(losses),
            "low_power_warning": True,
        },
    }
    output = Path(
        f"/kaggle/working/experiment_{CONFIG['experiment']:03d}_output"
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    shutil.copytree(WORK / "manifests", output / "manifests")
    for arm in CONFIG["arms"]:
        destination = output / "arms" / arm
        destination.mkdir(parents=True)
        shutil.copy2(WORK / "arms" / arm / "worker.log", destination / "worker.log")
        shutil.copytree(WORK / "arms" / arm / "candidates", destination / "candidates")


def main() -> None:
    started = time.time()
    source = Path(f"/tmp/experiment_{CONFIG['experiment']:03d}_source")
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
        archive.extractall(source)
    sys.path.insert(0, str(source))
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    atomic_json(WORK / "runtime-config.json", CONFIG)
    from transformers import AutoTokenizer
    from arc_solver.qwen2d_metadata import assert_grids15_tokenizer

    assert_grids15_tokenizer(
        AutoTokenizer.from_pretrained(model_dir(), local_files_only=True)
    )
    manifest = prepare(source)
    run_workers(source, manifest, started)
    evaluate(manifest, started)


if __name__ == "__main__":
    main()
