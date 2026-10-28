"""Restart-safe, sequential RunPod runner for preregistered Experiment 024."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def validate_config(config: dict) -> None:
    expected = {
        "global_loss_token_budget_per_arm": 1_500_000,
        "arms": ["text_control", "rowcol_component"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise AssertionError(f"{key} changed: {config.get(key)!r} != {value!r}")
    local = config["task_local"]
    required_local = {
        "steps": 128,
        "train_color_permutations": 16,
        "train_geometries": 8,
        "decode_color_permutations": 2,
        "decode_geometries": 8,
        "rescore_views": 8,
    }
    for key, value in required_local.items():
        if local.get(key) != value:
            raise AssertionError(f"task_local.{key} changed")
    if local.get("merge_for_inference", False) is not False:
        raise AssertionError("accepted path requires merge_for_inference=false/absent")
    if int(local.get("rescore_batch_size", 1)) != 1:
        raise AssertionError("accepted path requires rescore_batch_size=1/absent")
    dfs = config["dfs"]
    required_dfs = {
        "batch_size": 4,
        "max_nodes_per_view": 4096,
        "max_model_forward_tokens_per_view": 12288,
        "max_candidates_per_view": 128,
        "max_new_tokens": 931,
        "emergency_seconds_per_view": 900,
        "min_prob": 0.2,
    }
    for key, value in required_dfs.items():
        if dfs.get(key) != value:
            raise AssertionError(f"dfs.{key} changed")


def prepare(config: dict, repo: Path, input_dir: Path, work: Path) -> dict:
    sys.path.insert(0, str(repo))
    from arc_solver.qwen2d_experiment import (
        fixed_token_budget,
        make_arc_loo_rows,
        make_synthetic_rows,
        make_task_manifest,
    )
    from arc_solver.qwen2d_runtime import serialize_chatml_bounded

    prereg = read_json(
        repo / "experiments/024_qwen_2d_hard_paired/preregistered_manifest.json"
    )
    exclusions = read_json(
        repo / "kaggle_runs/qwen_2d_paired_accuracy_l4/exclusions.json"
    )
    challenges = read_json(input_dir / "arc-agi_training_challenges.json")
    expected = prereg["primary_ids"]
    if set(expected) - challenges.keys():
        raise AssertionError("merged challenge source lacks preregistered IDs")
    manifest = make_task_manifest(
        challenges,
        exclusions,
        seed=config["split_seed"],
        reserve_count=config["reserve_tasks"],
        primary_count=32,
        negative_count=12,
        replay_count=4,
    )
    manifest["primary_ids"] = expected
    manifest["negative_control_ids"] = []
    manifest["replay_ids"] = []
    targets = set(expected)
    manifest["global_train_ids"] = [
        task_id for task_id in manifest["global_train_ids"] if task_id not in targets
    ]
    manifest["reserved_ids"] = sorted(set(manifest["reserved_ids"]) | targets)
    if targets & set(manifest["global_train_ids"]):
        raise AssertionError("target train pairs leaked into global training")

    arc_rows = make_arc_loo_rows(challenges, manifest["global_train_ids"])
    synthetic, _ = make_synthetic_rows(
        seeds_per_family=config["synthetic_seeds_per_family"],
        seed=config["seed"],
        heldout_modulus=config["synthetic_holdout_modulus"],
    )
    bounded = []
    for row in arc_rows + synthetic:
        _, _, dropped = serialize_chatml_bounded(
            row.get("train", []),
            row["query"],
            row["target"],
            max_length=config["global_max_sequence_length"],
        )
        item = dict(row)
        item["train"] = row.get("train", [])[dropped:]
        item["dropped_oldest_demos"] = dropped
        bounded.append(item)
    corpus, corpus_manifest = fixed_token_budget(
        bounded, config["global_loss_token_budget_per_arm"]
    )
    manifests = work / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    atomic_json(manifests / "prepared.json", {"manifest": manifest})
    atomic_json(manifests / "task_manifest.json", manifest)
    atomic_json(manifests / "corpus_manifest.json", corpus_manifest)
    atomic_json(manifests / "probe_rows.json", [])
    (manifests / "global_corpus.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus)
    )
    actual = sum(int(row["loss_token_limit"]) for row in corpus)
    if actual != 1_500_000:
        raise AssertionError(f"global corpus budget is {actual}, expected 1500000")
    return manifest


def run_workers(
    config: dict,
    repo: Path,
    model_dir: Path,
    work: Path,
    wall_cap_seconds: int,
) -> None:
    worker = repo / "kaggle_runs/qwen_2d_grid_adapter_l4/gpu_arm_worker.py"
    for arm in config["arms"]:
        arm_root = work / "arms" / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        if (arm_root / "ARM_COMPLETE.sha256").is_file():
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["ARC021_DEADLINE"] = str(time.time() + wall_cap_seconds)
        env["PYTHONPATH"] = str(repo)
        command = [
            sys.executable,
            str(worker),
            "--arm",
            arm,
            "--config",
            str(work / "runtime-config.json"),
            "--prepared",
            str(work / "manifests/prepared.json"),
            "--model-dir",
            str(model_dir),
            "--work",
            str(work),
        ]
        with (arm_root / "worker.log").open("a") as log:
            log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
            log.flush()
            code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f"arm {arm} failed with exit code {code}; restart is safe")


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


def evaluate(
    config: dict, input_dir: Path, work: Path, output: Path, started: float
) -> None:
    manifest = read_json(work / "manifests/task_manifest.json")
    for arm in config["arms"]:
        for task_id in manifest["primary_ids"]:
            candidate = work / "arms" / arm / "candidates" / f"{task_id}.json"
            if not candidate.is_file():
                raise AssertionError(f"missing sealed candidates {arm}/{task_id}")

    # The sealed labels are first loaded only after every candidate file exists.
    solutions = read_json(input_dir / "arc-agi_training_solutions.json")
    arms, task_exact = {}, {}
    for arm in config["arms"]:
        tasks = []
        for task_id in manifest["primary_ids"]:
            payload = read_json(
                work / "arms" / arm / "candidates" / f"{task_id}.json"
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
                "task_top2_exact": sum(t["task_top2_exact"] for t in tasks),
                "item_top2_exact": sum(
                    item["top2_exact"] for task in tasks for item in task["items"]
                ),
                "truth_in_pool": sum(
                    item["truth_in_pool"] for task in tasks for item in task["items"]
                ),
            },
        }
    control, metadata = config["arms"]
    wins = [
        task_id
        for task_id in manifest["primary_ids"]
        if task_exact[metadata][task_id] and not task_exact[control][task_id]
    ]
    losses = [
        task_id
        for task_id in manifest["primary_ids"]
        if task_exact[control][task_id] and not task_exact[metadata][task_id]
    ]
    report = {
        "experiment": 24,
        "profile": config["profile"],
        "elapsed_seconds": time.time() - started,
        "labels_opened_after_all_candidates": True,
        "primary_metric": "task_top2_exact",
        "interpretation": "hard_set_directional_not_population_accuracy",
        "arms": arms,
        "paired": {
            "metadata_minus_control": (
                arms[metadata]["aggregate"]["task_top2_exact"]
                - arms[control]["aggregate"]["task_top2_exact"]
            ),
            "wins": wins,
            "losses": losses,
            "ties": [
                task_id
                for task_id in manifest["primary_ids"]
                if task_exact[control][task_id] == task_exact[metadata][task_id]
            ],
            "directionally_better": len(wins) > len(losses),
            "low_power_warning": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    for name in ("manifests", "arms"):
        shutil.copytree(work / name, output / name, dirs_exist_ok=True)
    shutil.copy2(input_dir / "merged_data_manifest.json", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=int, default=604_800)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = read_json(
        args.repo / "kaggle_runs/qwen_2d_hard_paired_l4/full-config.json"
    )
    validate_config(config)
    if args.wall_cap_seconds < config["hard_run_cap_seconds"]:
        raise AssertionError("RunPod operational cap may not be below production cap")
    if not args.model.joinpath("config.json").is_file():
        raise FileNotFoundError(args.model / "config.json")
    sys.path.insert(0, str(args.repo))
    from transformers import AutoTokenizer
    from arc_solver.qwen2d_metadata import assert_grids15_tokenizer

    assert_grids15_tokenizer(
        AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    )
    merged_manifest = read_json(args.input / "merged_data_manifest.json")
    expected = read_json(
        args.repo / "experiments/024_qwen_2d_hard_paired/preregistered_manifest.json"
    )["primary_ids"]
    if merged_manifest["primary_ids"] != expected:
        raise AssertionError("merged source IDs differ from preregistration")
    if args.validate_only:
        print("RunPod configuration and merged data validation passed")
        return

    args.work.mkdir(parents=True, exist_ok=True)
    started_file = args.work / "RUN_STARTED.json"
    if started_file.is_file():
        started = float(read_json(started_file)["epoch"])
    else:
        started = time.time()
        atomic_json(started_file, {"epoch": started})
    atomic_json(args.work / "runtime-config.json", config)
    if not (args.work / "manifests/prepared.json").is_file():
        prepare(config, args.repo, args.input, args.work)
    manifest = read_json(args.work / "manifests/task_manifest.json")
    if manifest["primary_ids"] != expected:
        raise AssertionError("prepared IDs differ from preregistration")

    # The unchanged worker discovers data under /kaggle/input. Use one dedicated
    # compatibility symlink rather than changing solver/worker behavior.
    kaggle_input = Path("/kaggle/input")
    kaggle_input.parent.mkdir(parents=True, exist_ok=True)
    if kaggle_input.is_symlink() and kaggle_input.resolve() != args.input.resolve():
        raise RuntimeError(f"{kaggle_input} points elsewhere")
    if kaggle_input.exists() and not kaggle_input.is_symlink():
        raise RuntimeError(f"{kaggle_input} exists and is not the experiment symlink")
    if not kaggle_input.exists():
        kaggle_input.symlink_to(args.input.resolve(), target_is_directory=True)

    run_workers(config, args.repo, args.model, args.work, args.wall_cap_seconds)
    evaluate(config, args.input, args.work, args.output, started)


if __name__ == "__main__":
    main()
