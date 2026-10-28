"""Leakage-safe paired Base-vs-RSLoRA leave-one-out runner."""

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
RUN_CONFIG = json.loads(r"""__RUN_CONFIG__""")
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/experiment_016")


def extract_sources() -> Path:
    root = Path("/tmp/experiment_016_source")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
        archive.extractall(root)
    return root


def find_arc_file(fragment: str) -> Path:
    matches = sorted(path for path in INPUT_ROOT.rglob("*.json") if fragment in path.name)
    if not matches:
        raise FileNotFoundError(f"ARC file containing {fragment!r} not found")
    return matches[0]


def find_package_root(package: str) -> Path:
    roots = [INPUT_ROOT, Path("/kaggle/usr/lib")]
    matches = sorted(
        match
        for root in roots
        if root.exists()
        for match in root.rglob(f"{package}/__init__.py")
    )
    if not matches:
        raise FileNotFoundError(f"offline package {package!r} not found")
    return matches[0].parent.parent


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def run_phase(
    phase: str,
    source_root: Path,
    dependency_root: Path,
    public_path: Path,
    *,
    label_dir: Path | None = None,
) -> None:
    phase_root = WORK_ROOT / phase
    output_dir = phase_root / "candidates"
    telemetry = phase_root / "telemetry.jsonl"
    log_path = phase_root / f"{phase}.log"
    marker_dir = phase_root / "markers"
    phase_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(source_root), str(dependency_root))),
            "ARC_LOO_MODE": phase,
            "ARC_OUT_DIR": str(output_dir),
            "ARC_TELEMETRY_FILE": str(telemetry),
            "ARC_N_TRAIN_AUG": str(RUN_CONFIG["n_train_aug"]),
            "ARC_N_EVAL_AUG": str(RUN_CONFIG["n_eval_aug"]),
            "ARC_TASK_CAP": str(RUN_CONFIG["decode_cap_seconds"]),
            "ARC_DFS_WINDOW": str(RUN_CONFIG["dfs_window_seconds"]),
            "ARC_DECODE_BATCH": str(RUN_CONFIG["decode_batch"]),
            "ARC_MIN_PROB": str(RUN_CONFIG["min_prob"]),
            "ARC_EPOCHS": str(RUN_CONFIG["epochs"]),
            "ARC_LR": str(RUN_CONFIG["learning_rate"]),
            "ARC_LORA_SEED": str(RUN_CONFIG["lora_seed"]),
            "ARC_TRAIN_AUG_SEED": str(RUN_CONFIG["train_aug_seed"]),
            "ARC_EVAL_AUG_SEED": str(RUN_CONFIG["eval_aug_seed"]),
            "ARC_SCORE_SEED_OFFSET": str(RUN_CONFIG["score_seed_offset"]),
            "ARC_ALL_KEYS": "1",
            "ARC_DETERMINISTIC": "1",
            "UNSLOTH_DISABLE_STATISTICS": "1",
            "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
            "OMP_NUM_THREADS": "12",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if label_dir is not None:
        env.update(
            {
                "ARC_LABEL_DIR": str(label_dir),
                "ARC_NLL_FILE": str(phase_root / "nll.jsonl"),
                "ARC_LIFECYCLE_FILE": str(phase_root / "lifecycle.jsonl"),
            }
        )
    deadline = RUN_CONFIG["phase_deadline_seconds"][phase]
    command = [
        sys.executable,
        str(source_root / "starter.py"),
        "--end-time",
        str(time.time() + deadline),
        "--order",
        RUN_CONFIG["order"],
        "--test-path",
        str(public_path),
        "--marker-dir",
        str(marker_dir),
        "--nprocs",
        str(RUN_CONFIG["nprocs"]),
    ]
    started = time.time()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    (phase_root / "phase_timing.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "seconds": round(time.time() - started, 3),
                "deadline_seconds": deadline,
                "return_code": return_code,
            },
            separators=(",", ":"),
        )
    )
    if return_code:
        raise RuntimeError(f"{phase} failed with exit code {return_code}")


def finalize_arm(arm, public_path, public_challenges, labels):
    from arc_loader import ArcDataset
    from arc_decoder import ArcDecoder, score_full_probmul_3, score_kgmon
    from arc_solver.nvarc_eval import analyze_candidate_oracles, finalize_top2

    output_dir = WORK_ROOT / f"{arm}_candidates" / "candidates"
    dataset = ArcDataset.from_file(str(public_path))
    decoder = ArcDecoder(dataset.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results(str(output_dir))
    selected = decoder.run_selection_algo(score_kgmon)
    submission, fallbacks = finalize_top2(public_challenges, selected)
    oracle = analyze_candidate_oracles(
        decoder.decoded_results,
        labels,
        {
            "score_kgmon": score_kgmon,
            "score_full_probmul_3": score_full_probmul_3,
        },
    )
    (WORK_ROOT / f"{arm}_top2.json").write_text(
        json.dumps(submission, separators=(",", ":"))
    )
    return submission, oracle, fallbacks, decoder.candidate_stats()


def main() -> None:
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError("Experiment 016 is an evaluation-only LOO study")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    source_root = extract_sources()
    dependency_root = find_package_root("unsloth")
    sys.path.insert(0, str(source_root))
    sys.path.insert(1, str(dependency_root))
    from arc_solver.nvarc_loo import (
        aggregate_paired_report,
        build_loo_solver_challenges,
        load_loo_labels,
        write_per_fold_labels,
    )

    challenge_source = find_arc_file("evaluation_challenges")
    challenges = json.loads(challenge_source.read_text())
    public, metadata = build_loo_solver_challenges(
        challenges,
        RUN_CONFIG["task_ids"],
        minimum_demos=RUN_CONFIG["minimum_demos"],
    )
    if metadata["fold_count"] != RUN_CONFIG["expected_fold_count"]:
        raise AssertionError(
            f"expected {RUN_CONFIG['expected_fold_count']} folds, "
            f"got {metadata['fold_count']}"
        )
    solver_dir = WORK_ROOT / "solver_input"
    solver_dir.mkdir(parents=True, exist_ok=True)
    public_path = solver_dir / "loo_challenges.json"
    public_path.write_text(json.dumps(public, separators=(",", ":")))

    # Candidate workers receive only public_path. No label path or label file
    # exists until both arms and all their workers have exited.
    run_phase("base_candidates", source_root, dependency_root, public_path)
    run_phase("rslora_candidates", source_root, dependency_root, public_path)

    expected_folds = {fold["fold_key"] for fold in metadata["folds"]}
    telemetry = {
        arm: {
            record["fold_key"]: record
            for record in read_jsonl(
                WORK_ROOT / f"{arm}_candidates" / "telemetry.jsonl"
            )
        }
        for arm in ("base", "rslora")
    }
    for arm, records in telemetry.items():
        if set(records) != expected_folds:
            raise AssertionError(
                f"{arm} telemetry incomplete: "
                f"missing={sorted(expected_folds - set(records))}"
            )
        if any(
            record["decode_cap_seconds"] != RUN_CONFIG["decode_cap_seconds"]
            for record in records.values()
        ):
            raise AssertionError(f"{arm} used an unequal decode cap")
    if any(record["training_steps"] != 0 for record in telemetry["base"].values()):
        raise AssertionError("Base arm unexpectedly performed optimizer steps")

    # Parent-only candidate finalization begins after both subprocess waits.
    labels = load_loo_labels(challenges, metadata)
    parent_dir = WORK_ROOT / "parent_finalizer_only"
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "loo_metadata.json").write_text(
        json.dumps(metadata, separators=(",", ":"))
    )
    (parent_dir / "loo_labels.json").write_text(
        json.dumps(labels, separators=(",", ":"))
    )
    label_dir = parent_dir / "per_fold_labels"
    write_per_fold_labels(labels, label_dir)

    submissions = {}
    oracles = {}
    arm_details = {}
    for arm in ("base", "rslora"):
        submission, oracle, fallbacks, candidate_stats = finalize_arm(
            arm, public_path, public, labels
        )
        submissions[arm] = submission
        oracles[arm] = oracle
        arm_details[arm] = {
            "fallbacks": fallbacks,
            "candidate_stats": candidate_stats,
            "oracle_aggregate": oracle["aggregate"],
        }

    # NLL is a separate worker phase. Each worker receives per-fold label files
    # and opens only its own label after that fold's RSLoRA training completes.
    run_phase(
        "nll",
        source_root,
        dependency_root,
        public_path,
        label_dir=label_dir,
    )
    nll_records = read_jsonl(WORK_ROOT / "nll" / "nll.jsonl")
    if {record["fold_key"] for record in nll_records} != expected_folds:
        raise AssertionError("NLL records are incomplete")
    lifecycle = read_jsonl(WORK_ROOT / "nll" / "lifecycle.jsonl")
    for fold_key in expected_folds:
        events = [
            record["event"]
            for record in lifecycle
            if record["fold_key"] == fold_key
        ]
        if events != ["training_completed", "label_opened", "nll_completed"]:
            raise AssertionError(f"invalid NLL lifecycle for {fold_key}: {events}")
    paired = aggregate_paired_report(
        metadata, oracles, submissions, telemetry, nll_records
    )
    report = {
        "mode": RUN_CONFIG["mode"],
        "config": RUN_CONFIG,
        "architecture": {
            "candidate_phases": ["base_candidates", "rslora_candidates"],
            "candidate_labels_available": False,
            "decode_cap_excludes": ["model_load", "training"],
            "decode_cap_equal_seconds": RUN_CONFIG["decode_cap_seconds"],
            "nll_phase_separate": True,
            "nll_order": [
                "train",
                "open_fold_label",
                "base_teacher_forcing",
                "trained_rslora_teacher_forcing",
            ],
            "post_label_optimizer_steps": 0,
            "post_label_dfs_calls": 0,
        },
        "arms": arm_details,
        **paired,
    }
    report_path = Path("/kaggle/working") / RUN_CONFIG["report_filename"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (WORK_ROOT / RUN_CONFIG["report_filename"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
