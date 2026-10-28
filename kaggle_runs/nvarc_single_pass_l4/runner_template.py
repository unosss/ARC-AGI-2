"""Standalone leakage-safe Pass-A NVARC runner for Experiment 015."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile


SOURCE_ARCHIVE = "__SOURCE_ARCHIVE__"
RUN_CONFIG = json.loads(r"""__RUN_CONFIG__""")
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/experiment_015")


def extract_sources() -> Path:
    source_root = Path("/tmp/experiment_015_source")
    source_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
        archive.extractall(source_root)
    return source_root


def find_arc_file(fragment: str) -> Path:
    matches = sorted(path for path in INPUT_ROOT.rglob("*.json") if fragment in path.name)
    if not matches:
        raise FileNotFoundError(f"ARC file containing {fragment!r} not found")
    return matches[0]


def find_package_root(package: str) -> Path:
    search_roots = [INPUT_ROOT, Path("/kaggle/usr/lib")]
    matches = sorted(
        match
        for root in search_roots
        if root.exists()
        for match in root.rglob(f"{package}/__init__.py")
    )
    if not matches:
        package_matches = sorted(
            str(path)
            for root in search_roots
            if root.exists()
            for path in root.rglob("*")
            if package.lower() in path.name.lower()
        )
        mounts = {
            str(root): sorted(str(path) for path in root.iterdir())
            for root in search_roots
            if root.exists()
        }
        raise FileNotFoundError(
            f"offline package {package!r} not found; "
            f"mounts={mounts}; package_matches={package_matches[:50]}"
        )
    return matches[0].parent.parent


class MemoryMonitor:
    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(1):
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                values = [int(line) for line in output.splitlines() if line.strip()]
                self.peak_mib = max([self.peak_mib, *values])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=3)


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


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    source_root = extract_sources()
    dependency_root = find_package_root("unsloth")
    sys.path.insert(0, str(source_root))
    sys.path.insert(1, str(dependency_root))
    from arc_loader import ArcDataset
    from arc_decoder import ArcDecoder, score_full_probmul_3, score_kgmon
    from arc_solver.nvarc_eval import (
        analyze_candidate_oracles,
        finalize_top2,
        load_finalizer_labels,
        parse_training_stats,
        write_solver_challenges,
    )
    from arc_solver.ttt_eval import score_condition

    rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    challenge_fragment = "test_challenges" if rerun else "evaluation_challenges"
    challenge_source = find_arc_file(challenge_fragment)
    challenges = json.loads(challenge_source.read_text())
    task_ids = (
        sorted(challenges)
        if rerun
        else [task for task in RUN_CONFIG["task_ids"] if task in challenges]
    )
    if not task_ids:
        raise RuntimeError("no configured NVARC tasks were found")

    # This directory is the only data directory passed to the solver subprocess.
    # It is created before any solution file is opened or any labels are written.
    solver_data_dir = WORK_ROOT / "solver_input"
    public_path = write_solver_challenges(
        challenge_source, task_ids, solver_data_dir
    )
    public_challenges = json.loads(public_path.read_text())
    output_dir = WORK_ROOT / "pass_a_candidates"
    telemetry_path = WORK_ROOT / "pass_a_telemetry.jsonl"
    log_path = WORK_ROOT / "pass_a.log"
    marker_dir = WORK_ROOT / "markers"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(source_root), str(dependency_root))),
            "ARC_OUT_DIR": str(output_dir),
            "ARC_TELEMETRY_FILE": str(telemetry_path),
            "ARC_PASS_NAME": "pass_a",
            "ARC_LORA_SEED": str(RUN_CONFIG["lora_seed"]),
            "ARC_TRAIN_AUG_SEED": str(RUN_CONFIG["train_aug_seed"]),
            "ARC_N_TRAIN_AUG": str(RUN_CONFIG["n_train_aug"]),
            "ARC_EVAL_AUG_SEED": str(RUN_CONFIG["eval_aug_seed"]),
            "ARC_N_EVAL_AUG": str(RUN_CONFIG["n_eval_aug"]),
            "ARC_TASK_CAP": str(RUN_CONFIG["task_cap_seconds"]),
            "ARC_SCORE_SEED_OFFSET": str(RUN_CONFIG["score_seed_offset"]),
            "ARC_DEBUG_KEYS": ",".join(task_ids),
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
    budget = (
        RUN_CONFIG["submission_deadline_seconds"]
        if rerun
        else RUN_CONFIG["smoke_deadline_seconds"]
    )
    started = time.time()
    end_time = started + budget
    command = [
        sys.executable,
        str(source_root / "starter.py"),
        "--end-time",
        str(end_time),
        "--order",
        RUN_CONFIG["order"],
        "--test-path",
        str(public_path),
        "--marker-dir",
        str(marker_dir),
        "--nprocs",
        str(RUN_CONFIG["nprocs"]),
    ]
    with log_path.open("w") as log, MemoryMonitor() as monitor:
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
    elapsed = time.time() - started
    if return_code:
        raise RuntimeError(f"Pass A failed with exit code {return_code}")

    # Parent-only finalization starts after every solver worker has exited.
    dataset = ArcDataset.from_file(str(public_path))
    decoder = ArcDecoder(dataset.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results(str(output_dir))
    candidate_stats = decoder.candidate_stats()
    selected = decoder.run_selection_algo(score_kgmon)
    submission, fallback_counts = finalize_top2(public_challenges, selected)
    submission_path = Path("/kaggle/working/submission.json")
    submission_path.write_text(json.dumps(submission, separators=(",", ":")))
    (WORK_ROOT / "submission.json").write_text(
        json.dumps(submission, separators=(",", ":"))
    )

    expected_items = {
        task: len(public_challenges[task]["test"]) for task in task_ids
    }
    actual_items = {task: len(submission.get(task, [])) for task in task_ids}
    if actual_items != expected_items:
        raise AssertionError(
            f"submission item counts differ: expected={expected_items}, actual={actual_items}"
        )

    task_telemetry = {record["task"]: record for record in read_jsonl(telemetry_path)}
    training_stats = parse_training_stats(log_path.read_text())
    for task in task_ids:
        record = task_telemetry.setdefault(task, {"task": task})
        record.update(training_stats.get(task, {}))
        output_stats = {
            key.rsplit("_", 1)[-1]: value
            for key, value in candidate_stats.items()
            if key.startswith(task + "_")
        }
        record["outputs"] = output_stats
        record["unique_candidates"] = sum(
            value["unique"] for value in output_stats.values()
        )
        record["candidate_samples"] = sum(
            value["samples"] for value in output_stats.values()
        )

    report = {
        "mode": "competition_rerun" if rerun else RUN_CONFIG["mode"],
        "task_ids": task_ids,
        "config": RUN_CONFIG,
        "timing": {
            "total_seconds": round(elapsed, 3),
            "deadline_seconds": budget,
        },
        "peak_gpu_memory_mib": monitor.peak_mib,
        "fallbacks": fallback_counts,
        "tasks": [task_telemetry[task] for task in task_ids],
        "submission_item_counts": actual_items,
    }
    if not rerun:
        # Solutions and labels are first opened here, after the subprocess exits.
        solution_source = find_arc_file("evaluation_solutions")
        labels = load_finalizer_labels(
            challenge_source,
            solution_source,
            task_ids,
            WORK_ROOT / "parent_finalizer_only",
        )
        label_items = {task: len(labels[task]) for task in task_ids}
        if actual_items != label_items:
            raise AssertionError(
                f"submission/label item counts differ: "
                f"labels={label_items}, submission={actual_items}"
            )
        report["label_item_counts"] = label_items
        report["score_condition"] = score_condition(submission, labels)
        report["candidate_diagnostics"] = analyze_candidate_oracles(
            decoder.decoded_results,
            labels,
            {
                "score_kgmon": score_kgmon,
                "score_full_probmul_3": score_full_probmul_3,
            },
        )
    report_filename = RUN_CONFIG.get("report_filename", "smoke_report.json")
    (Path("/kaggle/working") / report_filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (WORK_ROOT / report_filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
