"""Standalone Kaggle runner for Experiment 013 Phase 1."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile
from io import BytesIO


SOURCE_ARCHIVE = "__SOURCE_ARCHIVE__"
RUN_CONFIG = json.loads(r"""__RUN_CONFIG__""")
TASK_IDS = json.loads(r"""__TASK_IDS__""")
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/experiment_013_phase1")


def extract_sources() -> Path:
    source_root = Path("/tmp/experiment_013_phase1_source")
    source_root.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(SOURCE_ARCHIVE)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        archive.extractall(source_root)
    return source_root


def find_arc_files() -> tuple[Path, Path]:
    challenges = sorted(
        path
        for path in INPUT_ROOT.rglob("*.json")
        if "training_challenges" in path.name
    )
    solutions = sorted(
        path
        for path in INPUT_ROOT.rglob("*.json")
        if "training_solutions" in path.name
    )
    if not challenges or not solutions:
        raise FileNotFoundError("ARC-AGI-2 training challenges/solutions not found")
    return challenges[0], solutions[0]


def find_model_dir() -> Path:
    candidates = []
    for config_path in INPUT_ROOT.rglob("config.json"):
        parent = config_path.parent
        has_tokenizer = (parent / "tokenizer.json").exists()
        has_weights = any(parent.glob("*.bin")) or any(parent.glob("*.safetensors"))
        if has_tokenizer and has_weights:
            candidates.append(parent)
    preferred = [
        path for path in candidates if "arc-codet5-660m" in str(path).lower()
    ]
    if preferred:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"unique CodeT5 model not found; candidates={candidates}")


class MemoryMonitor:
    def __init__(self):
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
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
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                self.peak_mib = max([self.peak_mib, *values])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=3)


def run_condition(
    condition: str,
    source_root: Path,
    model_dir: Path,
    challenges_path: Path,
    task_ids: list[str],
) -> tuple[dict, dict]:
    condition_dir = WORK_ROOT / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    vendor_root = source_root / "arc_2025_mindsai"
    env.update(
        {
            "PYTHONPATH": str(vendor_root),
            "ARC_DATA_PATH": str(challenges_path),
            "DATA_PATH": str(challenges_path),
            "MODEL_CONFIG_MODULES": "phase1_codet5",
            "PHASE1_MODEL_PATH": str(model_dir),
            "PHASE1_CONDITION": condition,
            "TEST_MODE": "true",
            "TEST_MODE_NUM_TASKS": str(len(task_ids)),
            "TEST_MODE_TASK_IDS": ",".join(task_ids),
            "TEST_MODE_TTT_ITEMS": str(RUN_CONFIG["ttt_items"]),
            "TEST_MODE_INFERENCE_ITEMS": str(RUN_CONFIG["inference_items"]),
            "GLOBAL_TTT_ENABLED": "false" if condition == "baseline" else "true",
            "SE": "1",
            "USE_TPU": "false",
            "USE_FLAX_GPU": "false",
            "PRESERVE_FINETUNED_MODEL": "false",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    log_path = condition_dir / "run.log"
    started = time.perf_counter()
    with log_path.open("w") as log, MemoryMonitor() as monitor:
        process = subprocess.Popen(
            [sys.executable, "-m", "src.main"],
            cwd=condition_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{condition}] {line}", end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code:
        raise RuntimeError(f"{condition} failed with exit code {return_code}")
    log_text = log_path.read_text()
    if condition != "baseline" and (
        "TTT failed" in log_text or "Training failed" in log_text
    ):
        raise RuntimeError(f"{condition} did not complete its gradient update")
    submission_path = condition_dir / "submission.json"
    if not submission_path.exists():
        raise FileNotFoundError(f"{condition} produced no submission")
    return json.loads(submission_path.read_text()), {
        "seconds": round(elapsed, 3),
        "peak_gpu_memory_mib": monitor.peak_mib,
    }


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    source_root = extract_sources()
    sys.path.insert(0, str(source_root))
    from arc_solver.ttt_eval import compare_conditions, write_phase1_bundle

    challenge_source, solution_source = find_arc_files()
    model_dir = find_model_dir()
    selected_ids = TASK_IDS[: int(RUN_CONFIG["task_limit"])]
    bundle_dir = WORK_ROOT / "data"
    task_ids_path = bundle_dir / "task_ids.json"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    task_ids_path.write_text(json.dumps(selected_ids))
    write_phase1_bundle(
        challenge_source, solution_source, task_ids_path, bundle_dir
    )
    challenges_path = bundle_dir / "phase1_challenges.json"
    labels = json.loads((bundle_dir / "phase1_solutions.json").read_text())

    print(
        json.dumps(
            {
                "mode": RUN_CONFIG["mode"],
                "task_ids": selected_ids,
                "conditions": RUN_CONFIG["conditions"],
                "model_dir": str(model_dir),
                "challenge_source": str(challenge_source),
            },
            indent=2,
        ),
        flush=True,
    )
    submissions = {}
    timings = {}
    for condition in RUN_CONFIG["conditions"]:
        submissions[condition], timings[condition] = run_condition(
            condition, source_root, model_dir, challenges_path, selected_ids
        )
        (WORK_ROOT / f"submission_{condition}.json").write_text(
            json.dumps(submissions[condition])
        )

    report = compare_conditions(submissions, labels, timings)
    report["mode"] = RUN_CONFIG["mode"]
    report["model_dir"] = str(model_dir)
    report["config"] = RUN_CONFIG
    (WORK_ROOT / "phase1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
