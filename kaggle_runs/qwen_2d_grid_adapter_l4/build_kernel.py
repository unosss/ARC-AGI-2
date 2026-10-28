"""Build the offline monolithic Experiment 021 Kaggle script."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "qwen_2d_grid_adapter_l4.py"


def build_archive() -> str:
    buffer = BytesIO()
    files = [
        ROOT / "arc_solver" / "__init__.py",
        ROOT / "arc_solver" / "qwen2d_metadata.py",
        ROOT / "arc_solver" / "qwen2d_adapter.py",
        ROOT / "arc_solver" / "qwen2d_augment.py",
        ROOT / "arc_solver" / "qwen2d_experiment.py",
        ROOT / "arc_solver" / "qwen2d_runtime.py",
    ]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("arc_solver") / path.name)
        archive.write(HERE / "experiment_worker.py", "experiment_worker.py")
        archive.write(HERE / "gpu_arm_worker.py", "gpu_arm_worker.py")
        archive.write(HERE / "truth_nll_worker.py", "truth_nll_worker.py")
        archive.write(HERE / "calibrate.py", "calibrate.py")
        archive.write(HERE / "orchestration_safety.py", "orchestration_safety.py")
        archive.write(
            ROOT / "experiments" / "021_qwen_2d_grid_adapter" / "preregistered_manifest.json",
            "preregistered_manifest.json",
        )
        archive.write(
            ROOT / "experiments" / "021_qwen_2d_grid_adapter" / "report_schema.json",
            "report_schema.json",
        )
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full-config.json")
    args = parser.parse_args()
    config = json.loads((HERE / args.config).read_text())
    template = (HERE / "runner_template.py").read_text()
    output = template.replace("__SOURCE_ARCHIVE__", build_archive()).replace(
        "__RUN_CONFIG__", json.dumps(config, separators=(",", ":"))
    )
    if "__SOURCE_ARCHIVE__" in output or "__RUN_CONFIG__" in output:
        raise AssertionError("unreplaced build placeholder")
    OUTPUT.write_text(output)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
