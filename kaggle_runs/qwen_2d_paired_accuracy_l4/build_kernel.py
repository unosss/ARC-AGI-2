"""Build Experiment 023's self-contained Kaggle script."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "qwen_2d_paired_accuracy_l4.py"


def archive() -> str:
    buffer = BytesIO()
    files = [
        ROOT / "arc_solver" / "__init__.py",
        ROOT / "arc_solver" / "qwen2d_metadata.py",
        ROOT / "arc_solver" / "qwen2d_adapter.py",
        ROOT / "arc_solver" / "qwen2d_augment.py",
        ROOT / "arc_solver" / "qwen2d_experiment.py",
        ROOT / "arc_solver" / "qwen2d_runtime.py",
    ]
    source_run = ROOT / "kaggle_runs" / "qwen_2d_grid_adapter_l4"
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as value:
        for path in files:
            value.write(path, Path("arc_solver") / path.name)
        value.write(source_run / "gpu_arm_worker.py", "gpu_arm_worker.py")
        value.write(HERE / "exclusions.json", "exclusions.json")
        value.write(
            ROOT
            / "experiments"
            / "023_qwen_2d_paired_accuracy"
            / "preregistered_manifest.json",
            "preregistered_manifest.json",
        )
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke-config.json")
    args = parser.parse_args()
    config = json.loads((HERE / args.config).read_text())
    output = (HERE / "runner_template.py").read_text()
    output = output.replace("__SOURCE_ARCHIVE__", archive()).replace(
        "__RUN_CONFIG__", json.dumps(config, separators=(",", ":"))
    )
    if "__SOURCE_ARCHIVE__" in output or "__RUN_CONFIG__" in output:
        raise AssertionError("unreplaced build placeholder")
    OUTPUT.write_text(output)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
