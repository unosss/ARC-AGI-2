"""Build the self-contained Experiment 022 Kaggle script."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "qwen_2d_batched_dfs_l4.py"


def archive() -> str:
    buffer = BytesIO()
    files = [
        ROOT / "arc_solver" / "__init__.py",
        ROOT / "arc_solver" / "qwen2d_metadata.py",
        ROOT / "arc_solver" / "qwen2d_adapter.py",
        ROOT / "arc_solver" / "qwen2d_augment.py",
        ROOT / "arc_solver" / "qwen2d_runtime.py",
    ]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as value:
        for path in files:
            value.write(path, Path("arc_solver") / path.name)
        value.write(HERE / "benchmark.py", "benchmark.py")
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke-config.json")
    args = parser.parse_args()
    config = json.loads((HERE / args.config).read_text())
    template = (HERE / "runner_template.py").read_text()
    output = template.replace("__SOURCE_ARCHIVE__", archive()).replace(
        "__RUN_CONFIG__", json.dumps(config, separators=(",", ":"))
    )
    if "__SOURCE_ARCHIVE__" in output or "__RUN_CONFIG__" in output:
        raise AssertionError("unreplaced build placeholder")
    OUTPUT.write_text(output)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
