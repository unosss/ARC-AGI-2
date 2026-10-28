"""Offline Kaggle entrypoint for Experiment 022 Phase A."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

SOURCE_ARCHIVE = "__SOURCE_ARCHIVE__"
CONFIG = json.loads(r"""__RUN_CONFIG__""")
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/experiment_022/phase_a_report.json")


def unique_file(fragment: str) -> Path:
    matches = sorted(path for path in INPUT.rglob("*.json") if fragment in path.name)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {fragment!r} JSON, found {matches}")
    return matches[0]


def model_dir() -> Path:
    matches = sorted(
        path.parent
        for path in INPUT.rglob("config.json")
        if "qwen3-4b-grids15-sft139" in str(path).lower().replace("_", "-")
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected exact Qwen checkpoint once, found {matches}")
    return matches[0]


def main() -> None:
    source = Path("/tmp/experiment_022_source")
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
        archive.extractall(source)
    sys.path.insert(0, str(source))
    config_path = source / "runtime-config.json"
    config_path.write_text(json.dumps(CONFIG))
    from benchmark import run

    run(
        config_path,
        model_dir(),
        unique_file("training_challenges"),
        OUTPUT,
    )


if __name__ == "__main__":
    main()
