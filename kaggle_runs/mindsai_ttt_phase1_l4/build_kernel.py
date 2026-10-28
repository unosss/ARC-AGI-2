"""Pack the pinned MindsAI runtime into one Kaggle script."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "mindsai_ttt_phase1.py"


def build_archive() -> str:
    buffer = BytesIO()
    sources = [
        (
            ROOT / "third_party" / "arc_2025_mindsai",
            Path("arc_2025_mindsai"),
        ),
        (ROOT / "arc_solver", Path("arc_solver")),
    ]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source_root, archive_root in sources:
            for path in sorted(source_root.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(
                        path,
                        archive_root / path.relative_to(source_root),
                    )
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    template = (HERE / "runner_template.py").read_text()
    run_config = json.loads((HERE / "phase1-config.json").read_text())
    task_ids = json.loads(
        (
            ROOT
            / "experiments"
            / "013_test_time_training"
            / "phase1_task_ids.json"
        ).read_text()
    )
    script = template.replace("__SOURCE_ARCHIVE__", build_archive())
    script = script.replace(
        "__RUN_CONFIG__", json.dumps(run_config, separators=(",", ":"))
    )
    script = script.replace(
        "__TASK_IDS__", json.dumps(task_ids, separators=(",", ":"))
    )
    OUTPUT.write_text(script)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
