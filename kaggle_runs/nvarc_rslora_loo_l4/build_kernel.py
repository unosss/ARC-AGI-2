"""Pack the dedicated NVARC paired LOO runtime into one Kaggle script."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "nvarc_rslora_loo_l4.py"


def build_archive() -> str:
    buffer = BytesIO()
    upstream = ROOT / "third_party" / "nvarc_single_pass"
    sources = [
        (upstream / "arc_loader.py", Path("arc_loader.py")),
        (upstream / "arc_decoder.py", Path("arc_decoder.py")),
        (upstream / "arc_solver.py", Path("nvarc_upstream_solver.py")),
        (HERE / "loo_worker.py", Path("loo_worker.py")),
        (ROOT / "arc_solver" / "__init__.py", Path("arc_solver/__init__.py")),
        (ROOT / "arc_solver" / "ttt_eval.py", Path("arc_solver/ttt_eval.py")),
        (ROOT / "arc_solver" / "nvarc_eval.py", Path("arc_solver/nvarc_eval.py")),
        (ROOT / "arc_solver" / "nvarc_loo.py", Path("arc_solver/nvarc_loo.py")),
    ]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, destination in sources:
            archive.write(source, destination)
        starter = (upstream / "starter.py").read_text().replace(
            "from arc_solver import worker", "from loo_worker import worker"
        )
        archive.writestr("starter.py", starter)
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    template = (HERE / "runner_template.py").read_text()
    config = json.loads((HERE / "config.json").read_text())
    script = template.replace("__SOURCE_ARCHIVE__", build_archive())
    script = script.replace(
        "__RUN_CONFIG__", json.dumps(config, separators=(",", ":"))
    )
    OUTPUT.write_text(script)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
