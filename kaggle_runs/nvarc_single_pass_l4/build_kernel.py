"""Pack the pinned NVARC single-pass runtime into one Kaggle script."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "nvarc_single_pass_l4.py"


def build_archive() -> str:
    buffer = BytesIO()
    sources = [
        (ROOT / "third_party" / "nvarc_single_pass", Path(".")),
        (ROOT / "arc_solver" / "__init__.py", Path("arc_solver/__init__.py")),
        (ROOT / "arc_solver" / "ttt_eval.py", Path("arc_solver/ttt_eval.py")),
        (ROOT / "arc_solver" / "nvarc_eval.py", Path("arc_solver/nvarc_eval.py")),
    ]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, destination in sources:
            if source.is_dir():
                for path in sorted(source.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts:
                        relative = path.relative_to(source)
                        if source.name == "nvarc_single_pass" and relative == Path(
                            "arc_solver.py"
                        ):
                            relative = Path("nvarc_arc_solver.py")
                        if source.name == "nvarc_single_pass" and relative == Path(
                            "starter.py"
                        ):
                            patched = path.read_text().replace(
                                "from arc_solver import worker",
                                "from nvarc_arc_solver import worker",
                            )
                            archive.writestr(str(destination / relative), patched)
                        else:
                            archive.write(path, destination / relative)
            else:
                archive.write(source, destination)
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke-config.json")
    args = parser.parse_args()
    template = (HERE / "runner_template.py").read_text()
    config = json.loads((HERE / args.config).read_text())
    script = template.replace("__SOURCE_ARCHIVE__", build_archive())
    script = script.replace(
        "__RUN_CONFIG__", json.dumps(config, separators=(",", ":"))
    )
    OUTPUT.write_text(script)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
