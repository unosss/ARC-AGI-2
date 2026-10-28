"""Build the standalone Kaggle script from the tested perception module."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "qwen38_perception_l4.py"


def main() -> None:
    module = (ROOT / "arc_solver" / "perception.py").read_text()
    runner = (HERE / "runner_template.py").read_text()
    OUTPUT.write_text(module + "\n\n" + runner)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
