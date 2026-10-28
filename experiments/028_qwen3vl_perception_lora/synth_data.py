"""Deterministic synthetic grids for Experiment 028."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_solver.perception import fixed_probe_description


Grid = list[list[int]]
FAMILIES = ("random", "components", "geometry", "structure")
MODALITIES = ("text", "image", "both")
SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def dimensions(rng: random.Random, index: int) -> tuple[int, int]:
    if index % 4 == 3:
        return rng.randint(17, 30), rng.randint(17, 30)
    return rng.randint(3, 16), rng.randint(3, 16)


def blank(height: int, width: int) -> Grid:
    return [[0 for _ in range(width)] for _ in range(height)]


def transform_shape(
    shape: tuple[tuple[int, int], ...], transform: int
) -> list[tuple[int, int]]:
    cells = []
    for row, column in shape:
        if transform == 0:
            cell = (row, column)
        elif transform == 1:
            cell = (column, -row)
        elif transform == 2:
            cell = (-row, -column)
        elif transform == 3:
            cell = (-column, row)
        elif transform == 4:
            cell = (row, -column)
        elif transform == 5:
            cell = (-row, column)
        elif transform == 6:
            cell = (column, row)
        else:
            cell = (-column, -row)
        cells.append(cell)
    min_row = min(row for row, _ in cells)
    min_column = min(column for _, column in cells)
    return sorted((row - min_row, column - min_column) for row, column in cells)


def place(
    grid: Grid,
    shape: list[tuple[int, int]],
    top: int,
    left: int,
    color: int,
    overwrite: bool = False,
) -> bool:
    height, width = len(grid), len(grid[0])
    cells = [(top + row, left + column) for row, column in shape]
    if any(not (0 <= row < height and 0 <= column < width) for row, column in cells):
        return False
    if not overwrite and any(grid[row][column] != 0 for row, column in cells):
        return False
    for row, column in cells:
        grid[row][column] = color
    return True


def random_grid(rng: random.Random, height: int, width: int) -> Grid:
    grid = blank(height, width)
    colors = rng.sample(range(1, 10), rng.randint(1, 4))
    density = rng.uniform(0.03, 0.14) if max(height, width) > 16 else rng.uniform(0.08, 0.48)
    for row in range(height):
        for column in range(width):
            if rng.random() < density:
                grid[row][column] = rng.choice(colors)
    if not any(cell for row in grid for cell in row):
        grid[rng.randrange(height)][rng.randrange(width)] = colors[0]
    return grid


def component_grid(rng: random.Random, height: int, width: int) -> Grid:
    grid = blank(height, width)
    shape_count = rng.randint(2, min(10, max(2, height * width // 12)))
    base_shapes = rng.sample(SHAPES, k=min(len(SHAPES), rng.randint(1, 4)))
    for _ in range(shape_count):
        shape = transform_shape(rng.choice(base_shapes), rng.randrange(8))
        shape_height = max(row for row, _ in shape) + 1
        shape_width = max(column for _, column in shape) + 1
        if shape_height > height or shape_width > width:
            continue
        color = rng.randint(1, 9)
        for _attempt in range(30):
            top = rng.randrange(height - shape_height + 1)
            left = rng.randrange(width - shape_width + 1)
            if place(grid, shape, top, left, color):
                break
    if not any(cell for row in grid for cell in row):
        grid[0][0] = rng.randint(1, 9)
    return grid


def geometry_grid(rng: random.Random, height: int, width: int) -> Grid:
    grid = blank(height, width)
    colors = rng.sample(range(1, 10), rng.randint(1, 3))
    primitive_count = rng.randint(1, 4)
    for index in range(primitive_count):
        color = colors[index % len(colors)]
        kind = rng.choice(("hline", "vline", "filled", "frame"))
        if kind == "hline":
            row = rng.randrange(height)
            left, right = sorted((rng.randrange(width), rng.randrange(width)))
            for column in range(left, right + 1):
                grid[row][column] = color
        elif kind == "vline":
            column = rng.randrange(width)
            top, bottom = sorted((rng.randrange(height), rng.randrange(height)))
            for row in range(top, bottom + 1):
                grid[row][column] = color
        else:
            top, bottom = sorted((rng.randrange(height), rng.randrange(height)))
            left, right = sorted((rng.randrange(width), rng.randrange(width)))
            if kind == "filled":
                for row in range(top, bottom + 1):
                    for column in range(left, right + 1):
                        grid[row][column] = color
            else:
                for column in range(left, right + 1):
                    grid[top][column] = color
                    grid[bottom][column] = color
                for row in range(top, bottom + 1):
                    grid[row][left] = color
                    grid[row][right] = color
    return grid


def structure_grid(rng: random.Random, height: int, width: int) -> Grid:
    mode = rng.choice(("horizontal", "vertical", "rotation", "tile"))
    grid = blank(height, width)
    colors = rng.sample(range(1, 10), rng.randint(1, 3))
    if mode == "horizontal":
        for row in range((height + 1) // 2):
            for column in range(width):
                if rng.random() < 0.18:
                    value = rng.choice(colors)
                    grid[row][column] = value
                    grid[height - 1 - row][column] = value
    elif mode == "vertical":
        for row in range(height):
            for column in range((width + 1) // 2):
                if rng.random() < 0.18:
                    value = rng.choice(colors)
                    grid[row][column] = value
                    grid[row][width - 1 - column] = value
    elif mode == "rotation":
        for row in range(height):
            for column in range(width):
                mirror = (height - 1 - row, width - 1 - column)
                if (row, column) <= mirror and rng.random() < 0.16:
                    value = rng.choice(colors)
                    grid[row][column] = value
                    grid[mirror[0]][mirror[1]] = value
    else:
        tile_height = rng.randint(1, min(5, height))
        tile_width = rng.randint(1, min(5, width))
        tile = [
            [
                rng.choice(colors) if rng.random() < 0.25 else 0
                for _ in range(tile_width)
            ]
            for _ in range(tile_height)
        ]
        if not any(cell for row in tile for cell in row):
            tile[0][0] = colors[0]
        for row in range(height):
            for column in range(width):
                grid[row][column] = tile[row % tile_height][column % tile_width]
    if not any(cell for row in grid for cell in row):
        grid[0][0] = colors[0]
        if mode == "horizontal":
            grid[-1][0] = colors[0]
        elif mode == "vertical":
            grid[0][-1] = colors[0]
        elif mode == "rotation":
            grid[-1][-1] = colors[0]
    return grid


GENERATORS = {
    "random": random_grid,
    "components": component_grid,
    "geometry": geometry_grid,
    "structure": structure_grid,
}


def generate_rows(
    seed: int, count: int, split: str, max_target_characters: int = 8000
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError("could not generate enough unique grids")
        index = len(rows)
        family = FAMILIES[index % len(FAMILIES)]
        modality = MODALITIES[index % len(MODALITIES)]
        height, width = dimensions(rng, index)
        grid = GENERATORS[family](rng, height, width)
        grid_hash = digest(grid)
        if grid_hash in seen:
            continue
        target = fixed_probe_description(grid)
        if len(json.dumps(target, separators=(",", ":"))) > max_target_characters:
            continue
        seen.add(grid_hash)
        rows.append(
            {
                "id": f"{split}-{len(rows):05d}",
                "split": split,
                "family": family,
                "modality": modality,
                "grid": grid,
                "grid_sha256": grid_hash,
                "target": target,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = generate_rows(
        config["train_seed"],
        config["train_grid_count"],
        split="train",
        max_target_characters=config["max_target_characters"],
    )
    evaluation = generate_rows(
        config["synthetic_eval_seed"],
        config["synthetic_eval_grid_count"],
        split="synthetic_eval",
        max_target_characters=config["max_target_characters"],
    )
    train_hashes = {row["grid_sha256"] for row in train}
    eval_hashes = {row["grid_sha256"] for row in evaluation}
    overlap = train_hashes & eval_hashes
    if overlap:
        raise RuntimeError(f"train/eval synthetic collision: {sorted(overlap)}")
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "synthetic_eval.jsonl"
    train_file_sha256 = write_jsonl(train_path, train)
    synthetic_eval_file_sha256 = write_jsonl(eval_path, evaluation)
    if train_file_sha256 != config["train_corpus_sha256"]:
        raise RuntimeError("generated training corpus differs from fixed SHA-256")
    if synthetic_eval_file_sha256 != config["synthetic_eval_sha256"]:
        raise RuntimeError("generated synthetic evaluation differs from fixed SHA-256")
    manifest = {
        "schema_version": 1,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "train_count": len(train),
        "synthetic_eval_count": len(evaluation),
        "train_file_sha256": train_file_sha256,
        "synthetic_eval_file_sha256": synthetic_eval_file_sha256,
        "train_grid_set_sha256": digest(sorted(train_hashes)),
        "synthetic_eval_grid_set_sha256": digest(sorted(eval_hashes)),
        "train_family_counts": dict(Counter(row["family"] for row in train)),
        "train_modality_counts": dict(Counter(row["modality"] for row in train)),
        "synthetic_eval_family_counts": dict(
            Counter(row["family"] for row in evaluation)
        ),
        "synthetic_eval_modality_counts": dict(
            Counter(row["modality"] for row in evaluation)
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
