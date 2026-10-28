"""Deterministic transformations, prompts, and scoring for Experiment 029."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_solver.perception import render_grid_image, validate_grid


Grid = list[list[int]]
SYSTEM_PROMPT = (
    "Learn the transformation for this ARC task from task-time training. "
    "Return only the output grid as a JSON array of integer rows. "
    "Do not explain the answer."
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def d4(grid: Grid, index: int) -> Grid:
    validate_grid(grid)
    if not 0 <= index < 8:
        raise ValueError("D4 index must be in 0..7")
    value = [row[:] for row in grid]
    if index == 0:
        return value
    if index == 1:
        return [list(row) for row in zip(*reversed(value))]
    if index == 2:
        return [list(reversed(row)) for row in reversed(value)]
    if index == 3:
        return [list(row) for row in reversed(list(zip(*value)))]
    if index == 4:
        return [list(reversed(row)) for row in value]
    if index == 5:
        return list(reversed(value))
    if index == 6:
        return [list(row) for row in zip(*value)]
    return [list(reversed(row)) for row in reversed(list(zip(*value)))]


def color_permutations(seed: int, count: int = 16) -> list[list[int]]:
    if count < 1:
        raise ValueError("at least one color permutation is required")
    rng = random.Random(seed)
    permutations = [list(range(10))]
    seen = {tuple(permutations[0])}
    while len(permutations) < count:
        foreground = list(range(1, 10))
        rng.shuffle(foreground)
        permutation = [0] + foreground
        if tuple(permutation) not in seen:
            permutations.append(permutation)
            seen.add(tuple(permutation))
    return permutations


def recolor(grid: Grid, permutation: list[int]) -> Grid:
    validate_grid(grid)
    if len(permutation) != 10 or sorted(permutation) != list(range(10)):
        raise ValueError("color permutation must be a bijection over 0..9")
    if permutation[0] != 0:
        raise ValueError("color 0 must remain fixed")
    return [[permutation[cell] for cell in row] for row in grid]


def augment(grid: Grid, permutation: list[int], geometry: int) -> Grid:
    return d4(recolor(grid, permutation), geometry)


def step_transform(config: dict[str, Any], step: int) -> tuple[list[int], int]:
    total = config["color_permutations"] * config["geometric_transforms"]
    if total != config["optimizer_steps"]:
        raise ValueError("optimizer steps must cover color x geometry exactly")
    if not 0 <= step < total:
        raise ValueError("step is outside the fixed augmentation schedule")
    permutations = color_permutations(
        config["color_seed"], config["color_permutations"]
    )
    return permutations[step // config["geometric_transforms"]], (
        step % config["geometric_transforms"]
    )


def user_content(grid: Grid, cell_size: int) -> list[dict[str, Any]]:
    validate_grid(grid)
    return [
        {
            "type": "image",
            "image": render_grid_image(grid, cell_size=cell_size),
        },
        {
            "type": "text",
            "text": (
                "Transform this input grid.\nInput grid JSON:\n"
                + json.dumps(grid, separators=(",", ":"))
                + "\nReturn only the output grid JSON."
            ),
        },
    ]


def messages(
    input_grid: Grid,
    output_grid: Grid | None,
    cell_size: int,
) -> list[dict[str, Any]]:
    payload = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {"role": "user", "content": user_content(input_grid, cell_size)},
    ]
    if output_grid is not None:
        validate_grid(output_grid)
        payload.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(output_grid, separators=(",", ":")),
                    }
                ],
            }
        )
    return payload


def apply_template(processor, payload, generation: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": generation,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(
            payload, enable_thinking=False, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(payload, **kwargs)


def supervised_encoding(
    processor,
    input_grid: Grid,
    output_grid: Grid,
    cell_size: int,
    max_length: int,
):
    prompt = apply_template(
        processor, messages(input_grid, None, cell_size), generation=True
    )
    full = apply_template(
        processor, messages(input_grid, output_grid, cell_size), generation=False
    )
    prompt_ids = prompt["input_ids"][0]
    full_ids = full["input_ids"][0]
    prompt_length = int(prompt_ids.shape[0])
    full_length = int(full_ids.shape[0])
    if not (full_ids[:prompt_length] == prompt_ids).all():
        raise RuntimeError("generation prompt is not a prefix of supervised sequence")
    if prompt_length >= max_length:
        raise RuntimeError("prompt leaves no supervised token inside sequence budget")
    if full_length > max_length:
        raise RuntimeError(
            f"full supervised sequence has {full_length} tokens and exceeds "
            f"the fixed limit {max_length}"
        )
    labels = full["input_ids"].clone()
    labels[:, :prompt_length] = -100
    full["labels"] = labels
    return full


def valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        return False
    if not 1 <= len(value) <= 30 or not 1 <= len(value[0]) <= 30:
        return False
    width = len(value[0])
    return all(
        isinstance(row, list)
        and len(row) == width
        and all(type(cell) is int and 0 <= cell <= 9 for cell in row)
        for row in value
    )


def parse_grids(text: str, limit: int = 2) -> list[Grid]:
    decoder = json.JSONDecoder()
    results = []
    seen = set()
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if valid_grid(value):
            key = canonical_bytes(value)
            if key not in seen:
                results.append(value)
                seen.add(key)
                if len(results) == limit:
                    break
    return results


def exact(prediction: Grid, truth: Grid) -> bool:
    return prediction == truth


def cell_accuracy(prediction: Grid | None, truth: Grid) -> float:
    if prediction is None or not valid_grid(prediction):
        return 0.0
    if len(prediction) != len(truth) or len(prediction[0]) != len(truth[0]):
        return 0.0
    total = len(truth) * len(truth[0])
    return sum(
        prediction[row][column] == truth[row][column]
        for row in range(len(truth))
        for column in range(len(truth[0]))
    ) / total
