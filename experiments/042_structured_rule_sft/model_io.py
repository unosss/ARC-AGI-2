"""Shared prompt, target, parsing, and multimodal dataset helpers."""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


Grid = list[list[int]]
ARC_COLORS = (
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
)
SYSTEM_PROMPTS = {
    "direct": (
        "Infer the single transformation shared by the demonstrations. "
        "Return only the query output grid as a compact JSON array of arrays."
    ),
    "program": (
        "Infer the single transformation shared by the demonstrations. "
        "Return only an executable Python program between <program> and "
        "</program>. It must define transform(grid), accept a rectangular "
        "list of integer rows, and return the output grid. Only Python, safe "
        "builtins, and numpy as np may be used."
    ),
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_grid(grid: Grid, cell_size: int) -> Image.Image:
    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width * cell_size, height * cell_size))
    draw = ImageDraw.Draw(image)
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            x0, y0 = column * cell_size, row * cell_size
            draw.rectangle(
                (x0, y0, x0 + cell_size - 1, y0 + cell_size - 1),
                fill=ARC_COLORS[value],
            )
    return image


def _add_grid(
    content: list[dict[str, Any]],
    label: str,
    grid: Grid,
    modality: str,
    cell_size: int,
) -> None:
    content.append({"type": "text", "text": f"\n{label}\nJSON:{compact(grid)}\n"})
    if modality == "both":
        content.append({"type": "image", "image": render_grid(grid, cell_size)})


def build_messages(
    row: dict[str, Any],
    arm: str,
    modality: str,
    cell_size: int,
    include_answer: bool,
) -> list[dict[str, Any]]:
    if arm not in SYSTEM_PROMPTS:
        raise ValueError(f"unsupported arm: {arm}")
    if modality not in {"text", "both"}:
        raise ValueError(f"unsupported modality: {modality}")
    content: list[dict[str, Any]] = []
    for index, pair in enumerate(row["demos"], 1):
        _add_grid(content, f"DEMONSTRATION {index} INPUT", pair["input"], modality, cell_size)
        _add_grid(content, f"DEMONSTRATION {index} OUTPUT", pair["output"], modality, cell_size)
    _add_grid(content, "QUERY INPUT", row["query_input"], modality, cell_size)
    content.append({"type": "text", "text": "\nProduce the requested answer now."})
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPTS[arm]}],
        },
        {"role": "user", "content": content},
    ]
    if include_answer:
        if arm == "direct":
            target = compact(row["query_output"])
        else:
            target = "<program>\n" + row["program"].rstrip() + "\n</program>"
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        )
    return messages


def apply_template(processor, messages, add_generation_prompt: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


class EpisodeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        processor,
        arm: str,
        modality: str,
        cell_size: int,
        max_length: int,
    ):
        self.rows = rows
        self.processor = processor
        self.arm = arm
        self.modality = modality
        self.cell_size = cell_size
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt = apply_template(
            self.processor,
            build_messages(
                row, self.arm, self.modality, self.cell_size, include_answer=False
            ),
            add_generation_prompt=True,
        )
        full = apply_template(
            self.processor,
            build_messages(
                row, self.arm, self.modality, self.cell_size, include_answer=True
            ),
            add_generation_prompt=False,
        )
        prompt_ids = prompt["input_ids"][0]
        full_ids = full["input_ids"][0]
        prompt_length, full_length = len(prompt_ids), len(full_ids)
        if not torch.equal(full_ids[:prompt_length], prompt_ids):
            raise RuntimeError(f"{row['id']}: prompt is not a prefix of full sequence")
        if full_length > self.max_length:
            raise RuntimeError(
                f"{row['id']}: sequence has {full_length} tokens, limit is "
                f"{self.max_length}; truncation is forbidden"
            )
        labels = full["input_ids"].clone()
        labels[:, :prompt_length] = -100
        full["labels"] = labels
        return dict(full)


def single_item_collator(features: list[dict[str, torch.Tensor]]):
    if len(features) != 1:
        raise RuntimeError("physical micro-batch size must remain one")
    return features[0]


def valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    return (
        1 <= width <= 30
        and all(len(row) == width for row in value)
        and all(
            isinstance(cell, int)
            and not isinstance(cell, bool)
            and 0 <= cell <= 9
            for row in value
            for cell in row
        )
    )


def parse_grid(text: str) -> Grid | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if valid_grid(value):
            return value
    return None


def parse_program(text: str) -> str | None:
    tagged = re.search(r"<program>\s*(.*?)\s*</program>", text, flags=re.DOTALL)
    if tagged:
        return tagged.group(1).strip() + "\n"
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    if "def transform(" in text:
        return text[text.index("def transform(") :].strip() + "\n"
    return None
