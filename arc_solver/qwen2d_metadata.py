"""Token-aligned 2D sidecar metadata for the grids15 Qwen serialization.

This module is deliberately tokenizer-independent at import time.  Kaggle code
must call :func:`assert_grids15_tokenizer` against the loaded checkpoint before
constructing any training rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import random
from typing import Any, Iterable, Sequence

import torch


class SegmentRole(IntEnum):
    NULL = 0
    DEMO_INPUT = 1
    DEMO_OUTPUT = 2
    QUERY_INPUT = 3
    QUERY_OUTPUT = 4


@dataclass(frozen=True)
class TokenPins:
    cells: tuple[int, ...] = tuple(range(10))
    newline: int = 10
    user: int = 11
    assistant: int = 12
    pad: int = 13
    im_start: int = 14
    eos: int = 15


PINS = TokenPins()
NULL = 0
MAX_GRID_DIM = 30
FEATURE_NAMES = (
    "row",
    "col",
    "role",
    "grid",
    "component",
    "component_size",
    "bbox_height",
    "bbox_width",
    "topology",
)


@dataclass
class LayoutMetadata:
    """Integer feature ids aligned one-for-one with input tokens.

    Zero is the null id for every feature. Rows, columns and grid indices are
    shifted by one, leaving values 1..30 for legal ARC coordinates.
    """

    row: torch.Tensor
    col: torch.Tensor
    role: torch.Tensor
    grid: torch.Tensor
    component: torch.Tensor
    component_size: torch.Tensor
    bbox_height: torch.Tensor
    bbox_width: torch.Tensor
    topology: torch.Tensor

    def __len__(self) -> int:
        return int(self.row.numel())

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    def to(self, device: torch.device | str) -> "LayoutMetadata":
        return LayoutMetadata(**{k: v.to(device) for k, v in self.as_dict().items()})

    @classmethod
    def stack(
        cls, layouts: Sequence["LayoutMetadata"], pad_to: int | None = None
    ) -> "LayoutMetadata":
        if not layouts:
            raise ValueError("cannot stack an empty layout list")
        width = pad_to or max(len(layout) for layout in layouts)
        if any(len(layout) > width for layout in layouts):
            raise ValueError("pad_to is shorter than a layout")
        fields: dict[str, torch.Tensor] = {}
        for name in FEATURE_NAMES:
            rows = []
            for layout in layouts:
                value = getattr(layout, name)
                rows.append(torch.nn.functional.pad(value, (0, width - len(layout))))
            fields[name] = torch.stack(rows)
        return cls(**fields)


@dataclass(frozen=True)
class _Segment:
    role_token: int
    start: int
    end: int
    grid_start: int


def _encoded(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def assert_grids15_tokenizer(tokenizer: Any, pins: TokenPins = PINS) -> None:
    """Fail loudly unless the checkpoint has the exact 16-token contract."""
    expected = {
        **{str(i): [i] for i in range(10)},
        "\n": [pins.newline],
        "<|im_start|>": [pins.im_start],
        "<|im_start|>user\n": [pins.im_start, pins.user, pins.newline],
        "<|im_start|>assistant\n": [
            pins.im_start,
            pins.assistant,
            pins.newline,
        ],
        "<|im_end|>": [pins.eos],
    }
    failures = {
        text: {"expected": ids, "actual": _encoded(tokenizer, text)}
        for text, ids in expected.items()
        if _encoded(tokenizer, text) != ids
    }
    if getattr(tokenizer, "pad_token_id", None) != pins.pad:
        failures["pad_token_id"] = {
            "expected": pins.pad,
            "actual": getattr(tokenizer, "pad_token_id", None),
        }
    # HF reports the 13-token base vocabulary separately from the three added
    # special tokens (pad, im_start, im_end). ``len(tokenizer)`` is the model
    # embedding vocabulary and must therefore be 16.
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and vocab_size != 13:
        failures["base_vocab_size"] = {"expected": 13, "actual": vocab_size}
    try:
        total_size = len(tokenizer)
    except TypeError:
        total_size = None
    if total_size is not None and total_size != 16:
        failures["total_vocab_size"] = {"expected": 16, "actual": total_size}
    if failures:
        raise RuntimeError(f"grids15 tokenizer contract mismatch: {failures}")


def _segments(tokens: Sequence[int], pins: TokenPins) -> list[_Segment]:
    segments: list[_Segment] = []
    index = 0
    while index < len(tokens):
        token = int(tokens[index])
        if token == pins.pad:
            if any(int(value) != pins.pad for value in tokens[index:]):
                raise ValueError("non-pad token follows padding")
            break
        if token != pins.im_start:
            raise ValueError(f"expected im_start token at {index}, got {token}")
        if index + 1 >= len(tokens) or int(tokens[index + 1]) not in (
            pins.user,
            pins.assistant,
        ):
            raise ValueError(f"im_start at {index} is not followed by a role")
        role_token = int(tokens[index + 1])
        if index + 2 >= len(tokens) or int(tokens[index + 2]) != pins.newline:
            raise ValueError(f"role token at {index + 1} is not followed by newline")
        end = index + 3
        while end < len(tokens) and int(tokens[end]) not in (
            pins.eos,
            pins.im_start,
            pins.pad,
        ):
            if int(tokens[end]) not in pins.cells and int(tokens[end]) != pins.newline:
                raise ValueError(f"illegal grid token {tokens[end]} at {end}")
            end += 1
        # A generation prefix may end directly after assistant + newline.
        if end < len(tokens) and int(tokens[end]) == pins.eos:
            segment_end = end + 1
        elif end == len(tokens) and role_token == pins.assistant:
            segment_end = end
        else:
            raise ValueError(f"unterminated segment beginning at {index}")
        segments.append(_Segment(role_token, index, segment_end, index + 3))
        index = segment_end
    return segments


def _grid_shape_and_cells(
    tokens: Sequence[int], segment: _Segment, pins: TokenPins
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    grid_end = segment.end
    if grid_end > segment.grid_start and int(tokens[grid_end - 1]) == pins.eos:
        grid_end -= 1
    if grid_end == segment.grid_start:
        return 0, 0, []
    rows: list[list[tuple[int, int]]] = [[]]
    for pos in range(segment.grid_start, grid_end):
        token = int(tokens[pos])
        if token == pins.newline:
            if not rows[-1]:
                raise ValueError(f"empty grid row before token {pos}")
            rows.append([])
        else:
            rows[-1].append((pos, token))
    if not rows[-1]:
        raise ValueError("grid ends in newline; serialization must use strip()")
    width = len(rows[0])
    if not 1 <= len(rows) <= MAX_GRID_DIM or not 1 <= width <= MAX_GRID_DIM:
        raise ValueError(f"grid shape is outside ARC bounds: {len(rows)}x{width}")
    if any(len(row) != width for row in rows):
        raise ValueError("ragged serialized grid")
    cells = [
        (pos, row_index, col_index, color)
        for row_index, row in enumerate(rows)
        for col_index, (pos, color) in enumerate(row)
    ]
    return len(rows), width, cells


def _background(grid: list[list[int]]) -> int:
    # ARC's canonical background is color 0; using frequency would make
    # membership change when an object occupies most of a small grid.
    return 0


def _component_features(
    height: int, width: int, cells: Sequence[tuple[int, int, int, int]]
) -> dict[tuple[int, int], tuple[int, int, int, int, int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for _, row, col, color in cells:
        grid[row][col] = color
    background = _background(grid)
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] != background
    }
    components: list[list[tuple[int, int]]] = []
    while unseen:
        root = min(unseen)
        color = grid[root[0]][root[1]]
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            row, col = stack.pop()
            component.append((row, col))
            for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nxt in unseen and grid[nxt[0]][nxt[1]] == color:
                    unseen.remove(nxt)
                    stack.append(nxt)
        components.append(sorted(component))
    components.sort(key=lambda comp: comp[0])
    result: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    for slot, component in enumerate(components, 1):
        rows = [cell[0] for cell in component]
        cols = [cell[1] for cell in component]
        bbox_h = max(rows) - min(rows) + 1
        bbox_w = max(cols) - min(cols) + 1
        cell_set = set(component)
        edges = sum(
            (row + dr, col + dc) in cell_set
            for row, col in component
            for dr, dc in ((1, 0), (0, 1))
        )
        perimeter = 4 * len(component) - 2 * edges
        holes = _count_holes(cell_set, min(rows), max(rows), min(cols), max(cols))
        # Topology excludes color and absolute location.  Bucket combines holes,
        # perimeter and tree/cycle character and is invariant to cell ordering.
        topology = min(63, holes * 16 + min(7, perimeter // 2) * 2 + int(edges >= len(component)))
        feature = (
            min(slot, 64),
            min(len(component), 64),
            min(bbox_h, 30),
            min(bbox_w, 30),
            topology + 1,
        )
        for cell in component:
            result[cell] = feature
    return result


def _count_holes(
    component: set[tuple[int, int]],
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
) -> int:
    empty = {
        (row, col)
        for row in range(min_row - 1, max_row + 2)
        for col in range(min_col - 1, max_col + 2)
        if (row, col) not in component
    }
    outside = {(min_row - 1, min_col - 1)}
    stack = list(outside)
    while stack:
        row, col = stack.pop()
        for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nxt in empty and nxt not in outside:
                outside.add(nxt)
                stack.append(nxt)
    holes = 0
    remaining = empty - outside
    while remaining:
        holes += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return holes


def build_layout(
    input_ids: Sequence[int] | torch.Tensor,
    *,
    arm: str = "rowcol_component",
    shuffle_seed: int | None = None,
    pins: TokenPins = PINS,
) -> LayoutMetadata:
    """Parse a complete ChatML token sequence into aligned sidecar features."""
    tokens = [int(value) for value in input_ids]
    parsed = _segments(tokens, pins)
    fields = {name: torch.zeros(len(tokens), dtype=torch.long) for name in FEATURE_NAMES}
    user_indices = [i for i, segment in enumerate(parsed) if segment.role_token == pins.user]
    query_user = user_indices[-1] if user_indices else -1
    grid_index = 0
    rng = random.Random(0 if shuffle_seed is None else shuffle_seed)
    # Use fixed 30-coordinate permutations rather than shape-dependent maps.
    # The same mapping can then be reconstructed causally while an output grid
    # is generated, before its final height and width are known.
    shuffled_rows = list(range(MAX_GRID_DIM))
    shuffled_cols = list(range(MAX_GRID_DIM))
    if arm == "shuffled_rowcol":
        rng.shuffle(shuffled_rows)
        rng.shuffle(shuffled_cols)
    for segment_index, segment in enumerate(parsed):
        is_user = segment.role_token == pins.user
        is_query = segment_index >= query_user
        role = (
            SegmentRole.QUERY_INPUT
            if is_user and is_query
            else SegmentRole.DEMO_INPUT
            if is_user
            else SegmentRole.QUERY_OUTPUT
            if is_query
            else SegmentRole.DEMO_OUTPUT
        )
        fields["role"][segment.start : segment.end] = int(role)
        grid_index += 1
        fields["grid"][segment.start : segment.end] = min(grid_index, 31)
        height, width, cells = _grid_shape_and_cells(tokens, segment, pins)
        if not cells:
            continue
        row_map = (
            shuffled_rows if arm == "shuffled_rowcol" else list(range(height))
        )
        col_map = (
            shuffled_cols if arm == "shuffled_rowcol" else list(range(width))
        )
        for pos, row, col, _ in cells:
            fields["row"][pos] = row_map[row] + 1
            fields["col"][pos] = col_map[col] + 1
        if arm in ("rowcol_component", "shuffled_component") and is_user:
            component_map = _component_features(height, width, cells)
            assignments = [component_map.get((row, col), (0, 0, 0, 0, 0)) for _, row, col, _ in cells]
            if arm == "shuffled_component":
                non_null = [feature for feature in assignments if feature[0]]
                rng.shuffle(non_null)
                iterator = iter(non_null)
                assignments = [next(iterator) if feature[0] else feature for feature in assignments]
            for (pos, _, _, _), feature in zip(cells, assignments):
                for name, value in zip(FEATURE_NAMES[4:], feature):
                    fields[name][pos] = value
    if arm == "text_control":
        # The adapter still executes an exactly parameter-matched all-null path.
        fields = {name: torch.zeros_like(value) for name, value in fields.items()}
    if arm not in {
        "text_control",
        "rowcol",
        "rowcol_component",
        "shuffled_rowcol",
        "shuffled_component",
    }:
        raise ValueError(f"unknown arm: {arm}")
    layout = LayoutMetadata(**fields)
    assert_layout(input_ids=tokens, layout=layout, pins=pins)
    return layout


def assert_layout(
    input_ids: Sequence[int] | torch.Tensor,
    layout: LayoutMetadata,
    pins: TokenPins = PINS,
) -> None:
    tokens = [int(value) for value in input_ids]
    if len(layout) != len(tokens):
        raise AssertionError(f"layout/token length mismatch: {len(layout)} != {len(tokens)}")
    for name, values in layout.as_dict().items():
        if values.shape != (len(tokens),) or values.dtype != torch.long:
            raise AssertionError(f"invalid {name} layout tensor")
        if torch.any(values < 0):
            raise AssertionError(f"negative ids in {name}")
    pad_positions = torch.tensor([token == pins.pad for token in tokens])
    for name, values in layout.as_dict().items():
        if pad_positions.any() and torch.any(values[pad_positions] != 0):
            raise AssertionError(f"padding has non-null {name} metadata")
    assistant = (layout.role == int(SegmentRole.DEMO_OUTPUT)) | (
        layout.role == int(SegmentRole.QUERY_OUTPUT)
    )
    for name in FEATURE_NAMES[4:]:
        if torch.any(getattr(layout, name)[assistant] != 0):
            raise AssertionError(f"assistant tokens leak {name} metadata")


def layout_hash(layout: LayoutMetadata) -> str:
    digest = hashlib.sha256()
    for name in FEATURE_NAMES:
        digest.update(name.encode())
        digest.update(getattr(layout, name).numpy().tobytes())
    return digest.hexdigest()


class GeneratedGridState:
    """Causal row/column metadata state for KV-cached output decoding."""

    def __init__(
        self,
        *,
        grid_bucket: int,
        max_dim: int = MAX_GRID_DIM,
        arm: str = "rowcol_component",
        shuffle_seed: int | None = None,
    ):
        if not 1 <= max_dim <= MAX_GRID_DIM:
            raise ValueError(f"max_dim must be in 1..{MAX_GRID_DIM}")
        if arm not in {
            "text_control",
            "rowcol",
            "rowcol_component",
            "shuffled_rowcol",
            "shuffled_component",
        }:
            raise ValueError(f"unknown arm: {arm}")
        self.grid_bucket = min(max(1, grid_bucket), 31)
        self.max_dim = max_dim
        self.arm = arm
        rng = random.Random(0 if shuffle_seed is None else shuffle_seed)
        self.row_map = list(range(MAX_GRID_DIM))
        self.col_map = list(range(MAX_GRID_DIM))
        if arm == "shuffled_rowcol":
            rng.shuffle(self.row_map)
            rng.shuffle(self.col_map)
        self.row = 0
        self.col = 0
        self.width: int | None = None
        self.finished = False

    def allowed_tokens(self) -> list[int]:
        """Return tokens that preserve a rectangular, bounded grid prefix."""
        if self.finished:
            return []
        allowed: list[int] = []
        row_width = self.max_dim if self.width is None else self.width
        if self.row < self.max_dim and self.col < row_width:
            allowed.extend(PINS.cells)
        complete_row = self.col > 0 and (
            self.width is None or self.col == self.width
        )
        if complete_row:
            if self.row + 1 < self.max_dim:
                allowed.append(PINS.newline)
            allowed.append(PINS.eos)
        return allowed

    def metadata_for(self, token_id: int) -> LayoutMetadata:
        if self.finished:
            raise ValueError("cannot append after EOS")
        if token_id not in self.allowed_tokens():
            raise ValueError(
                f"token {token_id} would make an invalid generated grid "
                f"at row={self.row + 1}, col={self.col + 1}"
            )
        values = {name: torch.zeros(1, dtype=torch.long) for name in FEATURE_NAMES}
        if self.arm != "text_control":
            values["role"][0] = int(SegmentRole.QUERY_OUTPUT)
            values["grid"][0] = self.grid_bucket
            if token_id in PINS.cells:
                values["row"][0] = self.row_map[self.row] + 1
                values["col"][0] = self.col_map[self.col] + 1
        if token_id == PINS.newline:
            if self.width is None:
                self.width = self.col
            self.row += 1
            self.col = 0
        elif token_id in PINS.cells:
            self.col += 1
        elif token_id == PINS.eos:
            if self.width is None:
                self.width = self.col
            self.finished = True
        return LayoutMetadata(**values)

    def clone(self) -> "GeneratedGridState":
        other = GeneratedGridState(
            grid_bucket=self.grid_bucket,
            max_dim=self.max_dim,
            arm=self.arm,
        )
        other.row_map = self.row_map.copy()
        other.col_map = self.col_map.copy()
        other.row, other.col = self.row, self.col
        other.width, other.finished = self.width, self.finished
        return other


def serialize_grid(grid: Iterable[Iterable[int]], pins: TokenPins = PINS) -> list[int]:
    rows = [list(row) for row in grid]
    if not rows or not 1 <= len(rows) <= 30 or not 1 <= len(rows[0]) <= 30:
        raise ValueError("invalid grid shape")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("ragged grid")
    output: list[int] = []
    for row_index, row in enumerate(rows):
        if any(int(cell) not in pins.cells for cell in row):
            raise ValueError("cell outside 0..9")
        if row_index:
            output.append(pins.newline)
        output.extend(int(cell) for cell in row)
    return output
