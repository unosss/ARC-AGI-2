"""Non-autoregressive 2-D grid refinement head for Qwen hidden states.

The language-model sequence is compressed into a fixed number of latent
vectors before any of the 900 canvas cells attend to it.  This keeps memory
bounded for long Qwen contexts while preserving a fully parallel grid decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MAX_GRID_SIZE = 30
NUM_COLORS = 10
UNKNOWN = 10
OUTSIDE = 11
NUM_CANVAS_STATES = 12


def _validate_rectangular_grid(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or not grid:
        raise ValueError("grid must be a non-empty sequence of rows")
    width = len(grid[0])
    if width == 0:
        raise ValueError("grid rows cannot be empty")
    if len(grid) > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
        raise ValueError(f"grid dimensions must be at most {MAX_GRID_SIZE}x{MAX_GRID_SIZE}")
    for row in grid:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError("each grid row must be a sequence")
        if len(row) != width:
            raise ValueError("grid must be rectangular")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("ARC colors must be integers")
            if not 0 <= value < NUM_COLORS:
                raise ValueError(f"ARC colors must be in 0..{NUM_COLORS - 1}")
    return len(grid), width


def grid_to_canvas(
    grid: Sequence[Sequence[int]] | Tensor,
    *,
    max_size: int = MAX_GRID_SIZE,
    device: torch.device | str | None = None,
) -> Tensor:
    """Pad one rectangular ARC grid to ``[max_size, max_size]`` with OUTSIDE."""
    if not 1 <= max_size <= MAX_GRID_SIZE:
        raise ValueError(f"max_size must be in 1..{MAX_GRID_SIZE}")
    if isinstance(grid, Tensor):
        if grid.ndim != 2 or grid.shape[0] == 0 or grid.shape[1] == 0:
            raise ValueError("grid tensor must have non-empty shape [height, width]")
        if grid.shape[0] > max_size or grid.shape[1] > max_size:
            raise ValueError("grid tensor is larger than the requested canvas")
        if grid.dtype == torch.bool or grid.is_floating_point() or grid.is_complex():
            raise TypeError("grid tensor must have an integer dtype")
        values = grid.to(device=device, dtype=torch.long)
        if torch.any((values < 0) | (values >= NUM_COLORS)):
            raise ValueError(f"ARC colors must be in 0..{NUM_COLORS - 1}")
    else:
        height, width = _validate_rectangular_grid(grid)
        if height > max_size or width > max_size:
            raise ValueError("grid is larger than the requested canvas")
        values = torch.tensor(grid, dtype=torch.long, device=device)
    canvas = torch.full(
        (max_size, max_size), OUTSIDE, dtype=torch.long, device=values.device
    )
    canvas[: values.shape[0], : values.shape[1]] = values
    return canvas


def batch_grids_to_canvas(
    grids: Sequence[Sequence[Sequence[int]] | Tensor],
    *,
    max_size: int = MAX_GRID_SIZE,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert grids to a canvas batch and return true heights and widths."""
    if not grids:
        raise ValueError("grids cannot be empty")
    canvases, heights, widths = [], [], []
    for grid in grids:
        canvas = grid_to_canvas(grid, max_size=max_size, device=device)
        canvases.append(canvas)
        if isinstance(grid, Tensor):
            heights.append(grid.shape[0])
            widths.append(grid.shape[1])
        else:
            height, width = _validate_rectangular_grid(grid)
            heights.append(height)
            widths.append(width)
    return (
        torch.stack(canvases),
        torch.tensor(heights, dtype=torch.long, device=device),
        torch.tensor(widths, dtype=torch.long, device=device),
    )


def canvas_shape(canvas: Tensor) -> tuple[int, int]:
    """Infer the smallest top-left rectangle containing all non-OUTSIDE cells."""
    if canvas.ndim != 2:
        raise ValueError("canvas must have shape [height, width]")
    present = canvas.ne(OUTSIDE)
    rows = torch.nonzero(present.any(dim=1), as_tuple=False).flatten()
    cols = torch.nonzero(present.any(dim=0), as_tuple=False).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return 1, 1
    return int(rows[-1]) + 1, int(cols[-1]) + 1


def canvas_to_grid(
    canvas: Tensor,
    *,
    height: int | None = None,
    width: int | None = None,
    invalid_fill: int = 0,
) -> list[list[int]]:
    """Convert a canvas to a valid ARC grid, replacing non-color states."""
    if canvas.ndim != 2 or canvas.shape[0] == 0 or canvas.shape[1] == 0:
        raise ValueError("canvas must have non-empty shape [height, width]")
    if canvas.dtype == torch.bool or canvas.is_floating_point() or canvas.is_complex():
        raise TypeError("canvas must have an integer dtype")
    inferred_height, inferred_width = canvas_shape(canvas)
    height = inferred_height if height is None else height
    width = inferred_width if width is None else width
    if not 1 <= height <= min(MAX_GRID_SIZE, canvas.shape[0]):
        raise ValueError("height is outside the canvas")
    if not 1 <= width <= min(MAX_GRID_SIZE, canvas.shape[1]):
        raise ValueError("width is outside the canvas")
    if not 0 <= invalid_fill < NUM_COLORS:
        raise ValueError("invalid_fill must be an ARC color")
    region = canvas[:height, :width].detach().to(device="cpu", dtype=torch.long)
    region = torch.where(
        (region >= 0) & (region < NUM_COLORS),
        region,
        torch.full_like(region, invalid_fill),
    )
    return region.tolist()


@dataclass(frozen=True)
class RefinerConfig:
    """Architecture settings for :class:`Qwen2DRefiner`."""

    hidden_size: int
    model_dim: int = 128
    latent_dim: int = 128
    latent_slots: int = 32
    num_heads: int = 4
    ff_multiplier: int = 2
    num_stages: int = 3
    max_grid_size: int = MAX_GRID_SIZE

    def __post_init__(self) -> None:
        for name in ("hidden_size", "model_dim", "latent_dim", "latent_slots", "num_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_stages != 3:
            raise ValueError("Experiment 035 requires exactly three refinement stages")
        if self.max_grid_size != MAX_GRID_SIZE:
            raise ValueError(f"Experiment 035 uses a fixed {MAX_GRID_SIZE}x{MAX_GRID_SIZE} canvas")
        if self.model_dim % self.num_heads or self.latent_dim % self.num_heads:
            raise ValueError("model_dim and latent_dim must be divisible by num_heads")
        if self.ff_multiplier <= 0:
            raise ValueError("ff_multiplier must be positive")


@dataclass
class RefinementStageOutput:
    """Predictions emitted after one refinement stage."""

    color_logits: Tensor
    height_logits: Tensor
    width_logits: Tensor
    confidence: Tensor
    changed_mask: Tensor
    update_gate: Tensor
    canvas: Tensor


@dataclass
class RefinerOutput:
    """All three deep-supervised predictions and their discrete trajectory."""

    stages: tuple[RefinementStageOutput, ...]
    context_latents: Tensor
    trajectory: Tensor
    changed_trajectory: Tensor

    @property
    def final(self) -> RefinementStageOutput:
        return self.stages[-1]

    @property
    def color_logits(self) -> Tensor:
        return self.final.color_logits

    @property
    def height_logits(self) -> Tensor:
        return self.final.height_logits

    @property
    def width_logits(self) -> Tensor:
        return self.final.width_logits

    @property
    def confidence(self) -> Tensor:
        return self.final.confidence


@dataclass
class RefinerLoss:
    """Scalar training objective with detached diagnostic components."""

    loss: Tensor
    cell_loss: Tensor
    shape_loss: Tensor
    stage_losses: tuple[Tensor, ...]


class _FeedForward(nn.Module):
    def __init__(self, width: int, multiplier: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width * multiplier),
            nn.GELU(),
            nn.Linear(width * multiplier, width),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class LatentContextBottleneck(nn.Module):
    """Compress arbitrary token lengths to a fixed latent sequence."""

    def __init__(self, config: RefinerConfig):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.input_projection = nn.Linear(config.hidden_size, config.latent_dim)
        self.latent_queries = nn.Parameter(
            torch.empty(config.latent_slots, config.latent_dim)
        )
        self.cross_norm = nn.LayerNorm(config.latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.latent_dim, config.num_heads, dropout=0.0, batch_first=True
        )
        self.self_norm = nn.LayerNorm(config.latent_dim)
        self.self_attention = nn.MultiheadAttention(
            config.latent_dim, config.num_heads, dropout=0.0, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(config.latent_dim)
        self.ff = _FeedForward(config.latent_dim, config.ff_multiplier)
        nn.init.normal_(self.latent_queries, std=config.latent_dim**-0.5)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        tokens = self.input_projection(self.input_norm(hidden_states))
        queries = self.latent_queries.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        normalized = self.cross_norm(queries)
        attended, _ = self.cross_attention(
            normalized,
            tokens,
            tokens,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        latents = queries + attended
        normalized = self.self_norm(latents)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        latents = latents + attended
        return latents + self.ff(self.ff_norm(latents))


class _RefinementStage(nn.Module):
    def __init__(self, config: RefinerConfig):
        super().__init__()
        dim, heads = config.model_dim, config.num_heads
        self.size = config.max_grid_size
        self.row_norm = nn.LayerNorm(dim)
        self.row_attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.column_norm = nn.LayerNorm(dim)
        self.column_attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.context_norm = nn.LayerNorm(dim)
        self.context_attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True, kdim=config.latent_dim,
            vdim=config.latent_dim
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, config.ff_multiplier)
        self.candidate = nn.Linear(dim, dim)
        self.update_gate = nn.Linear(dim * 2, dim)
        self.color_head = nn.Linear(dim, NUM_CANVAS_STATES)
        self.confidence_head = nn.Linear(dim, 1)
        self.shape_norm = nn.LayerNorm(dim + config.latent_dim)
        self.height_head = nn.Linear(dim + config.latent_dim, self.size)
        self.width_head = nn.Linear(dim + config.latent_dim, self.size)

    def forward(
        self, state: Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, size, _, dim = state.shape
        rows = state.reshape(batch * size, size, dim)
        normalized = self.row_norm(rows)
        attended, _ = self.row_attention(
            normalized, normalized, normalized, need_weights=False
        )
        state = (rows + attended).reshape(batch, size, size, dim)

        columns = state.permute(0, 2, 1, 3).reshape(batch * size, size, dim)
        normalized = self.column_norm(columns)
        attended, _ = self.column_attention(
            normalized, normalized, normalized, need_weights=False
        )
        state = (columns + attended).reshape(batch, size, size, dim).permute(0, 2, 1, 3)

        flat = state.reshape(batch, size * size, dim)
        normalized = self.context_norm(flat)
        attended, _ = self.context_attention(
            normalized, context, context, need_weights=False
        )
        candidate_input = flat + attended
        candidate_input = candidate_input + self.ff(self.ff_norm(candidate_input))
        candidate = torch.tanh(self.candidate(candidate_input))
        gate = torch.sigmoid(self.update_gate(torch.cat((flat, candidate), dim=-1)))
        updated = (1.0 - gate) * flat + gate * candidate

        pooled = torch.cat((updated.mean(dim=1), context.mean(dim=1)), dim=-1)
        pooled = self.shape_norm(pooled)
        return (
            updated.reshape(batch, size, size, dim),
            self.color_head(updated).reshape(batch, size, size, NUM_CANVAS_STATES),
            self.height_head(pooled),
            self.width_head(pooled),
            self.confidence_head(updated).squeeze(-1).sigmoid().reshape(batch, size, size),
            gate.mean(dim=-1).reshape(batch, size, size),
        )


class Qwen2DRefiner(nn.Module):
    """Three-stage, non-autoregressive ARC canvas refiner."""

    def __init__(
        self,
        config: RefinerConfig | int | None = None,
        *,
        hidden_size: int | None = None,
        **config_overrides: Any,
    ):
        super().__init__()
        if config is None:
            if hidden_size is None:
                raise TypeError("provide config or hidden_size")
            config = hidden_size
        elif hidden_size is not None:
            raise TypeError("hidden_size cannot be combined with config")
        if isinstance(config, int):
            config = RefinerConfig(hidden_size=config, **config_overrides)
        elif config_overrides:
            raise TypeError("config_overrides are only valid when config is a hidden-size integer")
        if not isinstance(config, RefinerConfig):
            raise TypeError("config must be RefinerConfig or an integer hidden size")
        self.config = config
        self.context_bottleneck = LatentContextBottleneck(config)
        self.state_embedding = nn.Embedding(NUM_CANVAS_STATES, config.model_dim)
        self.row_embedding = nn.Embedding(config.max_grid_size, config.model_dim)
        self.column_embedding = nn.Embedding(config.max_grid_size, config.model_dim)
        self.confidence_projection = nn.Linear(1, config.model_dim, bias=False)
        self.stages = nn.ModuleList(
            [_RefinementStage(config) for _ in range(config.num_stages)]
        )

    def _prepare_canvas(self, draft_canvas: Tensor, batch: int, device: torch.device) -> Tensor:
        if not isinstance(draft_canvas, Tensor):
            raise TypeError("draft_canvas must be a tensor")
        if draft_canvas.ndim == 2:
            draft_canvas = draft_canvas.unsqueeze(0)
        if draft_canvas.ndim != 3 or draft_canvas.shape[0] != batch:
            raise ValueError("draft_canvas must have shape [batch, height, width]")
        if draft_canvas.shape[1] == 0 or draft_canvas.shape[2] == 0:
            raise ValueError("draft canvas dimensions cannot be empty")
        if draft_canvas.shape[1] > MAX_GRID_SIZE or draft_canvas.shape[2] > MAX_GRID_SIZE:
            raise ValueError(f"draft canvas dimensions cannot exceed {MAX_GRID_SIZE}")
        if draft_canvas.dtype == torch.bool or draft_canvas.is_floating_point() or draft_canvas.is_complex():
            raise TypeError("draft_canvas must have an integer dtype")
        draft = draft_canvas.to(device=device, dtype=torch.long)
        if torch.any((draft < 0) | (draft >= NUM_CANVAS_STATES)):
            raise ValueError(f"draft states must be in 0..{NUM_CANVAS_STATES - 1}")
        padded = torch.full(
            (batch, MAX_GRID_SIZE, MAX_GRID_SIZE),
            OUTSIDE,
            dtype=torch.long,
            device=device,
        )
        padded[:, : draft.shape[1], : draft.shape[2]] = draft
        return padded

    def _prepare_confidence(
        self, confidence: Tensor | None, batch: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if confidence is None:
            return torch.zeros(
                batch, MAX_GRID_SIZE, MAX_GRID_SIZE, device=device, dtype=dtype
            )
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(0)
        if confidence.ndim != 3 or confidence.shape[0] != batch:
            raise ValueError("draft_confidence must have shape [batch, height, width]")
        if confidence.shape[1] > MAX_GRID_SIZE or confidence.shape[2] > MAX_GRID_SIZE:
            raise ValueError("draft_confidence dimensions cannot exceed the canvas")
        if not confidence.is_floating_point():
            raise TypeError("draft_confidence must be floating point")
        if not torch.isfinite(confidence).all() or torch.any((confidence < 0) | (confidence > 1)):
            raise ValueError("draft_confidence must contain finite values in [0, 1]")
        padded = torch.zeros(
            batch, MAX_GRID_SIZE, MAX_GRID_SIZE, device=device, dtype=dtype
        )
        padded[:, : confidence.shape[1], : confidence.shape[2]] = confidence.to(
            device=device, dtype=dtype
        )
        return padded

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        draft_canvas: Tensor,
        draft_confidence: Tensor | None = None,
    ) -> RefinerOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden]")
        batch, tokens, hidden = hidden_states.shape
        if hidden != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, received {hidden}"
            )
        if tokens == 0 or batch == 0:
            raise ValueError("hidden_states batch and token dimensions must be non-empty")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")
        if attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask must have shape [batch, tokens]")
        mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("each sample must contain at least one attended token")

        draft = self._prepare_canvas(draft_canvas, batch, hidden_states.device)
        confidence = self._prepare_confidence(
            draft_confidence, batch, hidden_states.device, hidden_states.dtype
        )
        context = self.context_bottleneck(hidden_states, mask)
        positions = torch.arange(MAX_GRID_SIZE, device=hidden_states.device)
        state = self.state_embedding(draft)
        state = (
            state
            + self.row_embedding(positions)[None, :, None, :]
            + self.column_embedding(positions)[None, None, :, :]
            + self.confidence_projection(confidence.unsqueeze(-1))
        )

        prior = draft
        outputs = []
        trajectory, changes = [], []
        for stage in self.stages:
            state, color_logits, height_logits, width_logits, stage_confidence, gate = stage(
                state, context
            )
            # Use the categorical distribution itself as the calibrated
            # inference confidence.  The auxiliary head remains in the module
            # for checkpoint compatibility but is not trusted for edits.
            stage_confidence = color_logits.softmax(dim=-1)[..., :NUM_COLORS].amax(dim=-1)
            canvas = color_logits.argmax(dim=-1)
            changed = canvas.ne(prior)
            outputs.append(
                RefinementStageOutput(
                    color_logits=color_logits,
                    height_logits=height_logits,
                    width_logits=width_logits,
                    confidence=stage_confidence,
                    changed_mask=changed,
                    update_gate=gate,
                    canvas=canvas,
                )
            )
            trajectory.append(canvas)
            changes.append(changed)
            # The next stage sees a differentiable expected state embedding.
            probabilities = color_logits.softmax(dim=-1)
            expected = probabilities @ self.state_embedding.weight
            state = state + expected
            prior = canvas
        return RefinerOutput(
            stages=tuple(outputs),
            context_latents=context,
            trajectory=torch.stack(trajectory, dim=1),
            changed_trajectory=torch.stack(changes, dim=1),
        )


def _targets_to_canvas(
    target: Tensor | Sequence[Sequence[Sequence[int]]],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(target, Tensor):
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if target.ndim != 3:
            raise ValueError("target canvas must have shape [batch, height, width]")
        if target.shape[1] > MAX_GRID_SIZE or target.shape[2] > MAX_GRID_SIZE:
            raise ValueError("target dimensions exceed the canvas")
        if target.dtype == torch.bool or target.is_floating_point() or target.is_complex():
            raise TypeError("target canvas must have an integer dtype")
        values = target.to(device=device, dtype=torch.long)
        if torch.any((values < 0) | (values >= NUM_CANVAS_STATES)):
            raise ValueError("target canvas contains an invalid state")
        canvases = torch.full(
            (values.shape[0], MAX_GRID_SIZE, MAX_GRID_SIZE),
            OUTSIDE, dtype=torch.long, device=device
        )
        canvases[:, : values.shape[1], : values.shape[2]] = values
        shapes = [canvas_shape(value) for value in values]
        heights = torch.tensor([shape[0] for shape in shapes], device=device)
        widths = torch.tensor([shape[1] for shape in shapes], device=device)
        return canvases, heights, widths
    return batch_grids_to_canvas(target, device=device)


def refiner_loss(
    output: RefinerOutput,
    target: Tensor | Sequence[Sequence[Sequence[int]]],
    *,
    target_heights: Tensor | None = None,
    target_widths: Tensor | None = None,
    stage_weights: Sequence[float] = (1.0, 2.0, 3.0),
    shape_weight: float = 1.0,
) -> RefinerLoss:
    """Compute increasing deep supervision over cells and output dimensions."""
    if len(output.stages) != 3 or len(stage_weights) != len(output.stages):
        raise ValueError("one stage weight is required for each of the three stages")
    weights = torch.as_tensor(
        stage_weights, device=output.final.color_logits.device, dtype=torch.float32
    )
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("stage weights must be finite and positive")
    if not torch.all(weights[1:] > weights[:-1]):
        raise ValueError("deep-supervision weights must be strictly increasing")
    if shape_weight < 0:
        raise ValueError("shape_weight cannot be negative")

    canvases, inferred_heights, inferred_widths = _targets_to_canvas(
        target, device=output.final.color_logits.device
    )
    heights = inferred_heights if target_heights is None else target_heights.to(
        device=canvases.device, dtype=torch.long
    )
    widths = inferred_widths if target_widths is None else target_widths.to(
        device=canvases.device, dtype=torch.long
    )
    batch = output.final.color_logits.shape[0]
    if canvases.shape[0] != batch or heights.shape != (batch,) or widths.shape != (batch,):
        raise ValueError("target batch and shape tensors must match model output")
    if torch.any((heights < 1) | (heights > MAX_GRID_SIZE)):
        raise ValueError("target heights must be in 1..30")
    if torch.any((widths < 1) | (widths > MAX_GRID_SIZE)):
        raise ValueError("target widths must be in 1..30")
    rows = torch.arange(MAX_GRID_SIZE, device=canvases.device)[None, :, None]
    cols = torch.arange(MAX_GRID_SIZE, device=canvases.device)[None, None, :]
    valid = (rows < heights[:, None, None]) & (cols < widths[:, None, None])
    if torch.any(canvases[valid] >= NUM_COLORS):
        raise ValueError("target rectangle must contain only ARC colors")

    per_stage, cell_losses, shape_losses = [], [], []
    for stage in output.stages:
        if stage.color_logits.shape != (
            batch, MAX_GRID_SIZE, MAX_GRID_SIZE, NUM_CANVAS_STATES
        ):
            raise ValueError("stage color logits have an invalid shape")
        cell = F.cross_entropy(stage.color_logits[valid], canvases[valid])
        shape = (
            F.cross_entropy(stage.height_logits, heights - 1)
            + F.cross_entropy(stage.width_logits, widths - 1)
        ) / 2.0
        cell_losses.append(cell)
        shape_losses.append(shape)
        per_stage.append(cell + shape_weight * shape)
    normalized = weights / weights.sum()
    loss = sum(weight * value for weight, value in zip(normalized, per_stage))
    cell_loss = sum(weight * value for weight, value in zip(normalized, cell_losses))
    shape_loss = sum(weight * value for weight, value in zip(normalized, shape_losses))
    return RefinerLoss(loss, cell_loss, shape_loss, tuple(per_stage))


def decode_stage(
    stage: RefinementStageOutput,
    *,
    draft_canvas: Tensor | None = None,
    invalid_fill: int = 0,
    edit_threshold: float = 0.35,
    shape_threshold: float = 0.95,
) -> list[list[list[int]]]:
    """Decode one conservative correction step.

    A weak refiner must not erase a strong draft.  Cell edits therefore require
    joint categorical confidence and update-gate support, while dimensions are
    changed only by a high-confidence shape prediction.
    """
    if stage.color_logits.ndim != 4 or stage.color_logits.shape[-1] != NUM_CANVAS_STATES:
        raise ValueError("color_logits must have shape [batch, height, width, 12]")
    batch = stage.color_logits.shape[0]
    if not 0.0 <= edit_threshold <= 1.0 or not 0.0 <= shape_threshold <= 1.0:
        raise ValueError("decode thresholds must be in [0, 1]")
    height_probabilities = stage.height_logits.softmax(dim=-1)
    width_probabilities = stage.width_logits.softmax(dim=-1)
    heights = height_probabilities.argmax(dim=-1) + 1
    widths = width_probabilities.argmax(dim=-1) + 1
    color_probabilities = stage.color_logits.softmax(dim=-1)[..., :NUM_COLORS]
    color_confidence, predictions = color_probabilities.max(dim=-1)
    fallback: Tensor | None = None
    fallback_heights: list[int] = []
    fallback_widths: list[int] = []
    if draft_canvas is not None:
        if draft_canvas.ndim == 2:
            draft_canvas = draft_canvas.unsqueeze(0)
        if draft_canvas.ndim != 3 or draft_canvas.shape[0] != batch:
            raise ValueError("draft_canvas batch must match stage output")
        fallback = torch.full_like(predictions, invalid_fill)
        height = min(draft_canvas.shape[1], predictions.shape[1])
        width = min(draft_canvas.shape[2], predictions.shape[2])
        draft = draft_canvas[:, :height, :width].to(predictions.device, torch.long)
        draft = torch.where(
            (draft >= 0) & (draft < NUM_COLORS), draft, torch.full_like(draft, invalid_fill)
        )
        fallback[:, :height, :width] = draft
        for item in draft_canvas:
            if item.shape == (MAX_GRID_SIZE, MAX_GRID_SIZE):
                item_height, item_width = canvas_shape(item)
            else:
                item_height, item_width = item.shape
            fallback_heights.append(int(item_height))
            fallback_widths.append(int(item_width))
    if fallback is None:
        fallback = torch.full_like(predictions, invalid_fill)
    else:
        edit_support = color_confidence * stage.update_gate
        predictions = torch.where(edit_support >= edit_threshold, predictions, fallback)
        height_confidence = height_probabilities.amax(dim=-1)
        width_confidence = width_probabilities.amax(dim=-1)
        heights = torch.where(
            height_confidence >= shape_threshold,
            heights,
            torch.tensor(fallback_heights, device=heights.device),
        )
        widths = torch.where(
            width_confidence >= shape_threshold,
            widths,
            torch.tensor(fallback_widths, device=widths.device),
        )
    return [
        canvas_to_grid(
            predictions[index],
            height=int(heights[index]),
            width=int(widths[index]),
            invalid_fill=invalid_fill,
        )
        for index in range(batch)
    ]


def decode_stages(
    output: RefinerOutput,
    *,
    draft_canvas: Tensor | None = None,
    invalid_fill: int = 0,
) -> list[list[list[list[int]]]]:
    """Decode every refinement stage; outer order is stage, then batch."""
    decoded = []
    fallback = draft_canvas
    for stage in output.stages:
        grids = decode_stage(
            stage,
            draft_canvas=fallback,
            invalid_fill=invalid_fill,
        )
        decoded.append(grids)
        fallback, _, _ = batch_grids_to_canvas(
            grids,
            device=stage.color_logits.device,
        )
    return decoded


# Explicit descriptive aliases for callers that use experiment terminology.
QwenGridRefiner = Qwen2DRefiner
compute_refiner_loss = refiner_loss
ARC_NUM_COLORS = NUM_COLORS
CANVAS_UNKNOWN = UNKNOWN
CANVAS_OUTSIDE = OUTSIDE


__all__ = [
    "MAX_GRID_SIZE",
    "NUM_COLORS",
    "UNKNOWN",
    "OUTSIDE",
    "NUM_CANVAS_STATES",
    "ARC_NUM_COLORS",
    "CANVAS_UNKNOWN",
    "CANVAS_OUTSIDE",
    "RefinerConfig",
    "RefinementStageOutput",
    "RefinerOutput",
    "RefinerLoss",
    "LatentContextBottleneck",
    "Qwen2DRefiner",
    "QwenGridRefiner",
    "grid_to_canvas",
    "batch_grids_to_canvas",
    "canvas_shape",
    "canvas_to_grid",
    "refiner_loss",
    "compute_refiner_loss",
    "decode_stage",
    "decode_stages",
]
