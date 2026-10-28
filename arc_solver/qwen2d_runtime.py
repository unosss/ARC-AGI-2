"""HF/PEFT-compatible training, scoring, and deterministic cached DFS runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import torch

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter, StructuralForward
from arc_solver.qwen2d_metadata import (
    GeneratedGridState,
    LayoutMetadata,
    PINS,
    SegmentRole,
    build_layout,
    serialize_grid,
)


def serialize_chatml(
    train_pairs: Sequence[dict[str, Any]],
    query: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]] | None = None,
) -> tuple[list[int], int]:
    """Serialize exactly like the grids15 QwenFormatter, without text retokenizing."""
    tokens: list[int] = []
    for pair in train_pairs:
        tokens.extend((PINS.im_start, PINS.user, PINS.newline))
        tokens.extend(serialize_grid(pair["input"]))
        tokens.append(PINS.eos)
        tokens.extend((PINS.im_start, PINS.assistant, PINS.newline))
        tokens.extend(serialize_grid(pair["output"]))
        tokens.append(PINS.eos)
    tokens.extend((PINS.im_start, PINS.user, PINS.newline))
    tokens.extend(serialize_grid(query))
    tokens.append(PINS.eos)
    tokens.extend((PINS.im_start, PINS.assistant, PINS.newline))
    answer_start = len(tokens)
    if target is not None:
        tokens.extend(serialize_grid(target))
        tokens.append(PINS.eos)
    return tokens, answer_start


def serialize_chatml_bounded(
    train_pairs: Sequence[dict[str, Any]],
    query: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]] | None = None,
    *,
    max_length: int,
    reserve_new_tokens: int = 0,
) -> tuple[list[int], int, int]:
    """Serialize after deterministically dropping oldest demonstrations.

    ``reserve_new_tokens`` is used for generation prefixes so the combined
    prompt and maximum completion fit the declared context budget.
    """
    if max_length <= 0 or reserve_new_tokens < 0:
        raise ValueError("invalid context limits")
    limit = max_length - reserve_new_tokens
    if limit <= 0:
        raise ValueError("completion reserve leaves no prompt capacity")
    demos = list(train_pairs)
    dropped = 0
    while True:
        tokens, answer_start = serialize_chatml(demos, query, target)
        if len(tokens) <= limit:
            return tokens, answer_start, dropped
        if not demos:
            raise ValueError(
                f"single query/target uses {len(tokens)} tokens, exceeding {limit}"
            )
        demos.pop(0)
        dropped += 1


def completion_labels(
    input_ids: torch.Tensor, layout: LayoutMetadata, loss_token_limit: int | None = None
) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError("completion_labels expects one unpadded token row")
    labels = torch.full_like(input_ids, -100)
    index = 0
    while index < len(input_ids):
        if int(input_ids[index]) != PINS.assistant:
            index += 1
            continue
        if (
            index == 0
            or int(input_ids[index - 1]) != PINS.im_start
            or index + 1 >= len(input_ids)
            or int(input_ids[index + 1]) != PINS.newline
        ):
            raise ValueError("assistant role token lacks prefix newline")
        start = index + 2
        end = start
        while end < len(input_ids) and int(input_ids[end]) != PINS.eos:
            end += 1
        if end >= len(input_ids):
            raise ValueError("unterminated assistant completion")
        labels[start : end + 1] = input_ids[start : end + 1]
        index = end + 1
    if loss_token_limit is not None:
        positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
        if len(positions) > loss_token_limit:
            labels[positions[loss_token_limit:]] = -100
    return labels


def collate_token_rows(
    rows: Sequence[dict[str, Any]], *, arm: str, shuffle_seed: int
) -> dict[str, Any]:
    layouts = [
        build_layout(row["input_ids"], arm=arm, shuffle_seed=shuffle_seed + index)
        for index, row in enumerate(rows)
    ]
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), PINS.pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for index, (row, layout) in enumerate(zip(rows, layouts)):
        values = torch.as_tensor(row["input_ids"], dtype=torch.long)
        ids[index, : len(values)] = values
        labels[index, : len(values)] = completion_labels(
            values,
            layout,
            loss_token_limit=row.get("loss_token_limit"),
        )
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": ids.ne(PINS.pad),
        "layout": LayoutMetadata.stack(layouts, pad_to=width),
    }


def train_fixed_steps(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    batches: Iterable[dict[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    steps: int,
    scale: float = 1.0,
    deadline_epoch: float | None = None,
) -> list[float]:
    """Train through the common sidecar route; exhaustion is a hard failure."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    model.train()
    structure.train()
    forward = StructuralForward(model, structure, scale=scale)
    iterator = iter(batches)
    losses = []
    for step in range(steps):
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise TimeoutError("run deadline reached during training")
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise RuntimeError(f"training data exhausted at step {step}/{steps}") from error
        device = next(model.parameters()).device
        optimizer.zero_grad(set_to_none=True)
        output = forward(
            input_ids=batch["input_ids"].to(device),
            layout=batch["layout"],
            labels=batch["labels"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            return_dict=True,
            use_cache=False,
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("non-finite or absent structural training loss")
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))
    return losses


@torch.no_grad()
def teacher_forced_metrics(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    input_ids: torch.Tensor,
    layout: LayoutMetadata,
    answer_start: int,
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    forward = StructuralForward(model, structure, scale=scale)
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    output = forward(
        input_ids=ids,
        layout=layout,
        return_dict=True,
        use_cache=False,
    )
    logits = output.logits.float()
    target = ids[:, answer_start:]
    prediction = logits[:, answer_start - 1 : -1]
    token_nll = torch.nn.functional.cross_entropy(
        prediction.reshape(-1, prediction.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    answer_layout = layout.row[:, answer_start:].to(token_nll.device)
    first_token = float(token_nll[:, 0].mean())
    per_row = []
    for row_id in sorted(set(int(value) for value in answer_layout.flatten()) - {0}):
        mask = answer_layout == row_id
        per_row.append({"row": row_id - 1, "mean_nll": float(token_nll[mask].mean())})
    return {
        "sum_nll": float(token_nll.sum()),
        "mean_nll": float(token_nll.mean()),
        "first_token_nll": first_token,
        "per_row_nll": per_row,
        "token_count": int(token_nll.numel()),
    }


def multi_view_nll(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    views: Sequence[tuple[torch.Tensor, LayoutMetadata, int]],
    *,
    scale: float = 1.0,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    metrics = []
    for ids, layout, start in views:
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise TimeoutError("run deadline reached during multi-view scoring")
        metrics.append(
            teacher_forced_metrics(model, structure, ids, layout, start, scale=scale)
        )
    return {
        "view_count": len(metrics),
        "mean_sum_nll": sum(item["sum_nll"] for item in metrics) / len(metrics),
        "views": metrics,
    }


@torch.no_grad()
def batched_multi_view_nll(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    views: Sequence[tuple[torch.Tensor, LayoutMetadata, int]],
    *,
    max_batch_size: int = 8,
    scale: float = 1.0,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    """Score equal-length views together while preserving input order.

    Grouping by exact token length avoids padding and its possible numerical or
    positional side effects. Different answer offsets are supported because
    each row is reduced independently after the shared forward pass.
    """
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if not views:
        return {"view_count": 0, "mean_sum_nll": math.nan, "views": []}
    model.eval()
    device = next(model.parameters()).device
    forward = StructuralForward(model, structure, scale=scale)
    by_length: dict[int, list[int]] = {}
    for index, (ids, layout, start) in enumerate(views):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("each view must have input_ids shape [1, sequence]")
        if any(
            tuple(value.shape) != tuple(ids.shape)
            for value in layout.as_dict().values()
        ):
            raise ValueError("view layout shape does not match input_ids")
        if not 1 <= start < ids.shape[1]:
            raise ValueError("answer_start must select a non-empty completion")
        by_length.setdefault(ids.shape[1], []).append(index)

    metrics: list[dict[str, Any] | None] = [None] * len(views)
    for indices in by_length.values():
        for offset in range(0, len(indices), max_batch_size):
            if deadline_epoch is not None and time.time() >= deadline_epoch:
                raise TimeoutError("run deadline reached during batched multi-view scoring")
            chunk = indices[offset : offset + max_batch_size]
            ids = torch.cat([views[index][0] for index in chunk], dim=0).to(device)
            if any(views[index][0].shape[1] != ids.shape[1] for index in chunk):
                raise AssertionError("batched scoring attempted to pad unequal lengths")
            layouts = LayoutMetadata.stack(
                [
                    LayoutMetadata(
                        **{
                            name: value.squeeze(0)
                            for name, value in views[index][1].as_dict().items()
                        }
                    )
                    for index in chunk
                ]
            )
            output = forward(
                input_ids=ids,
                layout=layouts,
                return_dict=True,
                use_cache=False,
            )
            logits = output.logits.float()
            for batch_index, view_index in enumerate(chunk):
                start = views[view_index][2]
                target = ids[batch_index, start:]
                prediction = logits[batch_index, start - 1 : -1]
                token_nll = torch.nn.functional.cross_entropy(
                    prediction.reshape(-1, prediction.shape[-1]),
                    target,
                    reduction="none",
                )
                answer_layout = layouts.row[batch_index, start:].to(token_nll.device)
                per_row = []
                for row_id in sorted(
                    set(int(value) for value in answer_layout.flatten()) - {0}
                ):
                    mask = answer_layout == row_id
                    per_row.append(
                        {"row": row_id - 1, "mean_nll": float(token_nll[mask].mean())}
                    )
                metrics[view_index] = {
                    "sum_nll": float(token_nll.sum()),
                    "mean_nll": float(token_nll.mean()),
                    "first_token_nll": float(token_nll[0]),
                    "per_row_nll": per_row,
                    "token_count": int(token_nll.numel()),
                }
    complete = [item for item in metrics if item is not None]
    if len(complete) != len(views):
        raise AssertionError("batched multi-view scoring omitted a view")
    return {
        "view_count": len(complete),
        "mean_sum_nll": sum(item["sum_nll"] for item in complete) / len(complete),
        "views": complete,
    }


@dataclass(frozen=True)
class DFSBudget:
    max_nodes: int
    max_model_forward_tokens: int
    max_candidates: int
    max_new_tokens: int = 931
    emergency_seconds: float = 900.0
    min_prob: float = 0.2
    deadline_epoch: float | None = None


@dataclass
class DFSResult:
    candidates: list[dict[str, Any]]
    nodes_used: int
    model_forward_tokens: int
    stopped_by: str


@dataclass(frozen=True)
class BatchedDFSTelemetry:
    """Physical model work performed by a batched DFS invocation."""

    model_calls: int
    physical_batch_size: int
    physical_forward_tokens: int
    active_lanes_per_call: tuple[int, ...]
    physical_tokens_per_call: tuple[int, ...]

    @property
    def active_lanes(self) -> tuple[int, ...]:
        """Alias emphasizing that counts are recorded for each model call."""
        return self.active_lanes_per_call


@dataclass
class BatchedDFSResult:
    """Per-lane logical results and shared physical execution telemetry."""

    results: list[DFSResult]
    telemetry: BatchedDFSTelemetry

    @property
    def lane_results(self) -> list[DFSResult]:
        return self.results


def _next_allowed(tokens: tuple[int, ...], state: GeneratedGridState) -> list[int]:
    del tokens  # Kept in the private signature for compatibility with callers.
    return state.allowed_tokens()


def _mutable_cache_length(cache: Any) -> int | None:
    """Validate cache semantics and snapshot a supported mutable cache."""
    if cache is None or isinstance(cache, tuple):
        return None
    get_seq_length = getattr(cache, "get_seq_length", None)
    crop = getattr(cache, "crop", None)
    if callable(get_seq_length) and callable(crop):
        length = int(get_seq_length())
        if length < 0:
            raise ValueError("cache sequence length cannot be negative")
        return length
    raise TypeError(
        "unsupported mutable cache type "
        f"{type(cache).__module__}.{type(cache).__qualname__}; "
        "expected a legacy tuple cache or a cache with get_seq_length() and crop()"
    )


@torch.no_grad()
def deterministic_cached_dfs(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    prefix_ids: torch.Tensor,
    prefix_layout: LayoutMetadata,
    *,
    grid_bucket: int,
    budget: DFSBudget,
    scale: float = 1.0,
    generated_arm: str = "rowcol_component",
    generated_shuffle_seed: int | None = None,
) -> DFSResult:
    """Depth-first constrained search with deterministic compute accounting.

    DynamicCache is cropped on backtracking, so sibling branches see exactly
    the same prefix cache without retaining one full KV copy per open beam.
    Legacy tuple caches are immutable and are passed by value.
    """
    if prefix_ids.shape[0] != 1:
        raise ValueError("DFS currently requires batch size one")
    if not 0.0 < budget.min_prob <= 1.0:
        raise ValueError("DFS min_prob must be in (0, 1]")
    if sys.getrecursionlimit() < budget.max_new_tokens + 256:
        sys.setrecursionlimit(budget.max_new_tokens + 256)
    model.eval()
    device = next(model.parameters()).device
    ids = prefix_ids.to(device)
    forward = StructuralForward(model, structure, scale=scale)
    initial = forward(
        input_ids=ids,
        layout=prefix_layout,
        return_dict=True,
        use_cache=True,
    )
    forward_tokens = int(ids.numel())
    _mutable_cache_length(initial.past_key_values)
    candidates: list[dict[str, Any]] = []
    nodes = 0
    started = time.monotonic()
    max_nll = -math.log(budget.min_prob)
    stopped_by = "search_exhausted"
    halted = False

    def visit(
        score: float,
        tokens: tuple[int, ...],
        cache: Any,
        logits: torch.Tensor,
        state: GeneratedGridState,
    ) -> None:
        nonlocal nodes, forward_tokens, stopped_by, halted
        if halted:
            return
        if len(candidates) >= budget.max_candidates:
            stopped_by, halted = "candidate_budget", True
            return
        if nodes >= budget.max_nodes:
            stopped_by, halted = "node_budget", True
            return
        if time.monotonic() - started >= budget.emergency_seconds:
            stopped_by, halted = "emergency_time_cap", True
            return
        if (
            budget.deadline_epoch is not None
            and time.time() >= budget.deadline_epoch
        ):
            stopped_by, halted = "run_deadline", True
            return
        log_probs = torch.log_softmax(logits.float(), dim=-1)[0]
        ranked = sorted(
            _next_allowed(tokens, state),
            key=lambda token: (-float(log_probs[token]), token),
        )
        for token in ranked:
            if halted:
                return
            if nodes >= budget.max_nodes:
                stopped_by, halted = "node_budget", True
                return
            child_tokens = tokens + (token,)
            child_score = score - float(log_probs[token])
            # Match upstream turbo_dfs exactly: equality is pruned.
            if not child_score < max_nll:
                continue
            child_state = state.clone()
            token_layout = child_state.metadata_for(token)
            nodes += 1
            if token == PINS.eos:
                candidates.append({"tokens": list(child_tokens), "nll": child_score})
                if len(candidates) >= budget.max_candidates:
                    stopped_by, halted = "candidate_budget", True
                    return
                continue
            if len(child_tokens) >= budget.max_new_tokens:
                continue
            if forward_tokens + 1 > budget.max_model_forward_tokens:
                stopped_by, halted = "forward_token_budget", True
                return
            cache_length = _mutable_cache_length(cache)
            one = torch.tensor([[token]], dtype=torch.long, device=device)
            try:
                output = forward(
                    input_ids=one,
                    layout=LayoutMetadata.stack([token_layout]),
                    past_key_values=cache,
                    return_dict=True,
                    use_cache=True,
                )
                forward_tokens += 1
                _mutable_cache_length(output.past_key_values)
                visit(
                    child_score,
                    child_tokens,
                    output.past_key_values,
                    output.logits[:, -1],
                    child_state,
                )
            finally:
                if cache_length is not None:
                    cache.crop(cache_length)

    visit(
        0.0,
        (),
        initial.past_key_values,
        initial.logits[:, -1],
        GeneratedGridState(
            grid_bucket=grid_bucket,
            arm=generated_arm,
            shuffle_seed=generated_shuffle_seed,
        ),
    )
    candidates.sort(key=lambda item: (item["nll"], item["tokens"]))
    return DFSResult(candidates, nodes, forward_tokens, stopped_by)


@dataclass
class _BatchedDFSFrame:
    score: float
    tokens: tuple[int, ...]
    logits: torch.Tensor
    state: GeneratedGridState


@torch.no_grad()
def batched_deterministic_cached_dfs(
    model: torch.nn.Module,
    structure: AdditiveStructureAdapter,
    prefix_ids: torch.Tensor,
    prefix_layout: LayoutMetadata,
    *,
    grid_buckets: Sequence[int],
    budgets: Sequence[DFSBudget],
    arms: Sequence[str],
    shuffle_seeds: Sequence[int | None],
    scale: float = 1.0,
    max_physical_batch: int = 4,
) -> BatchedDFSResult:
    """Run independent deterministic DFS lanes in one physical model batch.

    Prefixes must already be a rectangular tensor: variable-length/padded
    prefixes are deliberately unsupported. At recursive calls, lanes without
    work send PAD with entirely null metadata. Logical DFS counters exclude
    those PAD tokens; telemetry reports all physical work.
    """
    if prefix_ids.ndim != 2:
        raise ValueError("prefix_ids must have shape [batch, equal_prefix_length]")
    batch_size, prefix_length = prefix_ids.shape
    if not 1 <= max_physical_batch <= 4:
        raise ValueError("max_physical_batch must be in 1..4")
    if not 1 <= batch_size <= max_physical_batch:
        raise ValueError(
            f"physical batch size {batch_size} exceeds limit {max_physical_batch}"
        )
    if torch.any(prefix_ids == PINS.pad):
        raise ValueError("padded or unequal-length prefixes are not supported")
    lane_inputs = (grid_buckets, budgets, arms, shuffle_seeds)
    if any(len(values) != batch_size for values in lane_inputs):
        raise ValueError("grid_buckets, budgets, arms, and shuffle_seeds must match batch")
    expected_shape = tuple(prefix_ids.shape)
    if any(tuple(value.shape) != expected_shape for value in prefix_layout.as_dict().values()):
        raise ValueError(
            "prefix layout must have shape [batch, equal_prefix_length] for every field"
        )
    for budget in budgets:
        if not 0.0 < budget.min_prob <= 1.0:
            raise ValueError("DFS min_prob must be in (0, 1]")
    recursion_limit = max(budget.max_new_tokens for budget in budgets) + 256
    if sys.getrecursionlimit() < recursion_limit:
        sys.setrecursionlimit(recursion_limit)

    model.eval()
    device = next(model.parameters()).device
    ids = prefix_ids.to(device)
    forward = StructuralForward(model, structure, scale=scale)
    initial = forward(
        input_ids=ids,
        layout=prefix_layout,
        return_dict=True,
        use_cache=True,
    )
    _mutable_cache_length(initial.past_key_values)

    candidates: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    nodes = [0] * batch_size
    logical_forward_tokens = [prefix_length] * batch_size
    stopped_by = ["search_exhausted"] * batch_size
    halted = [False] * batch_size
    started = [time.monotonic() for _ in range(batch_size)]
    max_nll = [-math.log(budget.min_prob) for budget in budgets]
    active_lanes_per_call = [batch_size]
    physical_tokens_per_call = [int(ids.numel())]

    null_metadata = LayoutMetadata.stack(
        [
            LayoutMetadata(
                **{
                    name: torch.zeros(1, dtype=torch.long)
                    for name in prefix_layout.as_dict()
                }
            )
            for _ in range(batch_size)
        ]
    )

    def halt_at_entry(lane: int) -> bool:
        budget = budgets[lane]
        if halted[lane]:
            return True
        if len(candidates[lane]) >= budget.max_candidates:
            stopped_by[lane], halted[lane] = "candidate_budget", True
        elif nodes[lane] >= budget.max_nodes:
            stopped_by[lane], halted[lane] = "node_budget", True
        elif time.monotonic() - started[lane] >= budget.emergency_seconds:
            stopped_by[lane], halted[lane] = "emergency_time_cap", True
        elif budget.deadline_epoch is not None and time.time() >= budget.deadline_epoch:
            stopped_by[lane], halted[lane] = "run_deadline", True
        return halted[lane]

    def visit(frames: list[_BatchedDFSFrame | None], cache: Any) -> None:
        rankings: list[list[int] | None] = [None] * batch_size
        positions = [0] * batch_size
        for lane, frame in enumerate(frames):
            if frame is None or halt_at_entry(lane):
                continue
            log_probs = torch.log_softmax(frame.logits.float(), dim=-1)
            rankings[lane] = sorted(
                _next_allowed(frame.tokens, frame.state),
                key=lambda token: (-float(log_probs[token]), token),
            )

        while True:
            child_frames: list[_BatchedDFSFrame | None] = [None] * batch_size
            child_tokens = torch.full(
                (batch_size, 1), PINS.pad, dtype=torch.long, device=device
            )
            child_layouts = [
                LayoutMetadata(
                    **{
                        name: getattr(null_metadata, name)[lane].clone()
                        for name in null_metadata.as_dict()
                    }
                )
                for lane in range(batch_size)
            ]

            for lane, frame in enumerate(frames):
                ranking = rankings[lane]
                if frame is None or ranking is None or halted[lane]:
                    continue
                log_probs = torch.log_softmax(frame.logits.float(), dim=-1)
                while positions[lane] < len(ranking):
                    if nodes[lane] >= budgets[lane].max_nodes:
                        stopped_by[lane], halted[lane] = "node_budget", True
                        break
                    token = ranking[positions[lane]]
                    positions[lane] += 1
                    child_score = frame.score - float(log_probs[token])
                    if not child_score < max_nll[lane]:
                        continue
                    state = frame.state.clone()
                    token_layout = state.metadata_for(token)
                    tokens = frame.tokens + (token,)
                    nodes[lane] += 1
                    if token == PINS.eos:
                        candidates[lane].append(
                            {"tokens": list(tokens), "nll": child_score}
                        )
                        if len(candidates[lane]) >= budgets[lane].max_candidates:
                            stopped_by[lane], halted[lane] = "candidate_budget", True
                            break
                        continue
                    if len(tokens) >= budgets[lane].max_new_tokens:
                        continue
                    if (
                        logical_forward_tokens[lane] + 1
                        > budgets[lane].max_model_forward_tokens
                    ):
                        stopped_by[lane], halted[lane] = "forward_token_budget", True
                        break
                    child_tokens[lane, 0] = token
                    child_layouts[lane] = token_layout
                    child_frames[lane] = _BatchedDFSFrame(
                        child_score, tokens, torch.empty(0), state
                    )
                    break

            active = sum(frame is not None for frame in child_frames)
            if active == 0:
                return

            cache_length = _mutable_cache_length(cache)
            try:
                output = forward(
                    input_ids=child_tokens,
                    layout=LayoutMetadata.stack(child_layouts),
                    past_key_values=cache,
                    return_dict=True,
                    use_cache=True,
                )
                active_lanes_per_call.append(active)
                physical_tokens_per_call.append(batch_size)
                output_cache_length = _mutable_cache_length(output.past_key_values)
                if (
                    cache_length is not None
                    and output_cache_length != cache_length + 1
                ):
                    raise AssertionError(
                        "mutable cache did not advance by exactly one token"
                    )
                for lane, frame in enumerate(child_frames):
                    if frame is not None:
                        logical_forward_tokens[lane] += 1
                        frame.logits = output.logits[lane, -1]
                visit(child_frames, output.past_key_values)
            finally:
                if cache_length is not None:
                    cache.crop(cache_length)
                    if _mutable_cache_length(cache) != cache_length:
                        raise AssertionError("mutable cache restoration failed")

    roots = [
        _BatchedDFSFrame(
            0.0,
            (),
            initial.logits[lane, -1],
            GeneratedGridState(
                grid_bucket=grid_buckets[lane],
                arm=arms[lane],
                shuffle_seed=shuffle_seeds[lane],
            ),
        )
        for lane in range(batch_size)
    ]
    visit(roots, initial.past_key_values)

    results = []
    for lane in range(batch_size):
        candidates[lane].sort(key=lambda item: (item["nll"], item["tokens"]))
        results.append(
            DFSResult(
                candidates[lane],
                nodes[lane],
                logical_forward_tokens[lane],
                stopped_by[lane],
            )
        )
    telemetry = BatchedDFSTelemetry(
        model_calls=len(active_lanes_per_call),
        physical_batch_size=batch_size,
        physical_forward_tokens=sum(physical_tokens_per_call),
        active_lanes_per_call=tuple(active_lanes_per_call),
        physical_tokens_per_call=tuple(physical_tokens_per_call),
    )
    return BatchedDFSResult(results, telemetry)


def save_candidate_pool(path: str | Path, result: DFSResult, metadata: dict[str, Any]) -> None:
    """Atomically materialize candidates before any evaluator can open labels."""
    import json
    import os
    import tempfile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "candidates": result.candidates,
        "nodes_used": result.nodes_used,
        "model_forward_tokens": result.model_forward_tokens,
        "stopped_by": result.stopped_by,
        **metadata,
    }
    with tempfile.NamedTemporaryFile(
        "w", dir=destination.parent, delete=False, prefix=destination.name
    ) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        temporary = handle.name
    os.replace(temporary, destination)
