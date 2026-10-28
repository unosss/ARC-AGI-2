"""Additive structural embeddings and the single model-forward integration point."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from arc_solver.qwen2d_metadata import FEATURE_NAMES, LayoutMetadata


VOCAB_SIZES = {
    "row": 31,
    "col": 31,
    "role": 5,
    "grid": 32,
    "component": 65,
    "component_size": 65,
    "bbox_height": 31,
    "bbox_width": 31,
    "topology": 65,
}


def hidden_size_from_config(config: Any) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return hidden_size_from_config(text_config)
    raise ValueError("model config has no supported hidden-size field")


class AdditiveStructureAdapter(nn.Module):
    """Parameter-identical adapter used by all experimental arms.

    Every feature has a scalar zero-initialized gate.  Null id 0 is fixed to
    zero through ``padding_idx``.  ``scale=0`` bypasses the complete delta,
    providing a numerical baseline-equivalence check.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, hidden_size, padding_idx=0)
                for name, size in VOCAB_SIZES.items()
            }
        )
        self.gates = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in FEATURE_NAMES}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=self.hidden_size**-0.5)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for gate in self.gates.values():
            nn.init.zeros_(gate)

    def forward(self, layout: LayoutMetadata, *, scale: float = 1.0) -> torch.Tensor:
        fields = layout.as_dict()
        shape = fields["row"].shape
        if any(value.shape != shape for value in fields.values()):
            raise ValueError("metadata feature shapes differ")
        reference = next(self.parameters())
        delta = torch.zeros(*shape, self.hidden_size, device=reference.device, dtype=reference.dtype)
        if scale == 0:
            return delta
        for name in FEATURE_NAMES:
            ids = fields[name].to(reference.device)
            if ids.numel() and int(ids.max()) >= VOCAB_SIZES[name]:
                raise ValueError(f"{name} id exceeds adapter vocabulary")
            delta = delta + torch.tanh(self.gates[name]) * self.embeddings[name](ids)
        return delta * scale

    def freeze(self) -> "AdditiveStructureAdapter":
        self.requires_grad_(False)
        self.eval()
        return self

    def save_arm(self, directory: str | Path, arm: str, metadata: dict[str, Any]) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "structure_adapter.pt")
        payload = {
            "schema_version": 1,
            "arm": arm,
            "hidden_size": self.hidden_size,
            "feature_vocab_sizes": VOCAB_SIZES,
            **metadata,
        }
        (path / "structure_adapter.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2)
        )

    @classmethod
    def load_arm(
        cls, directory: str | Path, model_config: Any, *, freeze: bool = True
    ) -> "AdditiveStructureAdapter":
        path = Path(directory)
        metadata = json.loads((path / "structure_adapter.json").read_text())
        hidden_size = hidden_size_from_config(model_config)
        if metadata["hidden_size"] != hidden_size:
            raise ValueError("saved adapter/model hidden-size mismatch")
        adapter = cls(hidden_size)
        adapter.load_state_dict(
            torch.load(path / "structure_adapter.pt", map_location="cpu", weights_only=True)
        )
        return adapter.freeze() if freeze else adapter


@dataclass
class StructuralForward:
    """The only allowed model call for structure-enabled experiment phases."""

    model: nn.Module
    adapter: AdditiveStructureAdapter
    scale: float = 1.0

    def _embedding_layer(self) -> nn.Module:
        model = self.model
        if hasattr(model, "get_input_embeddings"):
            layer = model.get_input_embeddings()
            if layer is not None:
                return layer
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "get_input_embeddings"):
            layer = base.get_input_embeddings()
            if layer is not None:
                return layer
        raise RuntimeError("cannot locate input embedding layer")

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        layout: LayoutMetadata,
        **kwargs: Any,
    ) -> Any:
        """Embed ids, add sidecar delta, and call the model without ``input_ids``."""
        if "inputs_embeds" in kwargs or "input_ids" in kwargs:
            raise ValueError("pass token ids only through StructuralForward.input_ids")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if layout.row.shape != input_ids.shape:
            raise ValueError(
                f"layout shape {tuple(layout.row.shape)} != ids {tuple(input_ids.shape)}"
            )
        embedding_layer = self._embedding_layer()
        token_embeddings = embedding_layer(input_ids)
        delta = self.adapter(layout.to(token_embeddings.device), scale=self.scale)
        delta = delta.to(dtype=token_embeddings.dtype)
        return self.model(inputs_embeds=token_embeddings + delta, **kwargs)


def assert_scale_zero_equivalence(
    model: nn.Module,
    adapter: AdditiveStructureAdapter,
    input_ids: torch.Tensor,
    layout: LayoutMetadata,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, float]:
    """Compare native and sidecar-bypass logits and shifted cross entropy."""
    model.eval()
    with torch.no_grad():
        native = model(input_ids=input_ids, return_dict=True, use_cache=False).logits.float()
        wrapped = StructuralForward(model, adapter, scale=0.0)(
            input_ids=input_ids,
            layout=layout,
            return_dict=True,
            use_cache=False,
        ).logits.float()
    max_logit = float((native - wrapped).abs().max())
    native_nll = torch.nn.functional.cross_entropy(
        native[:, :-1].reshape(-1, native.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )
    wrapped_nll = torch.nn.functional.cross_entropy(
        wrapped[:, :-1].reshape(-1, wrapped.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )
    nll_error = float((native_nll - wrapped_nll).abs())
    if not torch.allclose(native, wrapped, atol=atol, rtol=rtol):
        raise AssertionError(
            f"scale-zero logits differ: max={max_logit}, atol={atol}, rtol={rtol}"
        )
    if nll_error > atol:
        raise AssertionError(f"scale-zero NLL differs by {nll_error}")
    return {"max_abs_logit_error": max_logit, "abs_nll_error": nll_error}
