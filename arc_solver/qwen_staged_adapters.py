"""Utilities for assigning named Qwen LoRA adapters to layer stages.

The helpers in this module deliberately avoid importing PEFT at module import
time.  This keeps data preparation and unit tests usable in environments where
the optional training dependency is not installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import re
from typing import Any


STAGE_NAMES = ("early", "middle", "late")
"""Canonical order of the three transformer-depth stages."""

QWEN_ATTENTION_PROJECTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
QWEN_MLP_PROJECTION_TARGETS = ("gate_proj", "up_proj", "down_proj")
QWEN_PROJECTION_TARGETS = (
    *QWEN_ATTENTION_PROJECTION_TARGETS,
    *QWEN_MLP_PROJECTION_TARGETS,
)
"""The seven per-layer Qwen projections eligible for staged LoRA.

Token embeddings and the language-model head are intentionally absent.
"""

EXPECTED_TARGETS_PER_LAYER = len(QWEN_PROJECTION_TARGETS)
QWEN_LAYER_TARGETS = QWEN_PROJECTION_TARGETS
"""Compatibility name for the complete per-layer target tuple."""

_LAYER_COMPONENT = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")
_ADAPTER_COMPONENTS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)
_NESTED_CONFIG_NAMES = (
    "text_config",
    "language_config",
    "llm_config",
    "decoder_config",
)


@dataclass(frozen=True)
class StageSpec:
    """A named, non-empty, ordered set of transformer layer indices."""

    name: str
    layer_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stage name must be a non-empty string")
        indices = tuple(self.layer_indices)
        if not indices:
            raise ValueError(f"stage {self.name!r} must contain at least one layer")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("layer indices must be integers")
        if any(index < 0 for index in indices):
            raise ValueError("layer indices must be non-negative")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("layer indices must be strictly increasing and unique")
        if indices != tuple(range(indices[0], indices[-1] + 1)):
            raise ValueError("a stage must be a contiguous layer band")
        object.__setattr__(self, "layer_indices", indices)

    @property
    def start(self) -> int:
        """Return the inclusive first layer index."""

        return self.layer_indices[0]

    @property
    def stop(self) -> int:
        """Return the exclusive final layer boundary."""

        return self.layer_indices[-1] + 1


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def num_hidden_layers_from_config(config_or_model: Any) -> int:
    """Extract ``num_hidden_layers`` from a model or common nested configs.

    Hugging Face text-only configs normally expose the value directly, while
    multimodal configs commonly place it under ``text_config`` or
    ``language_config``.  Mapping-shaped serialized configs are also accepted.
    """

    if config_or_model is None:
        raise ValueError("model config cannot be None")

    root = getattr(config_or_model, "config", config_or_model)
    pending = [root]
    seen: set[int] = set()
    while pending:
        config = pending.pop(0)
        identity = id(config)
        if identity in seen:
            continue
        seen.add(identity)

        value = _config_value(config, "num_hidden_layers")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("num_hidden_layers must be an integer")
            if value <= 0:
                raise ValueError("num_hidden_layers must be positive")
            return value

        for name in _NESTED_CONFIG_NAMES:
            nested = _config_value(config, name)
            if nested is not None:
                pending.append(nested)

    raise ValueError(
        "could not find num_hidden_layers on the model config or a supported "
        f"nested config ({', '.join(_NESTED_CONFIG_NAMES)})"
    )


def contiguous_thirds(num_hidden_layers: int) -> tuple[StageSpec, StageSpec, StageSpec]:
    """Split layers into balanced early/middle/late contiguous bands.

    Remainder layers are assigned in canonical stage order.  For example,
    ten layers produce bands of lengths 4, 3, and 3.  At least three layers
    are required so every stage is meaningful.
    """

    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int):
        raise TypeError("num_hidden_layers must be an integer")
    if num_hidden_layers < len(STAGE_NAMES):
        raise ValueError("at least three hidden layers are required")

    width, remainder = divmod(num_hidden_layers, len(STAGE_NAMES))
    specs: list[StageSpec] = []
    start = 0
    for position, name in enumerate(STAGE_NAMES):
        stop = start + width + (position < remainder)
        specs.append(StageSpec(name, tuple(range(start, stop))))
        start = stop
    result = tuple(specs)
    validate_stage_bands(result, num_hidden_layers=num_hidden_layers)
    return result  # type: ignore[return-value]


def stage_specs_from_config(config_or_model: Any) -> tuple[StageSpec, StageSpec, StageSpec]:
    """Build canonical stage specs from a model or model config."""

    return contiguous_thirds(num_hidden_layers_from_config(config_or_model))


def _coerce_stage_spec(stage_spec: StageSpec | Mapping[str, Any]) -> StageSpec:
    if isinstance(stage_spec, StageSpec):
        return stage_spec
    if not isinstance(stage_spec, Mapping):
        raise TypeError("stage_spec must be a StageSpec or mapping")
    name = stage_spec.get("name", stage_spec.get("stage"))
    indices = stage_spec.get("layer_indices", stage_spec.get("layers"))
    if indices is None:
        raise ValueError("stage_spec mapping must contain layer_indices or layers")
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Iterable):
        raise TypeError("stage layer indices must be an iterable of integers")
    return StageSpec(name, tuple(indices))


def make_lora_config(
    stage_spec: StageSpec | Mapping[str, Any],
    *,
    r: int,
    lora_alpha: int | float,
    lora_dropout: float = 0.0,
    use_rslora: bool = False,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    """Create a PEFT ``LoraConfig`` limited to one transformer stage.

    PEFT is imported lazily.  ``layers_to_transform`` is used with the seven
    projection suffixes so embeddings and ``lm_head`` cannot be selected.
    """

    spec = _coerce_stage_spec(stage_spec)
    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise ValueError("r must be a positive integer")
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, (int, float)):
        raise TypeError("lora_alpha must be numeric")
    if lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    if isinstance(lora_dropout, bool) or not isinstance(lora_dropout, (int, float)):
        raise TypeError("lora_dropout must be numeric")
    if not 0.0 <= lora_dropout <= 1.0:
        raise ValueError("lora_dropout must be between zero and one")
    if not isinstance(use_rslora, bool):
        raise TypeError("use_rslora must be a boolean")

    from peft import LoraConfig  # Imported only when adapter construction is used.

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=float(lora_dropout),
        bias=bias,
        use_rslora=use_rslora,
        task_type=task_type,
        target_modules=list(QWEN_PROJECTION_TARGETS),
        layers_to_transform=list(spec.layer_indices),
        layers_pattern="layers",
    )


def _validate_adapter_name(adapter_name: str) -> None:
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ValueError("adapter name must be a non-empty string")
    if "." in adapter_name:
        raise ValueError("adapter names containing '.' are not supported by PyTorch")


def activate_named_adapters(model: Any, adapter_names: Sequence[str]) -> None:
    """Activate named adapters through the PEFT 0.17.1-compatible API."""

    if isinstance(adapter_names, (str, bytes)):
        raise TypeError("adapter_names must be a sequence, not a single string")
    names = list(adapter_names)
    if not names:
        raise ValueError("at least one adapter must be activated")
    for name in names:
        _validate_adapter_name(name)
    if len(set(names)) != len(names):
        raise ValueError("adapter names must be unique")
    base_model = getattr(model, "base_model", None)
    setter = getattr(base_model, "set_adapter", None)
    if not callable(setter):
        raise TypeError("model.base_model.set_adapter is not callable")
    setter(names)


def _is_adapter_parameter(name: str, adapter_name: str) -> bool:
    marker = f".{adapter_name}."
    return marker in name and any(
        f".{component}.{adapter_name}." in f".{name}"
        for component in _ADAPTER_COMPONENTS
    )


def freeze_all_except_adapter(model: Any, adapter_name: str) -> tuple[str, ...]:
    """Freeze every parameter except tensors belonging to ``adapter_name``."""

    _validate_adapter_name(adapter_name)
    matched: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = _is_adapter_parameter(name, adapter_name)
        parameter.requires_grad_(trainable)
        if trainable:
            matched.append(name)
    if not matched:
        raise ValueError(f"no LoRA parameters found for adapter {adapter_name!r}")
    return tuple(sorted(matched))


def trainable_parameter_names(model: Any) -> tuple[str, ...]:
    """Return trainable parameter names in deterministic order."""

    return tuple(
        sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    )


def layer_index_from_name(name: str) -> int:
    """Extract a transformer layer index from a Qwen/PEFT qualified name."""

    if not isinstance(name, str) or not name:
        raise ValueError("module or parameter name must be a non-empty string")
    matches = {int(match) for match in _LAYER_COMPONENT.findall(name)}
    if not matches:
        raise ValueError(f"no transformer layer index found in {name!r}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous transformer layer indices in {name!r}")
    return next(iter(matches))


def _adapter_base_modules(model: Any, adapter_name: str) -> dict[str, int]:
    _validate_adapter_name(adapter_name)
    modules: dict[str, int] = {}
    for name, _ in model.named_parameters():
        if not _is_adapter_parameter(name, adapter_name):
            continue
        split_at = next(
            (
                name.index(f".{component}.{adapter_name}.")
                for component in _ADAPTER_COMPONENTS
                if f".{component}.{adapter_name}." in name
            ),
            None,
        )
        if split_at is None:
            continue
        base_name = name[:split_at]
        modules[base_name] = layer_index_from_name(base_name)
    if not modules:
        raise ValueError(f"no adapted base modules found for adapter {adapter_name!r}")
    return modules


def resolve_adapted_layer_indices(model: Any, adapter_name: str) -> tuple[int, ...]:
    """Resolve unique adapted base-model layer indices from injected tensors."""

    return tuple(sorted(set(_adapter_base_modules(model, adapter_name).values())))


def validate_trainable_adapter_band(
    model: Any,
    adapter_name: str,
    stage_spec: StageSpec | Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate that only the named adapter and intended layer band train."""

    spec = _coerce_stage_spec(stage_spec)
    names = trainable_parameter_names(model)
    if not names:
        raise ValueError("model has no trainable parameters")
    unexpected_adapter = [name for name in names if not _is_adapter_parameter(name, adapter_name)]
    if unexpected_adapter:
        raise ValueError(
            "trainable parameters outside adapter "
            f"{adapter_name!r}: {unexpected_adapter[:3]}"
        )
    intended = set(spec.layer_indices)
    outside = [name for name in names if layer_index_from_name(name) not in intended]
    if outside:
        raise ValueError(
            f"adapter {adapter_name!r} has trainable parameters outside "
            f"stage {spec.name!r}: {outside[:3]}"
        )
    observed = {layer_index_from_name(name) for name in names}
    missing = intended - observed
    if missing:
        raise ValueError(
            f"adapter {adapter_name!r} has no trainable tensors in layers "
            f"{sorted(missing)}"
        )
    return names


def validate_stage_bands(
    stage_specs: Sequence[StageSpec | Mapping[str, Any]],
    *,
    num_hidden_layers: int | None = None,
    require_complete: bool = True,
) -> tuple[StageSpec, ...]:
    """Validate unique names, disjoint bands, and optional full-model coverage."""

    specs = tuple(_coerce_stage_spec(spec) for spec in stage_specs)
    if not specs:
        raise ValueError("at least one stage spec is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("stage names must be unique")

    owner: dict[int, str] = {}
    for spec in specs:
        for index in spec.layer_indices:
            if index in owner:
                raise ValueError(
                    f"layer {index} is shared by stages {owner[index]!r} "
                    f"and {spec.name!r}"
                )
            owner[index] = spec.name

    if num_hidden_layers is not None:
        if (
            isinstance(num_hidden_layers, bool)
            or not isinstance(num_hidden_layers, int)
            or num_hidden_layers <= 0
        ):
            raise ValueError("num_hidden_layers must be a positive integer")
        out_of_range = sorted(index for index in owner if index >= num_hidden_layers)
        if out_of_range:
            raise ValueError(f"stage layers out of range: {out_of_range}")
        if require_complete:
            missing = sorted(set(range(num_hidden_layers)) - set(owner))
            if missing:
                raise ValueError(f"stage bands do not cover layers: {missing}")
    return specs


def validate_expected_target_count(
    model: Any,
    adapter_name: str,
    stage_spec: StageSpec | Mapping[str, Any],
) -> int:
    """Validate all seven Qwen projections are adapted in every stage layer."""

    spec = _coerce_stage_spec(stage_spec)
    modules = _adapter_base_modules(model, adapter_name)
    observed: set[tuple[int, str]] = set()
    unexpected: list[str] = []
    for module_name, layer_index in modules.items():
        target = module_name.rsplit(".", 1)[-1]
        if target not in QWEN_PROJECTION_TARGETS:
            unexpected.append(module_name)
        else:
            observed.add((layer_index, target))
    if unexpected:
        raise ValueError(
            "adapter includes non-Qwen projection targets: "
            f"{unexpected[:3]}"
        )

    expected = {
        (layer_index, target)
        for layer_index in spec.layer_indices
        for target in QWEN_PROJECTION_TARGETS
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicate_count = len(modules) - len(observed)
    if missing or extra or duplicate_count:
        raise ValueError(
            f"adapter target mismatch: expected {len(expected)}, observed "
            f"{len(modules)} base modules ({len(observed)} unique layer/target "
            f"pairs); missing={missing[:3]}, extra={extra[:3]}, "
            f"duplicates={duplicate_count}"
        )
    return len(observed)


def adapter_tensor_digest(model: Any, adapter_name: str) -> str:
    """Return a deterministic SHA-256 digest of one adapter's tensor state."""

    _validate_adapter_name(adapter_name)
    tensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _is_adapter_parameter(name, adapter_name)
    ]
    if not tensors:
        raise ValueError(f"no tensors found for adapter {adapter_name!r}")

    checksum = hashlib.sha256()
    for name, parameter in sorted(tensors, key=lambda item: item[0]):
        tensor = parameter.detach().cpu().contiguous()
        metadata = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode()
        checksum.update(len(metadata).to_bytes(8, "big"))
        checksum.update(metadata)
        try:
            payload = tensor.view(-1).view(_torch_uint8()).numpy().tobytes()
        except (RuntimeError, TypeError) as error:
            raise TypeError(f"cannot digest adapter tensor {name!r}") from error
        checksum.update(len(payload).to_bytes(8, "big"))
        checksum.update(payload)
    return checksum.hexdigest()


def _torch_uint8():
    """Import torch lazily for dtype-preserving tensor serialization."""

    import torch

    return torch.uint8


def assert_frozen_digest_unchanged(
    model: Any,
    adapter_name: str,
    expected_digest: str,
) -> str:
    """Assert that a frozen adapter still has its previously recorded digest."""

    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ):
        raise ValueError("expected_digest must be a lowercase SHA-256 hex digest")
    actual = adapter_tensor_digest(model, adapter_name)
    if actual != expected_digest:
        raise AssertionError(
            f"frozen adapter {adapter_name!r} changed: "
            f"expected {expected_digest}, observed {actual}"
        )
    return actual


# Concise compatibility aliases for callers that use extraction/factory wording.
extract_num_hidden_layers = num_hidden_layers_from_config
lora_config_for_stage = make_lora_config
build_lora_config = make_lora_config
set_active_adapters = activate_named_adapters
split_layer_stages = contiguous_thirds
tensor_digest = adapter_tensor_digest
validate_disjoint_bands = validate_stage_bands
validate_trainable_band = validate_trainable_adapter_band


__all__ = [
    "EXPECTED_TARGETS_PER_LAYER",
    "QWEN_ATTENTION_PROJECTION_TARGETS",
    "QWEN_MLP_PROJECTION_TARGETS",
    "QWEN_LAYER_TARGETS",
    "QWEN_PROJECTION_TARGETS",
    "STAGE_NAMES",
    "StageSpec",
    "activate_named_adapters",
    "adapter_tensor_digest",
    "assert_frozen_digest_unchanged",
    "build_lora_config",
    "contiguous_thirds",
    "extract_num_hidden_layers",
    "freeze_all_except_adapter",
    "layer_index_from_name",
    "lora_config_for_stage",
    "make_lora_config",
    "num_hidden_layers_from_config",
    "resolve_adapted_layer_indices",
    "set_active_adapters",
    "split_layer_stages",
    "stage_specs_from_config",
    "trainable_parameter_names",
    "tensor_digest",
    "validate_disjoint_bands",
    "validate_expected_target_count",
    "validate_stage_bands",
    "validate_trainable_adapter_band",
    "validate_trainable_band",
]
