"""Experiment 014 integration profile for the public 76.9M MindsAI checkpoint."""

from __future__ import annotations

import copy
import os

from .codet5_small import MODEL_CONFIG as UPSTREAM_MODEL_CONFIG


MODEL_PATH = os.environ.get("PHASE1_MODEL_PATH", "mindware/arc-codet5-small")
DEFAULT_ACTIVE = True

condition = os.environ.get("PHASE1_CONDITION", "ttt_airv")
if condition not in {"baseline", "ttt", "ttt_airv"}:
    raise ValueError(f"unsupported PHASE1_CONDITION: {condition}")

MODEL_CONFIG = copy.deepcopy(UPSTREAM_MODEL_CONFIG)
MODEL_CONFIG["general"]["ttt_enabled"] = condition != "baseline"
MODEL_CONFIG["general"]["airv_enabled"] = condition == "ttt_airv"
MODEL_CONFIG["training"]["self_ensemble_count"] = 1
MODEL_CONFIG["training"]["mini_lr_grid"]["enabled"] = False
MODEL_CONFIG["training"]["lora"]["enabled"] = False
MODEL_CONFIG["training"]["save_intermediate_checkpoints"] = False
MODEL_CONFIG["training"]["save_optimizer"] = False
MODEL_CONFIG["ensembling"]["enable_self_ensemble"] = False
MODEL_CONFIG["ensembling"]["enable_model_ensemble"] = False
