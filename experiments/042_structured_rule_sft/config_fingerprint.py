"""Separate data-selection provenance from train-time model settings."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CORPUS_CONFIG_KEYS = (
    "split_seed",
    "demo_count",
    "validation_rules",
    "test_rules",
    "train_budgets",
    "rule_reserve",
    "source_split_thresholds",
    "program_timeout_seconds",
    "program_memory_bytes",
)


def corpus_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in CORPUS_CONFIG_KEYS}


def corpus_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        corpus_config_payload(config),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
