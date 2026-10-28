from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
RUN = ROOT / "kaggle_runs" / "qwen_2d_grid_adapter_l4"
SPEC = importlib.util.spec_from_file_location(
    "orchestration_safety", RUN / "orchestration_safety.py"
)
assert SPEC and SPEC.loader
safety = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(safety)


@pytest.mark.parametrize("directory_name", ["Transformers", "transformers", "TRANSFORMERS"])
def test_exact_model_dir_accepts_transformers_mount_casing(tmp_path: Path, directory_name: str):
    model = (
        tmp_path
        / "dataset"
        / "qwen3_4b_grids15_sft139"
        / directory_name
        / "bfloat16"
        / "1"
    )
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    assert safety.find_exact_model_dir(tmp_path) == model


def test_exact_model_dir_requires_exactly_one_version(tmp_path: Path):
    for dataset in ("first", "second"):
        model = (
            tmp_path
            / dataset
            / "qwen3_4b_grids15_sft139"
            / "Transformers"
            / "bfloat16"
            / "1"
        )
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="exact pinned model source unavailable"):
        safety.find_exact_model_dir(tmp_path)


def test_full_runtime_config_is_strictly_validated():
    config = json.loads((RUN / "full-config.json").read_text())
    safety.validate_runtime_config(config)
    config["arms"] = list(reversed(config["arms"]))
    with pytest.raises(ValueError, match="main arms differ"):
        safety.validate_runtime_config(config)


def test_smoke_runtime_and_preregistration_are_compatible():
    config = json.loads((RUN / "smoke-config.json").read_text())
    prereg = json.loads(
        (
            ROOT
            / "experiments"
            / "021_qwen_2d_grid_adapter"
            / "preregistered_manifest.json"
        ).read_text()
    )
    safety.validate_runtime_config(config)
    safety.validate_preregistered_manifest(prereg, config)


def test_preregistered_manifest_matches_runtime_config():
    config = json.loads((RUN / "full-config.json").read_text())
    prereg = json.loads(
        (
            ROOT
            / "experiments"
            / "021_qwen_2d_grid_adapter"
            / "preregistered_manifest.json"
        ).read_text()
    )
    safety.validate_preregistered_manifest(prereg, config)
    prereg["negative_control_task_count"] += 1
    with pytest.raises(ValueError, match="negative_control_task_count"):
        safety.validate_preregistered_manifest(prereg, config)


def test_report_schema_document_is_valid():
    method = safety.validate_report_schema_document(
        ROOT / "experiments" / "021_qwen_2d_grid_adapter" / "report_schema.json"
    )
    assert method in {"internal_strict", "jsonschema+internal_strict"}
