from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import torch

from arc_solver.perception import (
    fixed_probe_description,
    fixed_probe_prompt,
    validate_grid,
)


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "028_qwen3vl_perception_lora"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_preserves_result_bearing_budget():
    config = json.loads((EXP / "config.json").read_text())
    assert config["model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert config["train_grid_count"] == 4096
    assert config["synthetic_eval_grid_count"] == 24
    assert config["arms"] == ["language_lora", "multimodal_lora"]
    assert config["lora_rank"] == 16
    assert config["epochs"] == 1
    assert config["pilot_rows"] == 16
    assert config["max_projected_total_training_seconds"] == 28800
    assert config["max_peak_allocated_bytes"] == 22 * 1024**3
    assert config["gradient_accumulation_steps"] == 8
    assert config["max_sequence_length"] == 4096
    assert config["max_target_characters"] == 8000
    assert config["train_corpus_sha256"] == (
        "3932f75d52e1630a56faa2a6ab5e0407de71e46e828d550d879791519822b6e8"
    )
    assert config["synthetic_eval_sha256"] == (
        "e25cd7595549cb4ee3c05f6994744416a46e699b256c5def6722b7803ba35d7f"
    )
    assert config["experiment_027_base_results_sha256"] == (
        "45ed56becf46dc1bdd39636fce75883906453f439a68e2587dd497a06cc12437"
    )


def test_runpod_setup_supports_pep668_images():
    setup = (EXP / "runpod_setup.sh").read_text()
    assert "pip install --upgrade --break-system-packages" in setup


def test_synthetic_rows_are_deterministic_balanced_and_exact():
    synth = load_module("synth_data")
    left = synth.generate_rows(seed=1234, count=24, split="test")
    right = synth.generate_rows(seed=1234, count=24, split="test")
    assert left == right
    assert len({row["grid_sha256"] for row in left}) == 24
    assert Counter(row["family"] for row in left) == {
        "random": 6,
        "components": 6,
        "geometry": 6,
        "structure": 6,
    }
    assert Counter(row["modality"] for row in left) == {
        "text": 8,
        "image": 8,
        "both": 8,
    }
    for row in left:
        validate_grid(row["grid"])
        assert row["target"] == fixed_probe_description(row["grid"])


def test_training_and_evaluation_seeds_do_not_collide():
    synth = load_module("synth_data")
    train = synth.generate_rows(seed=2801001, count=64, split="train")
    evaluation = synth.generate_rows(
        seed=2802001, count=24, split="synthetic_eval"
    )
    assert not (
        {row["grid_sha256"] for row in train}
        & {row["grid_sha256"] for row in evaluation}
    )


def test_fixed_image_prompt_has_no_grid_values_or_content_selected_coordinates():
    first = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    second = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    first_prompt = fixed_probe_prompt(first, "image")
    second_prompt = fixed_probe_prompt(second, "image")
    assert first_prompt == second_prompt
    assert "Grid as JSON" not in first_prompt
    assert '["0,0","0,1","1,0","1,1","2,2"]' in first_prompt


def test_small_grid_uses_null_for_fixed_out_of_bounds_cell():
    target = fixed_probe_description([[1, 2, 3], [4, 5, 6]])
    assert target["sampled_cells"]["2,2"] is None


def test_lora_scopes_and_parameter_counts_are_pinned():
    train = load_module("train")
    assert train.TARGETS["language_lora"][1] == 252
    assert train.TARGETS["multimodal_lora"][1] == 356
    assert train.EXPECTED_TRAINABLE == {
        "language_lora": 33_030_144,
        "multimodal_lora": 40_271_872,
    }
    assert "model\\.visual" not in train.LANGUAGE_TARGETS
    assert "model\\.visual" in train.MULTIMODAL_TARGETS


def test_training_sequences_are_truncated_to_the_fixed_budget():
    train = load_module("train")

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            length = 5 if len(messages) == 3 else 2
            return {
                "input_ids": torch.arange(length).unsqueeze(0),
                "attention_mask": torch.ones((1, length), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2)),
            }

    dataset = train.PerceptionDataset(
        [{"id": "long", "modality": "text", "grid": [[1]], "target": {}}],
        Processor(),
        max_length=4,
    )
    item = dataset[0]
    assert item["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert item["attention_mask"].shape == (1, 4)
    assert item["labels"].tolist() == [[-100, -100, 2, 3]]
    assert item["pixel_values"].shape == (1, 3, 2, 2)


def test_official_evaluation_is_sealed_until_both_arms_complete(tmp_path):
    evaluate = load_module("evaluate")
    config = json.loads((EXP / "config.json").read_text())
    try:
        evaluate.verify_training_completions(
            config, tmp_path, "config-hash", "corpus-hash"
        )
    except RuntimeError as exc:
        assert "official evaluation is sealed" in str(exc)
    else:
        raise AssertionError("evaluation opened before both arms completed")


def test_pilot_completions_cannot_unseal_official_evaluation(tmp_path):
    evaluate = load_module("evaluate")
    config = json.loads((EXP / "config.json").read_text())
    for arm in config["arms"]:
        path = tmp_path / arm
        path.mkdir()
        (path / "completed.json").write_text(
            json.dumps(
                {
                    "config_sha256": "config-hash",
                    "corpus_sha256": "corpus-hash",
                    "experiment": 28,
                    "arm": arm,
                    "pilot": True,
                    "row_count": 1,
                    "model_revision": config["model_revision"],
                    "resolved_module_count": 1,
                    "trainable_parameters": 1,
                    "adapter_sha256": {},
                }
            )
        )
    try:
        evaluate.verify_training_completions(
            config, tmp_path, "config-hash", "corpus-hash"
        )
    except ValueError as exc:
        assert "completion mismatch: pilot" in str(exc)
    else:
        raise AssertionError("pilot completions unsealed official evaluation")


def test_analysis_recomputes_scores():
    analyze = load_module("analyze")
    expected = fixed_probe_description([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    records = [
        {
            "id": "fabricated",
            "expected": expected,
            "prediction": None,
            "score": {
                "structured": True,
                "reading": 1.0,
                "objects": 1.0,
                "relations": 1.0,
                "overall": 1.0,
                "exact": True,
            },
        }
    ]
    try:
        analyze.validate_scores(records)
    except ValueError as exc:
        assert "stored score mismatch" in str(exc)
    else:
        raise AssertionError("fabricated score was accepted")
