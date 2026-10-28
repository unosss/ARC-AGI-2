from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "029_qwen3vl_global_task_lora"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"exp029_{name}", EXP / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_module("core")


def test_fixed_design_budget_and_tasks():
    config = json.loads((EXP / "config.json").read_text())
    assert config["arms"] == ["base_task_lora", "perception_task_lora"]
    assert len(config["task_ids"]) == 16
    assert config["optimizer_steps"] == 128
    assert config["color_permutations"] == 16
    assert config["geometric_transforms"] == 8
    assert config["task_lora_rank"] == 64
    assert config["num_beams"] == config["num_return_sequences"] == 2
    assert config["max_new_tokens"] == 3072
    assert config["max_sequence_length"] == 4352


def test_schedule_covers_every_color_geometry_pair_once():
    config = json.loads((EXP / "config.json").read_text())
    transforms = [
        core.step_transform(config, step)
        for step in range(config["optimizer_steps"])
    ]
    keys = [(tuple(colors), geometry) for colors, geometry in transforms]
    assert len(keys) == len(set(keys)) == 128
    assert {geometry for _, geometry in keys} == set(range(8))
    assert all(colors[0] == 0 for colors, _ in transforms)


def test_d4_is_complete_and_color_mapping_is_bijective():
    asymmetric = [[0, 1, 2], [3, 4, 5]]
    variants = [core.d4(asymmetric, index) for index in range(8)]
    assert len({json.dumps(value) for value in variants}) == 8
    for index in range(8):
        assert core.d4(core.d4(asymmetric, index), index) == asymmetric or index in {
            1,
            3,
        }
    for permutation in core.color_permutations(2902):
        assert permutation[0] == 0
        assert sorted(permutation) == list(range(10))


def test_parser_is_strict_and_retains_two_unique_grids():
    text = (
        'noise [[1,2],[3,4]] then [[1,2],[3,4]] '
        'and [[0]] but reject [[10]] and [[1],[2,3]]'
    )
    assert core.parse_grids(text) == [[[1, 2], [3, 4]], [[0]]]
    assert core.parse_grids("[[true]]") == []
    assert core.parse_grids("[]") == []


def test_supervised_encoding_masks_prompt_and_rejects_any_truncation():
    class Processor:
        def apply_chat_template(self, payload, add_generation_prompt, **_kwargs):
            length = 3 if add_generation_prompt else 7
            return {
                "input_ids": torch.arange(length).unsqueeze(0),
                "attention_mask": torch.ones((1, length), dtype=torch.long),
            }

    encoded = core.supervised_encoding(
        Processor(), [[1]], [[2]], cell_size=4, max_length=7
    )
    assert encoded["input_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    assert encoded["labels"].tolist() == [[-100, -100, -100, 3, 4, 5, 6]]
    with pytest.raises(RuntimeError, match="exceeds the fixed limit"):
        core.supervised_encoding(
            Processor(), [[1]], [[2]], cell_size=4, max_length=5
        )
    with pytest.raises(RuntimeError, match="leaves no supervised token"):
        core.supervised_encoding(
            Processor(), [[1]], [[2]], cell_size=4, max_length=3
        )


def test_resume_rejects_stale_fingerprint(tmp_path):
    worker = load_module("worker")
    run_dir = tmp_path / "arm" / "task"
    run_dir.mkdir(parents=True)
    candidate = run_dir / "candidate.json"
    candidate.write_text("{}")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": {"version": 1},
                "candidate_sha256": core.file_sha256(candidate),
                "task_adapter_sha256": {},
            }
        )
    )
    with pytest.raises(RuntimeError, match="stale run fingerprint"):
        worker.verify_resume(run_dir, {"version": 2})


def test_solution_file_is_not_touched_before_all_candidates_exist(tmp_path):
    finalize = load_module("finalize")
    config = json.loads((EXP / "config.json").read_text())
    challenge = tmp_path / "challenge.json"
    challenge.write_text("{}")
    config["challenge_sha256"] = core.file_sha256(challenge)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="official evaluation is sealed"):
        finalize.load_sealed_candidates(config_path, challenge, tmp_path / "runs")


def test_worker_enforces_adapter_isolation_and_pinned_counts_in_source():
    source = (EXP / "worker.py").read_text()
    assert "EXPECTED_TASK_TRAINABLE = 132_120_576" in source
    assert 'parameter.requires_grad = "lora_" in name and ".task." in name' in source
    assert 'model.base_model.set_adapter(["global", "task"])' in source
    assert "global_before != global_after" in source
    assert 'selected_adapters=["task"]' in source
    assert '"model": model_source' in source
    assert '"runtime": runtime' in source


def test_preflight_and_pilot_fail_closed():
    preflight = (EXP / "preflight.py").read_text()
    run = (EXP / "run.py").read_text()
    probe = (EXP / "adapter_probe.py").read_text()
    assert '"passed": truncated == 0' in preflight
    assert "supervised examples exceed the fixed sequence budget" in preflight
    assert "passed_cross_arm_initialization_parity" in run
    assert "train_task and" in probe
