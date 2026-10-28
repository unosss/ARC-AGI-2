from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path

import pytest
import torch

from arc_solver.qwen2d_experiment import (
    INTERPRETATION_FAMILIES,
    PERCEPTION_FAMILIES,
    SYNTHETIC_FAMILIES,
    make_staged_corpora,
    make_synthetic_rows,
)
from arc_solver.qwen_staged_adapters import (
    EXPECTED_TARGETS_PER_LAYER,
    activate_named_adapters,
    adapter_tensor_digest,
    assert_frozen_digest_unchanged,
    contiguous_thirds,
    freeze_all_except_adapter,
    make_lora_config,
    num_hidden_layers_from_config,
    validate_expected_target_count,
    validate_stage_bands,
    validate_trainable_adapter_band,
)
from kaggle_runs.qwen_staged_adapters_runpod import run_experiment, worker


def test_balanced_contiguous_thirds_cover_model():
    early, middle, late = contiguous_thirds(10)
    assert early.layer_indices == (0, 1, 2, 3)
    assert middle.layer_indices == (4, 5, 6)
    assert late.layer_indices == (7, 8, 9)
    assert validate_stage_bands((early, middle, late), num_hidden_layers=10)
    with pytest.raises(ValueError, match="at least three"):
        contiguous_thirds(2)


def test_layer_count_supports_nested_and_model_configs():
    assert num_hidden_layers_from_config({"text_config": {"num_hidden_layers": 12}}) == 12
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=36))
    assert num_hidden_layers_from_config(model) == 36


def _tiny_staged_model():
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.Qwen3Config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = transformers.Qwen3ForCausalLM(config)
    early, middle, late = contiguous_thirds(3)
    model = peft.get_peft_model(
        model,
        make_lora_config(
            early, r=2, lora_alpha=2, use_rslora=True
        ),
        adapter_name="perception",
    )
    model.add_adapter(
        "interpretation",
        make_lora_config(middle, r=2, lora_alpha=2, use_rslora=True),
    )
    model.add_adapter(
        "task",
        make_lora_config(late, r=2, lora_alpha=2, use_rslora=True),
    )
    return model, (early, middle, late)


def test_disjoint_named_adapters_have_exact_projection_counts():
    model, stages = _tiny_staged_model()
    for name, stage in zip(("perception", "interpretation", "task"), stages):
        assert validate_expected_target_count(model, name, stage) == (
            EXPECTED_TARGETS_PER_LAYER * len(stage.layer_indices)
        )

    activate_named_adapters(model, ["perception", "interpretation", "task"])
    names = freeze_all_except_adapter(model, "task")
    assert names
    assert all(".task." in name for name in names)
    validate_trainable_adapter_band(model, "task", stages[2])


def test_training_one_stage_preserves_frozen_adapter_digest():
    model, stages = _tiny_staged_model()
    activate_named_adapters(model, ["perception", "interpretation"])
    frozen = adapter_tensor_digest(model, "perception")
    freeze_all_except_adapter(model, "interpretation")
    validate_trainable_adapter_band(model, "interpretation", stages[1])

    ids = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    optimizer.zero_grad()
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    assert_frozen_digest_unchanged(model, "perception", frozen)


def test_task_adapter_can_intentionally_collide_with_perception_band():
    model, stages = _tiny_staged_model()
    model.add_adapter(
        "task_collision",
        make_lora_config(stages[0], r=2, lora_alpha=2, use_rslora=True),
    )
    activate_named_adapters(model, ["perception", "interpretation", "task_collision"])
    names = freeze_all_except_adapter(model, "task_collision")
    assert names and all(".task_collision." in name for name in names)
    validate_trainable_adapter_band(model, "task_collision", stages[0])
    assert validate_expected_target_count(
        model, "task_collision", stages[0]
    ) == EXPECTED_TARGETS_PER_LAYER


def test_stage_family_partition_and_arc_loo_assignment_are_complete():
    assert set(PERCEPTION_FAMILIES).isdisjoint(INTERPRETATION_FAMILIES)
    assert set(PERCEPTION_FAMILIES) | set(INTERPRETATION_FAMILIES) == set(
        SYNTHETIC_FAMILIES
    )
    train, probes = make_synthetic_rows(
        seeds_per_family=3, seed=34034, heldout_modulus=3
    )
    arc_loo = {
        "row_id": "arc:deadbeef:0",
        "source": "arc_training_loo",
        "family": "arc_loo",
        "source_task_id": "deadbeef",
        "seed": 0,
        "train": [{"input": [[1]], "output": [[2]]}],
        "query": [[3]],
        "target": [[4]],
    }
    corpora, manifest = make_staged_corpora(
        [*train, arc_loo],
        probes,
        perception_token_budget=200,
        interpretation_token_budget=200,
    )
    assert manifest["known_families_assigned_exactly_once"]
    assert all(
        row["family"] != "arc_loo" for row in corpora["perception"]["train"]
    )
    expected_interpretation_sources = sum(
        row["family"] in set(INTERPRETATION_FAMILIES) | {"arc_loo"}
        for row in [*train, arc_loo]
    )
    assert (
        manifest["stages"]["interpretation"]["source_train_row_count"]
        == expected_interpretation_sources
    )
    assert sum(
        row["loss_token_limit"] for row in corpora["perception"]["train"]
    ) == 200
    assert sum(
        row["loss_token_limit"] for row in corpora["interpretation"]["train"]
    ) == 200


def _production_config():
    path = (
        Path(__file__).parents[1]
        / "kaggle_runs"
        / "qwen_staged_adapters_runpod"
        / "config.json"
    )
    return json.loads(path.read_text())


def test_runner_config_and_control_selection_are_frozen():
    config = _production_config()
    run_experiment.validate_config(config)
    assert config["stage_budgets"] == {
        "perception": 600_000,
        "interpretation": 900_000,
    }
    ids = [f"{index:08x}" for index in range(40)]
    expected = sorted(
        ids,
        key=lambda task_id: (
            run_experiment.digest_bytes(
                f"experiment-034-control:{task_id}".encode()
            ),
            task_id,
        ),
    )[:8]
    assert len(expected) == 8
    assert len(set(expected)) == 8
    assert run_experiment.HARD_EXPLORATORY_IDS == [
        "195c6913",
        "221dfab4",
        "20a9e565",
        "13e47133",
        "142ca369",
        "269e22fb",
        "16b78196",
        "21897d95",
    ]


def test_candidate_checkpoint_is_atomic_and_rejects_stale_data(tmp_path):
    path = tmp_path / "candidate.json"
    worker.write_candidate(path, {"ranked_candidates": []}, "fingerprint")
    assert worker.existing_candidate(path, "fingerprint")
    path.write_text('{"ranked_candidates":[1]}\n')
    with pytest.raises(RuntimeError, match="digest mismatch"):
        worker.existing_candidate(path, "fingerprint")


def test_module_digest_supports_scalar_bfloat16_state():
    module = torch.nn.Module()
    module.register_buffer("scale", torch.tensor(1.0, dtype=torch.bfloat16))
    first = worker.module_digest(module)
    module.scale.fill_(2.0)
    assert worker.module_digest(module) != first


def test_optimizer_state_offload_preserves_adam_moments():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    before = {
        name: value.clone()
        for name, value in optimizer.state[parameter].items()
        if torch.is_tensor(value)
    }
    worker.optimizer_state_to(optimizer, "cpu")
    assert all(
        torch.equal(optimizer.state[parameter][name], value)
        for name, value in before.items()
    )


def test_optimizer_offload_is_disabled_on_a6000_memory():
    assert worker.optimizer_offload_for_total_memory(24 * 1024**3)
    assert not worker.optimizer_offload_for_total_memory(48 * 1024**3)


def test_task_rows_use_pre_tokenized_collation():
    task = {
        "train": [
            {"input": [[1]], "output": [[2]]},
            {"input": [[3]], "output": [[4]]},
        ],
        "test": [{"input": [[5]]}],
    }
    config = {
        "seed": 34,
        "global_max_sequence_length": 512,
        "task_local": {
            "train_color_permutations": 1,
            "train_geometries": 1,
        },
    }
    row = next(worker.training_rows(task, "deadbeef", config))
    assert set(row) == {"input_ids"}
    batch = worker.collate_token_rows(
        [row], arm="rowcol_component", shuffle_seed=34
    )
    assert batch["input_ids"].shape == batch["labels"].shape


def test_runtime_projection_does_not_scale_to_unreachable_node_cap():
    config = _production_config()
    manifest = {
        "primary_ids": [f"p{index}" for index in range(32)],
        "control_ids": [f"c{index}" for index in range(8)],
    }
    projected = worker.project_full_runtime(
        config,
        manifest,
        primary_item_count=50,
        control_item_count=12,
        global_sample_seconds=2302.0,
        global_sample_tokens_per_stage=32768,
        task_sample_total_seconds=50.0,
        dfs_sample_seconds=4.3,
    )
    config["dfs"]["max_nodes_per_view"] *= 100
    assert worker.project_full_runtime(
        config,
        manifest,
        primary_item_count=50,
        control_item_count=12,
        global_sample_seconds=2302.0,
        global_sample_tokens_per_stage=32768,
        task_sample_total_seconds=50.0,
        dfs_sample_seconds=4.3,
    ) == projected
    assert projected < 200_000


def test_worker_launch_never_receives_solution_path(tmp_path, monkeypatch):
    captured = {}

    def fake_call(command, **kwargs):
        captured["command"] = command
        return 0

    monkeypatch.setattr(run_experiment.subprocess, "call", fake_call)
    config = {"wall_cap_seconds": 10}
    run_experiment.run_worker(
        "pilot",
        config,
        tmp_path,
        tmp_path / "challenges.json",
        tmp_path / "model",
        tmp_path / "work",
    )
    command = captured["command"]
    assert "--solutions" not in command
    assert all("solution" not in str(value).lower() for value in command)


def test_calibration_gate_applies_exact_25_percent_margin(tmp_path):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "result.json").write_text(
        json.dumps({"projected_full_seconds": 80.0})
    )
    config = {
        "calibration": {
            "safety_margin": 0.25,
            "maximum_projected_seconds": 100,
        }
    }
    gate = run_experiment.calibration_gate(config, tmp_path)
    assert gate["projected_with_margin_seconds"] == 100.0
    assert gate["passed"] is True


def test_label_barrier_refuses_missing_primary_candidates(tmp_path):
    config = {
        "arms": ["staged_late", "mixed_late", "staged_early_collision"],
        "control_arms": ["reversed_bands", "shuffled_interpretation"],
    }
    manifest = {
        "primary_ids": ["00000001"],
        "control_ids": ["00000001"],
    }
    with pytest.raises(RuntimeError, match="candidate missing"):
        run_experiment.hash_all_candidates(config, manifest, tmp_path)


def test_prepare_smoke_is_label_blind_and_writes_exact_budgets(tmp_path):
    repo = Path(__file__).parents[1]
    challenges = {
        f"{index:08x}": {
            "train": [{"input": [[index % 10]], "output": [[(index + 1) % 10]]}],
            "test": [{"input": [[(index + 2) % 10]]}],
        }
        for index in range(180)
    }
    for task_id in ("03560426", "045e512c"):
        challenges[task_id] = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
    challenge_path = tmp_path / "challenges.json"
    challenge_path.write_text(json.dumps(challenges))
    config = _production_config()
    config["corpus"]["synthetic_seeds_per_family"] = 2
    config["corpus"]["synthetic_holdout_modulus"] = 2
    config["stage_budgets"] = {"perception": 200, "interpretation": 300}
    manifest = run_experiment.prepare(config, repo, challenge_path, tmp_path / "work")
    assert len(manifest["primary_ids"]) == 32
    assert len(manifest["control_ids"]) == 8
    assert all(
        "output" not in test
        for task_id in manifest["primary_ids"]
        for test in challenges[task_id]["test"]
    )
    corpus_manifests = json.loads(
        (tmp_path / "work" / "manifests" / "corpus_manifests.json").read_text()
    )
    assert corpus_manifests["staged_perception"]["loss_tokens"] == 200
    assert corpus_manifests["staged_interpretation"]["loss_tokens"] == 300
    assert corpus_manifests["shuffled_interpretation"]["loss_tokens"] == 300
