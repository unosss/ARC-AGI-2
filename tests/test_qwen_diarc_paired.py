from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from kaggle_runs.qwen_diarc_paired_runpod import protocol, run_experiment, worker


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "kaggle_runs" / "qwen_diarc_paired_runpod" / "config.json"


def config():
    return json.loads(CONFIG_PATH.read_text())


def test_production_config_is_frozen_and_calibration_only_scales_down():
    value = config()
    protocol.validate_config(value)
    assert value["task_local"]["steps"] == 16 * 8 == 128
    assert value["task_local"]["decode_color_permutations"] == 1
    assert value["task_local"]["decode_geometries"] == 8
    assert value["task_local"]["rescore_views"] == 8
    assert value["dfs"]["batch_size"] == 4
    assert value["dfs"]["max_nodes_per_view"] == 4096
    assert value["dfs"]["max_model_forward_tokens_per_view"] == 12288
    assert value["dfs"]["max_candidates_per_view"] == 128

    reduced = protocol.mode_config(value, "calibrate")
    assert reduced["task_local"]["steps"] == 2
    assert reduced["dfs"]["max_nodes_per_view"] == 64
    assert value["task_local"]["steps"] == 128

    changed = copy.deepcopy(value)
    changed["task_local"]["steps"] = 127
    with pytest.raises(ValueError, match="task_local.steps"):
        protocol.validate_config(changed)


def test_label_guard_rejects_test_outputs_and_worker_command_has_no_solution():
    with pytest.raises(ValueError, match="refuses"):
        protocol.reject_test_outputs(
            {"195c6913": {"train": [], "test": [{"input": [[0]], "output": [[1]]}]}}
        )
    command = run_experiment.worker_command(
        arm="base",
        mode="smoke",
        config_path=CONFIG_PATH,
        challenges_path=Path("/tmp/challenges.json"),
        task_ids=["195c6913"],
        work=Path("/tmp/work"),
        model_dir=None,
        adapter=None,
    )
    assert "--solutions" not in command
    assert all("solution" not in argument.lower() for argument in command)


def _payload(arm: str, candidates):
    nll = {"mean_sum_nll": 1.0, "items": []}
    return {
        "schema_version": 1,
        "experiment": 36,
        "arm": arm,
        "task_id": "195c6913",
        "ranked_candidates": candidates,
        "pre_task_loo_nll": nll,
        "post_task_loo_nll": nll,
        "timing": {
            "load_seconds": 0.0,
            "train_seconds": 0.0,
            "decode_seconds": 0.0,
        },
        "peak_memory_bytes": 0,
    }


def test_hash_barrier_requires_both_immutable_candidates(tmp_path):
    task_id = "195c6913"
    base = protocol.candidate_path(tmp_path, "base", task_id)
    protocol.seal_candidate(base, _payload("base", [[{"grid": [[1]], "votes": 1}]]))
    with pytest.raises(RuntimeError, match="candidate hash missing"):
        protocol.verify_hash_barrier(tmp_path, [task_id])

    diarc = protocol.candidate_path(tmp_path, "diarc", task_id)
    protocol.seal_candidate(diarc, _payload("diarc", [[{"grid": [[1]], "votes": 1}]]))
    hashes = protocol.verify_hash_barrier(tmp_path, [task_id])
    assert set(hashes) == {"base/195c6913", "diarc/195c6913"}
    base.write_text(base.read_text() + " ")
    with pytest.raises(RuntimeError, match="candidate hash mismatch"):
        protocol.verify_hash_barrier(tmp_path, [task_id])


def test_parent_grading_reports_misses_accuracy_votes_and_deltas(tmp_path):
    task_id = "195c6913"
    base_candidates = [[
        {"grid": [[0, 0]], "score": 2.0, "votes": 3, "rank": 1},
        {"grid": [[1, 0]], "score": 1.0, "votes": 1, "rank": 2},
    ]]
    diarc_candidates = [[
        {"grid": [[1, 1]], "score": 2.0, "votes": 4, "rank": 1},
        {"grid": [[0, 0]], "score": 1.0, "votes": 2, "rank": 2},
    ]]
    protocol.seal_candidate(
        protocol.candidate_path(tmp_path, "base", task_id),
        _payload("base", base_candidates),
    )
    protocol.seal_candidate(
        protocol.candidate_path(tmp_path, "diarc", task_id),
        _payload("diarc", diarc_candidates),
    )
    report = protocol.grade(tmp_path, [task_id], {task_id: [[[1, 1]]]})
    base_item = report["arms"]["base"]["tasks"][0]["items"][0]
    diarc_item = report["arms"]["diarc"]["tasks"][0]["items"][0]
    assert base_item["generation_miss"] is True
    assert base_item["best_top2_cell_accuracy"] == 0.5
    assert diarc_item["top1_exact"] is True
    assert diarc_item["truth_rank"] == 1
    assert diarc_item["ranked_votes"][0] == {"rank": 1, "votes": 4}
    assert report["pairwise_diarc_minus_base"]["item_top2_exact"] == 1


def test_smoke_worker_is_cpu_only_and_seals_candidates(tmp_path):
    task_id = "195c6913"
    challenges = {
        task_id: {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
    }
    worker.run_arm("base", "smoke", config(), challenges, [task_id], tmp_path)
    payload = json.loads(
        protocol.candidate_path(tmp_path, "base", task_id).read_text()
    )
    assert payload["peak_memory_bytes"] == 0
    assert payload["ranked_candidates"][0][0]["rank"] == 1


def _tiny_composed_model():
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    tiny = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    value = config()
    tiny_task = copy.deepcopy(value)
    tiny_task["task_local"]["rank"] = 2
    tiny_task["task_local"]["alpha"] = 2
    diarc_config = worker.task_lora_config(tiny_task)
    diarc_config.use_rslora = False
    model = peft.get_peft_model(tiny, diarc_config, adapter_name="diarc")
    model.add_adapter("task", worker.task_lora_config(tiny_task))
    model.base_model.set_adapter(["diarc", "task"])
    return model


def test_tiny_qwen_active_adapters_and_trainable_isolation():
    model = _tiny_composed_model()
    names = worker.freeze_all_except_task(model)
    assert set(worker.active_adapter_names(model)) == {"diarc", "task"}
    assert names
    assert all(".task." in name for name in names)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".diarc." in name
    )
    ids = torch.tensor([[1, 2, 3]])
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
