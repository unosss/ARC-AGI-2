from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from arc_solver.qwen2d_experiment import (
    corrupt_refiner_draft,
    make_refiner_rows,
    refiner_corpus_audit,
    refiner_prompt_payload,
)
from arc_solver.qwen2d_refiner import (
    MAX_GRID_SIZE,
    OUTSIDE,
    UNKNOWN,
    Qwen2DRefiner,
    RefinerConfig,
    batch_grids_to_canvas,
    canvas_to_grid,
    decode_stages,
    refiner_loss,
)
from kaggle_runs.qwen_iterative_refiner_runpod import run_experiment, worker


def tiny_refiner() -> Qwen2DRefiner:
    torch.manual_seed(35)
    return Qwen2DRefiner(
        RefinerConfig(
            hidden_size=16,
            model_dim=16,
            latent_dim=16,
            latent_slots=4,
            num_heads=4,
            ff_multiplier=1,
            num_stages=3,
        )
    )


def test_canvas_round_trip_and_shape_padding():
    canvases, heights, widths = batch_grids_to_canvas([[[1, 2], [3, 4]], [[5]]])
    assert canvases.shape == (2, MAX_GRID_SIZE, MAX_GRID_SIZE)
    assert heights.tolist() == [2, 1]
    assert widths.tolist() == [2, 1]
    assert torch.all(canvases[1, 1:] == OUTSIDE)
    canvases[0, 0, 1] = UNKNOWN
    assert canvas_to_grid(canvases[0], height=2, width=2) == [[1, 0], [3, 4]]


def test_refiner_is_parallel_three_stage_and_backpropagates():
    model = tiny_refiner()
    hidden = torch.randn(2, 7, 16)
    mask = torch.tensor([[1] * 7, [1] * 5 + [0] * 2], dtype=torch.bool)
    draft = torch.tensor([[[1, 10], [0, 2]], [[4, 4], [4, 4]]])
    output = model(hidden, mask, draft)
    assert len(output.stages) == 3
    assert output.trajectory.shape == (2, 3, 30, 30)
    assert output.context_latents.shape == (2, 4, 16)
    losses = refiner_loss(output, [[[1, 3], [0, 2]], [[4], [4]]])
    losses.loss.backward()
    assert torch.isfinite(losses.loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )
    decoded = decode_stages(output, draft_canvas=draft)
    assert len(decoded) == 3 and all(len(stage) == 2 for stage in decoded)
    assert all(
        1 <= len(grid) <= 30 and 1 <= len(grid[0]) <= 30
        for stage in decoded
        for grid in stage
    )


def test_corruption_is_deterministic_and_prompt_is_label_blind():
    target = [[1, 1, 0], [0, 2, 2]]
    first = corrupt_refiner_draft(target, seed=123, severity=0.6)
    second = corrupt_refiner_draft(target, seed=123, severity=0.6)
    assert first == second
    assert first[0] != target
    row = {
        "row_id": "synthetic:x",
        "train": [{"input": [[0]], "output": [[1]]}],
        "query": [[2]],
        "target": target,
    }
    refined = make_refiner_rows([row], seed=35, variants_per_row=3)
    assert len(refined) == 3
    assert len({item["row_id"] for item in refined}) == 3
    prompt = refiner_prompt_payload(refined[0])
    assert set(prompt) == {"train", "query", "draft"}
    assert "target" not in prompt
    assert refiner_corpus_audit(
        refined, global_task_ids=["global"], heldout_task_ids=["held"]
    )["passed"]


def test_refiner_corpus_audit_rejects_partition_overlap():
    row = {
        "row_id": "x",
        "train": [],
        "query": [[0]],
        "draft": [[UNKNOWN]],
        "target": [[1]],
    }
    with pytest.raises(AssertionError, match="audit failed"):
        refiner_corpus_audit(
            [row], global_task_ids=["same"], heldout_task_ids=["same"]
        )


def test_loo_drafts_are_generated_before_held_label_is_attached(monkeypatch):
    observed = []

    def fake_generate(_model, _structure, blind_task, _task_id, _config):
        observed.append(blind_task)
        assert all("output" not in item for item in blind_task["test"])
        return {"ranked_candidates": [[{"grid": [[7]]}]]}

    monkeypatch.setattr(worker, "generate_drafts", fake_generate)
    task = {
        "train": [
            {"input": [[1]], "output": [[2]]},
            {"input": [[3]], "output": [[4]]},
        ],
        "test": [{"input": [[5]]}],
    }
    rows = worker.loo_rows(None, None, task, "task", {})
    assert len(observed) == 2
    assert [row["target"] for row in rows] == [[[2]], [[4]]]
    assert rows[0]["train"] == [task["train"][1]]


def test_loo_search_miss_uses_label_blind_input_fallback(monkeypatch):
    monkeypatch.setattr(
        worker,
        "generate_drafts",
        lambda *_args, **_kwargs: {"ranked_candidates": [[]]},
    )
    task = {
        "train": [{"input": [[3, 0]], "output": [[8]]}],
        "test": [{"input": [[5]]}],
    }
    rows = worker.loo_rows(None, None, task, "task", {})
    assert rows[0]["draft"] == [[3, 0]]
    assert rows[0]["draft_source"] == "input_fallback"
    assert rows[0]["target"] == [[8]]


def test_worker_label_guard_and_frozen_config():
    worker.reject_test_outputs({"x": {"test": [{"input": [[0]]}]}})
    with pytest.raises(ValueError, match="test outputs"):
        worker.reject_test_outputs(
            {"x": {"test": [{"input": [[0]], "output": [[1]]}]}}
        )
    config_path = (
        Path(__file__).parents[1]
        / "kaggle_runs"
        / "qwen_iterative_refiner_runpod"
        / "config.json"
    )
    config = json.loads(config_path.read_text())
    run_experiment.validate_config(config)
    assert config["arms"] == ["baseline", "global_refiner", "task_refiner"]
    assert config["refiner"]["num_stages"] == 3
    assert config["task_local"]["steps"] > 0


def test_exploratory_grading_reports_each_refinement_stage(tmp_path):
    config = {"arms": ["baseline", "global_refiner", "task_refiner"]}
    task_id = "abc12345"
    truth = [[1, 2]]
    for arm in config["arms"]:
        path = tmp_path / "work" / "arms" / arm / "candidates" / f"{task_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ranked_candidates": [[{"grid": truth, "score": 0.0}]],
            "stage_trajectories": [
                [
                    {
                        "draft_rank": 0,
                        "G0": [[1, 0]],
                        **(
                            {"G1": [[1, 1]], "G2": truth, "G3": truth}
                            if arm != "baseline"
                            else {}
                        ),
                    }
                ]
            ],
        }
        path.write_text(json.dumps(payload))
        marker = path.with_suffix(".sha256")
        marker.write_text(
            json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        )
    solutions = tmp_path / "solutions.json"
    solutions.write_text(json.dumps({task_id: [truth]}))
    report = run_experiment.grade_exploratory_subset(
        config,
        [task_id],
        solutions,
        tmp_path / "work",
        tmp_path / "output",
    )
    assert report["arms"]["global_refiner"]["tasks"][0]["items"][0][
        "stage_best"
    ]["G2"]["exact"]
    assert report["arms"]["baseline"]["task_top2_exact"] == 1
