from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from arc_solver.perception import describe_grid, score_description


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "027_qwen3vl_grid_perception"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_config_is_fixed_and_valid():
    runner = load_module("runner")
    config = json.loads((EXP / "run_config.json").read_text())
    runner.validate_config(config)
    assert config["model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert len(config["task_ids"]) == 10
    assert config["modalities"] == ["text", "image", "both"]
    assert config["max_new_tokens"] == 3072
    assert config["do_sample"] is False


def test_messages_do_not_leak_json_into_image_arm():
    runner = load_module("runner")
    grid = [[0, 1], [2, 3]]
    image = runner.build_messages(grid, "image", 12)
    text = runner.build_messages(grid, "text", 12)
    both = runner.build_messages(grid, "both", 12)
    assert [part["type"] for part in image[1]["content"]] == ["image", "text"]
    assert [part["type"] for part in text[1]["content"]] == ["text"]
    assert [part["type"] for part in both[1]["content"]] == ["image", "text"]
    assert "Grid as JSON" not in image[1]["content"][-1]["text"]
    assert "Grid as JSON" in text[1]["content"][-1]["text"]
    assert "Grid as JSON" in both[1]["content"][-1]["text"]
    assert '["0,0","0,1","1,0","1,1","2,2"]' in image[1]["content"][-1]["text"]


def score(overall: float, reading: float | None = None):
    value = overall if reading is None else reading
    return {
        "structured": True,
        "reading": value,
        "objects": overall,
        "relations": overall,
        "overall": overall,
        "exact": False,
    }


def record(task: str, modality: str, overall: float, reading: float | None = None):
    return {
        "task_id": task,
        "role": "input",
        "modality": modality,
        "score": score(overall, reading),
        "elapsed_seconds": 1.0,
        "output_tokens": 10,
    }


def test_paired_decision_requires_mean_reading_and_win_count():
    analyze = load_module("analyze")
    records = []
    for index in range(20):
        task = f"{index:02d}"
        text_value = 0.2
        both_value = 0.8 if index < 12 else 0.2
        records.extend(
            [
                record(task, "text", text_value),
                record(task, "image", 0.1),
                record(task, "both", both_value),
            ]
        )
    report = analyze.summarize(records)
    assert report["paired_both_vs_text"]["wins"] == 12
    assert report["paired_both_vs_text"]["losses"] == 0
    assert report["decision_rule_passed"] is True

    for item in records:
        if item["modality"] == "both":
            item["score"]["reading"] = 0.0
    assert analyze.summarize(records)["decision_rule_passed"] is False


def test_analysis_rejects_incomplete_triplet():
    analyze = load_module("analyze")
    records = [record("a", "text", 0.2), record("a", "both", 0.8)]
    try:
        analyze.summarize(records)
    except ValueError as exc:
        assert "expected 60 responses" in str(exc)
    else:
        raise AssertionError("incomplete triplet was accepted")


def test_scoring_treats_malformed_mapping_fields_as_incorrect():
    grid = [[1, 0, 2], [0, 1, 0], [2, 0, 1]]
    expected = describe_grid(grid)
    expected["sampled_cells"] = {"0,0": 1}
    expected["sampled_rows"] = {"0": grid[0]}
    prediction = {
        **expected,
        "color_counts": [],
        "sampled_cells": [],
        "sampled_rows": [],
        "symmetry": [],
    }

    result = score_description(prediction, expected)

    assert result["structured"] is True
    assert result["reading"] < 1.0
    assert result["relations"] < 1.0
    assert result["exact"] is False


def test_fixed_probe_represents_out_of_bounds_cell_as_null():
    runner = load_module("runner")

    description = runner.fixed_probe_description([[1, 2, 3], [4, 5, 6]])

    assert description["sampled_cells"]["2,2"] is None
    assert description["sampled_rows"] == {
        "0": [1, 2, 3],
        "1": [4, 5, 6],
    }
