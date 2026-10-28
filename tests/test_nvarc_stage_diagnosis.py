import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from arc_solver.nvarc_stage_diagnosis import (
    DISTRACTOR_BBOXES,
    PREFIX_8_ROWS_PLUS_1_CELL,
    PREFIX_ROWS,
    SOURCE_BBOX,
    aggregate_report,
    assert_oracle_source,
    build_oracle_conditions,
    build_training_bundle,
    canonical_reply_tokens,
    forced_prefix_8_rows_plus_1_cell,
    forced_prefix_tokens,
)
from kaggle_runs.nvarc_stage_diagnosis_l4.build_kernel import build_archive


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "kaggle_runs" / "nvarc_stage_diagnosis_l4"
CHALLENGES = (
    ROOT
    / "kaggle_runs"
    / "nvarc_single_pass_l4"
    / "output_v007"
    / "experiment_015"
    / "solver_input"
    / "challenges.json"
)
LABELS = (
    ROOT
    / "kaggle_runs"
    / "nvarc_single_pass_l4"
    / "output_v007"
    / "experiment_015"
    / "parent_finalizer_only"
    / "labels.json"
)


def _real_bundle_and_truth():
    challenges = json.loads(CHALLENGES.read_text())
    labels = json.loads(LABELS.read_text())
    return build_training_bundle(challenges), labels["0934a4d8"][0]


def test_real_source_bbox_transposes_to_truth_and_bad_bbox_aborts():
    bundle, truth = _real_bundle_and_truth()
    test_input = bundle["0934a4d8"]["test"][0]["input"]
    crop = assert_oracle_source(test_input, truth)
    assert SOURCE_BBOX == (0, 3, 14, 23)
    assert len(crop) == 3
    assert all(len(row) == 9 for row in crop)
    assert [list(row) for row in zip(*crop)] == truth
    with pytest.raises(AssertionError):
        assert_oracle_source(test_input, truth, (0, 3, 13, 22))


def test_training_bundle_has_demos_but_no_test_label():
    raw = json.loads(CHALLENGES.read_text())
    bundle = build_training_bundle(raw)
    task = bundle["0934a4d8"]
    assert len(task["train"]) == 4
    assert all(set(pair) == {"input", "output"} for pair in task["train"])
    assert all(set(item) == {"input"} for item in task["test"])
    assert "output" not in json.dumps(task["test"])


def test_oracle_conditions_are_grid_only_tagged_and_have_equal_shape_crops():
    bundle, truth = _real_bundle_and_truth()
    payload = build_oracle_conditions(bundle["0934a4d8"], truth)
    assert payload["source_bbox"] == [0, 3, 14, 23]
    assert tuple(payload["prefix_rows"]) == PREFIX_ROWS
    assert len(payload["correspondence_crops"]) == len(DISTRACTOR_BBOXES) == 5
    assert sum(crop["is_true_source"] for crop in payload["correspondence_crops"]) == 1
    assert all(
        len(crop["test_input"]) == 3
        and all(len(row) == 9 for row in crop["test_input"])
        for crop in payload["correspondence_crops"]
    )
    probe = payload["conditions"]["transpose_demonstration_probe"]
    assert probe["oracle_operation_hint"] is True
    assert probe["fair_solver_condition"] is False
    assert len(probe["train"]) == len(bundle["0934a4d8"]["train"]) + 2
    assert all(
        pair["output"] == [list(row) for row in zip(*pair["input"])]
        for pair in probe["train"][-2:]
    )
    expected = {
        "transpose_hint_full",
        "transpose_hint_mask",
        *(f"transpose_hint_crop_{index}" for index in range(5)),
    }
    assert expected <= payload["conditions"].keys()
    assert payload["conditions"]["transpose_hint_full"]["test_input"] == bundle[
        "0934a4d8"
    ]["test"][0]["input"]
    mask = payload["conditions"]["transpose_hint_mask"]["test_input"]
    assert len(mask) == 30 and all(len(row) == 30 for row in mask)
    for index, crop in enumerate(payload["correspondence_crops"]):
        hinted = payload["conditions"][f"transpose_hint_crop_{index}"]
        assert hinted["test_input"] == crop["test_input"]
        assert hinted["is_true_source"] == crop["is_true_source"]
        assert hinted["train"][-2:] == probe["train"][-2:]


def test_v2_oracle_hints_do_not_embed_truth_as_a_demonstration():
    bundle, truth = _real_bundle_and_truth()
    payload = build_oracle_conditions(bundle["0934a4d8"], truth)
    original_train = bundle["0934a4d8"]["train"]
    for name, condition in payload["conditions"].items():
        if name.startswith("transpose_"):
            assert condition["train"][: len(original_train)] == original_train
            assert all(pair["output"] != truth for pair in condition["train"])
    assert all("truth" not in condition for condition in payload["conditions"].values())


def test_prefix_tokens_and_canonical_mapping_preserve_rows_and_boundaries():
    truth = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    tokens, mapping = canonical_reply_tokens(truth)
    assert tokens == [1, 2, 3, 10, 4, 5, 6, 10, 7, 8, 9, 15]
    assert mapping[0] == {"kind": "cell", "row": 0, "column": 0}
    assert mapping[3] == {"kind": "newline", "after_row": 0}
    assert mapping[-1] == {"kind": "eos"}
    assert forced_prefix_tokens(truth, 0) == []
    assert forced_prefix_tokens(truth, 1) == [1, 2, 3, 10]
    assert forced_prefix_tokens(truth, 2) == [1, 2, 3, 10, 4, 5, 6, 10]
    nine_rows = truth * 3
    plus_one = forced_prefix_8_rows_plus_1_cell(nine_rows)
    assert plus_one == forced_prefix_tokens(nine_rows, 8) + [nine_rows[8][0]]
    assert len(plus_one) == 33
    with pytest.raises(ValueError):
        forced_prefix_tokens(truth, 4)


def _lifecycle():
    return [
        {"event": name}
        for name in (
            "ordinary_training_completed",
            "ordinary_generation_completed",
            "ordinary_worker_exited",
            "labels_opened",
            "oracle_payload_written",
            "oracle_training_completed",
            "oracle_payload_opened",
            "oracle_diagnostics_completed",
            "oracle_worker_exited",
        )
    ]


def test_report_aggregation_separates_questions_and_audits_lifecycle():
    ordinary = {
        "baseline": {"candidates": []},
        "baseline_exp015_comparable": {"candidates": []},
        "training": {"steps": 128},
    }
    crop = lambda true, found: {
        "is_true_source": true,
        "truth_present": found,
    }
    oracle = {
        "execution": {"can_transpose_oracle_crop_with_hint": True},
        "conditions": {
            "oracle_source_crop": {"truth_present": True},
            "oracle_source_mask": {"truth_present": False},
            "transpose_hint_full": {"truth_present": False},
            "transpose_hint_mask": {"truth_present": True},
        },
        "correspondence": {
            "crops": [crop(True, True), crop(False, False), crop(False, False)]
        },
        "hinted_correspondence": {
            "crops": [crop(True, True), crop(False, False), crop(False, False)]
        },
        "nll": {"base": {}, "rslora": {}},
        "forced_prefix": {
            "conditions": [
                {"prefix_rows": 0, "exact_completion": False},
                {"prefix_rows": 3, "exact_completion": True},
                {"prefix_rows": 5, "exact_completion": True},
            ]
        },
        "training": {"steps": 128},
    }
    report = aggregate_report({}, ordinary, oracle, _lifecycle())
    assert set(report["causal_questions"]) == {
        "execution",
        "localization",
        "correspondence",
        "hinted_correspondence",
        "model_likelihood",
        "decoding",
    }
    assert report["causal_questions"]["correspondence"][
        "only_true_source_yields_truth"
    ]
    assert report["causal_questions"]["decoding"]["smallest_exact_prefix_rows"] == 3
    assert report["schema_version"] == report["run_version"] == 2
    assert report["label_lifecycle"]["ordinary_labels_available"] is False
    with pytest.raises(AssertionError):
        aggregate_report({}, ordinary, oracle, _lifecycle()[1:])


def test_config_metadata_and_kernel_sources_are_pinned():
    config = json.loads((RUN_DIR / "config.json").read_text())
    metadata = json.loads((RUN_DIR / "kernel-metadata.json").read_text())
    assert config["task_id"] == "0934a4d8"
    assert config["test_index"] == 0
    assert config["n_train_aug"] == 16
    assert config["train_geometries"] == 8
    assert config["epochs"] == 1
    assert config["optimizer_steps"] == 128
    assert config["lora"] == {
        "type": "RSLoRA",
        "rank": 256,
        "alpha": 32,
        "dropout": 0.0,
    }
    assert config["lora_seed"] == 42
    assert config["train_aug_seed"] == 1
    assert config["prefix_rows"] == [0, 1, 2, 3, 4, 5, 8]
    assert config["report_filename"] == "stage_diagnosis_report_v2.json"
    assert config["baseline_v2_eval_augmentations"] == 2
    assert config["baseline_v2_dfs_window_seconds"] == 540
    assert config["baseline_v2_budget_seconds"] == 1200
    assert config["source_bbox_half_open"] == [0, 3, 14, 23]
    assert metadata["machine_shape"] == "NvidiaL4"
    assert metadata["enable_internet"] is False
    assert metadata["model_sources"] == [
        "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
    ]
    assert metadata["docker_image"].endswith(
        "sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
    )


def test_archive_and_runner_enforce_parent_label_boundary():
    with zipfile.ZipFile(BytesIO(base64.b64decode(build_archive()))) as archive:
        names = set(archive.namelist())
    assert "stage_worker.py" in names
    assert "nvarc_upstream_solver.py" in names
    assert "arc_solver/nvarc_stage_diagnosis.py" in names
    runner = (RUN_DIR / "runner_template.py").read_text()
    ordinary = runner.index('run_phase(\n        "ordinary"')
    solutions = runner.index('find_arc_file("evaluation_solutions")')
    oracle = runner.index('run_phase(\n        "oracle"')
    assert ordinary < solutions < oracle
    ordinary_block = runner[ordinary:solutions]
    assert "oracle_payload=" not in ordinary_block
    worker = (RUN_DIR / "stage_worker.py").read_text()
    oracle_branch = worker[worker.index('elif PHASE == "oracle"') :]
    assert oracle_branch.index('"oracle_training_completed"') < oracle_branch.index(
        "ARC_STAGE_ORACLE_PAYLOAD"
    )
    after_payload = oracle_branch[oracle_branch.index("ARC_STAGE_ORACLE_PAYLOAD") :]
    assert "trainer.train()" not in after_payload
    assert PREFIX_8_ROWS_PLUS_1_CELL in worker
    assert "decode_exp015_augmented_baseline" in worker
    assert '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"' in runner
