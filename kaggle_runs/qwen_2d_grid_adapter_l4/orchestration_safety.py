"""Strict, dependency-optional validation for Experiment 021 orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


MAIN_ARMS = ["text_control", "rowcol", "rowcol_component"]
NEGATIVE_ARMS = ["shuffled_rowcol", "shuffled_component"]
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def find_exact_model_dir(input_root: Path) -> Path:
    """Resolve the pinned model, allowing only Transformers directory casing."""
    suffix = ("qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
    matches = sorted(
        path.parent
        for path in input_root.rglob("config.json")
        if len(path.parent.parts) >= len(suffix)
        and (
            path.parent.parts[-4],
            path.parent.parts[-3].lower(),
            path.parent.parts[-2],
            path.parent.parts[-1],
        )
        == suffix
    )
    if len(matches) != 1:
        raise RuntimeError(f"exact pinned model source unavailable: {matches}")
    return matches[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_runtime_config(config: Any) -> None:
    _require(isinstance(config, dict), "runtime config must be an object")
    required = {
        "profile", "experiment", "checkpoint", "seed", "split_seed", "arms",
        "negative_controls", "reserve_tasks", "primary_tasks",
        "minimum_primary_tasks", "negative_control_tasks", "replay_tasks",
        "replay_repetitions", "synthetic_seeds_per_family",
        "synthetic_holdout_modulus", "global_loss_token_budget_per_arm",
        "global_lora", "global_learning_rate", "structure_learning_rate",
        "global_max_sequence_length", "task_local", "dfs", "checkpoint_every",
        "gpu_count", "precision", "selection", "result_status",
    }
    _require(required <= set(config), f"runtime config missing {sorted(required - set(config))}")
    _require(
        set(config) <= required | {"throughput_calibration"},
        f"unknown runtime config fields {sorted(set(config) - required - {'throughput_calibration'})}",
    )
    _require(config["experiment"] == 21, "runtime config is not Experiment 021")
    _require(
        config["profile"]
        in {"full_preregistered", "cpu_and_gpu_smoke_not_for_inference"},
        "unknown runtime profile",
    )
    _require(config["arms"] == MAIN_ARMS, "main arms differ from preregistration")
    _require(config["negative_controls"] == NEGATIVE_ARMS, "negative controls differ from preregistration")
    _require(config["checkpoint"].endswith("/Transformers/bfloat16/1"), "checkpoint pin changed")
    for key in (
        "reserve_tasks", "primary_tasks", "minimum_primary_tasks",
        "negative_control_tasks", "replay_tasks", "replay_repetitions",
        "global_loss_token_budget_per_arm", "global_max_sequence_length", "gpu_count",
    ):
        _require(type(config[key]) is int and config[key] > 0, f"{key} must be a positive integer")
    _require(config["minimum_primary_tasks"] <= config["primary_tasks"], "minimum exceeds primary count")
    _require(config["primary_tasks"] <= config["reserve_tasks"], "primary count exceeds reserve")
    _require(config["negative_control_tasks"] <= config["minimum_primary_tasks"], "negative subset exceeds fallback")
    _require(config["replay_tasks"] <= config["minimum_primary_tasks"], "replay subset exceeds fallback")
    for section, keys in {
        "global_lora": {"type", "rank", "alpha", "dropout"},
        "task_local": {
            "type", "rank", "alpha", "dropout", "steps", "learning_rate",
            "train_color_permutations", "train_geometries",
            "decode_color_permutations", "decode_geometries", "rescore_views",
        },
        "dfs": {
            "max_nodes_per_view", "max_model_forward_tokens_per_view",
            "max_candidates_per_view", "max_new_tokens",
            "emergency_seconds_per_view", "min_prob",
        },
        "checkpoint_every": {"global_optimizer_steps", "completed_tasks"},
    }.items():
        _require(isinstance(config[section], dict), f"{section} must be an object")
        _require(keys <= set(config[section]), f"{section} missing {sorted(keys - set(config[section]))}")
        _require(set(config[section]) == keys, f"{section} has unregistered fields")
    _require(config["global_lora"]["type"] == "RSLoRA", "global adapter type changed")
    _require(config["task_local"]["type"] == "RSLoRA", "task adapter type changed")
    for section, keys in {
        "global_lora": ("rank", "alpha"),
        "task_local": (
            "rank", "alpha", "steps", "train_color_permutations",
            "train_geometries", "decode_color_permutations", "decode_geometries",
            "rescore_views",
        ),
        "dfs": (
            "max_nodes_per_view", "max_model_forward_tokens_per_view",
            "max_candidates_per_view", "max_new_tokens",
        ),
        "checkpoint_every": ("global_optimizer_steps", "completed_tasks"),
    }.items():
        for key in keys:
            _require(
                type(config[section][key]) is int and config[section][key] > 0,
                f"{section}.{key} must be a positive integer",
            )
    for value, name in (
        (config["global_learning_rate"], "global_learning_rate"),
        (config["structure_learning_rate"], "structure_learning_rate"),
        (config["global_lora"]["dropout"], "global_lora.dropout"),
        (config["task_local"]["dropout"], "task_local.dropout"),
        (config["task_local"]["learning_rate"], "task_local.learning_rate"),
        (config["dfs"]["emergency_seconds_per_view"], "dfs.emergency_seconds_per_view"),
    ):
        _require(type(value) in (int, float) and value >= 0, f"{name} must be nonnegative")
    _require(
        type(config["dfs"]["min_prob"]) in (int, float)
        and 0 < config["dfs"]["min_prob"] <= 1,
        "dfs.min_prob must be in (0, 1]",
    )
    _require(config["precision"] == "bfloat16", "precision changed")
    _require(config["selection"] == "upstream_score_kgmon_top2", "selection rule changed")
    if config["profile"] == "full_preregistered":
        _require(config["primary_tasks"] == 32, "full primary count must remain 32")
        _require(config["minimum_primary_tasks"] == 24, "full fallback count must remain 24")
        _require(config["negative_control_tasks"] == 12, "full negative count must remain 12")
        _require(config["replay_tasks"] == 4, "full replay count must remain 4")
        calibration = config.get("throughput_calibration")
        _require(isinstance(calibration, dict), "full profile requires throughput calibration")
        calibration_keys = {
            "global_loss_tokens", "task_local_steps", "dfs_nodes",
            "promote_to_32_if_projected_seconds_lte", "hard_run_cap_seconds",
        }
        _require(set(calibration) == calibration_keys, "throughput calibration fields changed")
        for key in calibration_keys:
            _require(
                type(calibration[key]) in (int, float) and calibration[key] > 0,
                f"throughput_calibration.{key} must be positive",
            )
    else:
        _require(
            config["result_status"] == "smoke_only_never_report_as_final",
            "smoke profile must remain non-inferential",
        )


def validate_preregistered_manifest(prereg: Any, config: dict[str, Any]) -> None:
    _require(isinstance(prereg, dict), "preregistered manifest must be an object")
    required = {
        "schema_version", "status", "experiment", "split_seed",
        "reviewed_10_excluded", "phase1_20_excluded", "reserved_task_count",
        "primary_task_count", "minimum_primary_if_throughput_projection_exceeds_11_5h",
        "negative_control_task_count", "replay_task_count",
        "selection_reads_test_outputs", "runtime_generated_fields",
    }
    _require(required <= set(prereg), f"preregistered manifest missing {sorted(required - set(prereg))}")
    _require(
        set(prereg) == required | {"task_selection", "strata"},
        "preregistered manifest fields changed",
    )
    _require(prereg["schema_version"] == 1 and prereg["experiment"] == 21, "wrong preregistration version")
    _require(prereg["status"] == "preregistered_before_gpu_run", "preregistration status changed")
    mappings = {"split_seed": "split_seed"}
    if config["profile"] == "full_preregistered":
        mappings.update(
            {
                "reserved_task_count": "reserve_tasks",
                "primary_task_count": "primary_tasks",
                "minimum_primary_if_throughput_projection_exceeds_11_5h": "minimum_primary_tasks",
                "negative_control_task_count": "negative_control_tasks",
                "replay_task_count": "replay_tasks",
            }
        )
    for manifest_key, config_key in mappings.items():
        _require(prereg[manifest_key] == config[config_key], f"{manifest_key} disagrees with runtime config")
    _require(prereg["selection_reads_test_outputs"] is False, "selection must remain label-blind")
    for key, expected in (("reviewed_10_excluded", 10), ("phase1_20_excluded", 20)):
        values = prereg[key]
        _require(isinstance(values, list) and len(values) == expected, f"{key} count changed")
        _require(
            len(set(values)) == len(values)
            and all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{8}", x) for x in values),
            f"{key} invalid",
        )
    _require(
        prereg["runtime_generated_fields"]
        == [
            "global_train_ids", "reserved_ids", "primary_ids",
            "negative_control_ids", "replay_ids", "manifest_sha256",
        ],
        "runtime-generated manifest fields changed",
    )


def validate_generated_manifest(manifest: Any, config: dict[str, Any]) -> None:
    _require(isinstance(manifest, dict), "generated manifest must be an object")
    required = {
        "schema_version", "seed", "excluded_ids", "global_train_ids", "reserved_ids",
        "primary_ids", "negative_control_ids", "replay_ids", "strata",
        "selection_uses_test_outputs", "manifest_sha256",
    }
    _require(required <= set(manifest), f"generated manifest missing {sorted(required - set(manifest))}")
    for key in ("excluded_ids", "global_train_ids", "reserved_ids", "primary_ids", "negative_control_ids", "replay_ids"):
        values = manifest[key]
        _require(isinstance(values, list) and len(values) == len(set(values)), f"{key} must be a unique list")
    _require(len(manifest["reserved_ids"]) == config["reserve_tasks"], "reserved count mismatch")
    _require(len(manifest["primary_ids"]) in {config["primary_tasks"], config["minimum_primary_tasks"]}, "primary count invalid")
    _require(len(manifest["negative_control_ids"]) == config["negative_control_tasks"], "negative count mismatch")
    _require(len(manifest["replay_ids"]) == config["replay_tasks"], "replay count mismatch")
    primary = set(manifest["primary_ids"])
    _require(set(manifest["negative_control_ids"]) <= primary, "negative controls are outside primary set")
    _require(set(manifest["replay_ids"]) <= primary, "replays are outside primary set")
    _require(manifest["selection_uses_test_outputs"] is False, "generated selection is not label-blind")
    _require(isinstance(manifest["manifest_sha256"], str) and SHA256.fullmatch(manifest["manifest_sha256"]) is not None, "invalid manifest hash")


def _internal_report_validation(report: Any) -> None:
    _require(isinstance(report, dict), "report must be an object")
    required = {
        "schema_version", "config_sha256", "manifest_sha256", "arms", "contrasts",
        "negative_controls", "negative_control_contrasts", "causal_interpretation", "replays",
    }
    _require(required <= set(report), f"report missing {sorted(required - set(report))}")
    _require(report["schema_version"] == 1, "wrong report schema version")
    for key in ("config_sha256", "manifest_sha256"):
        _require(isinstance(report[key], str) and SHA256.fullmatch(report[key]) is not None, f"invalid {key}")
    _require(set(report["arms"]) == set(MAIN_ARMS), "report main arms mismatch")
    _require(set(report["negative_controls"]) == set(NEGATIVE_ARMS), "report negative arms mismatch")
    _require(set(report["replays"]) == set(MAIN_ARMS), "report replay arms mismatch")
    for arm, arm_report in {**report["arms"], **report["negative_controls"]}.items():
        _require({"probe_metrics", "tasks", "aggregate"} <= set(arm_report), f"{arm} report incomplete")
        _require(arm_report["aggregate"]["task_count"] == len(arm_report["tasks"]), f"{arm} task aggregate mismatch")
        for task in arm_report["tasks"]:
            _require(isinstance(task.get("items"), list) and task["items"], f"{arm}/{task.get('task_id')} has no items")
    _require(isinstance(report["causal_interpretation"].get("supported"), bool), "causal signal must be boolean")


def validate_final_report(report: Any, schema_path: Path) -> str:
    """Validate with jsonschema when installed, always followed by strict checks."""
    method = "internal_strict"
    try:
        import jsonschema
    except ImportError:
        pass
    else:
        schema = json.loads(schema_path.read_text())
        jsonschema.Draft202012Validator(schema).validate(report)
        method = "jsonschema+internal_strict"
    _internal_report_validation(report)
    return method


def validate_report_schema_document(schema_path: Path) -> str:
    schema = json.loads(schema_path.read_text())
    _require(isinstance(schema, dict), "report schema must be an object")
    _require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "unexpected report schema dialect")
    _require(schema.get("type") == "object", "report schema root must be an object")
    _require(isinstance(schema.get("required"), list), "report schema required list missing")
    method = "internal_strict"
    try:
        import jsonschema
    except ImportError:
        return method
    jsonschema.Draft202012Validator.check_schema(schema)
    return "jsonschema+internal_strict"
