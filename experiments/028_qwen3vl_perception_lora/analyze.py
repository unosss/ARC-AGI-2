"""Analyze Experiment 028 against the frozen Experiment 027 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_solver.perception import score_description


METRICS = ("reading", "objects", "relations", "overall")
ARMS = ("language_lora", "multimodal_lora")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_scores(records: list[dict[str, Any]]) -> None:
    for record in records:
        recomputed = score_description(record.get("prediction"), record["expected"])
        if record.get("score") != recomputed:
            raise ValueError(f"stored score mismatch: {record.get('id')}")


def validate_evaluation_artifacts(
    root: Path,
    arm: str,
    config: dict[str, Any],
    expected_config_hash: str,
    training_completion_path: Path,
) -> list[dict[str, Any]]:
    manifest_path = root / "manifest.json"
    results_path = root / "results.jsonl"
    completion_path = root / "completed.json"
    if not all(path.exists() for path in (manifest_path, results_path, completion_path)):
        raise RuntimeError(f"incomplete evaluation artifacts: {root}")
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads(completion_path.read_text())
    required_manifest = {
        "experiment": 28,
        "arm": arm,
        "config_sha256": expected_config_hash,
        "train_corpus_sha256": config["train_corpus_sha256"],
        "synthetic_eval_sha256": config["synthetic_eval_sha256"],
        "official_challenges_sha256": config["official_challenges_sha256"],
        "adapter_completion_sha256": file_sha256(training_completion_path),
        "row_count": 84,
    }
    for key, value in required_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"evaluation identity mismatch: {root} {key}")
    for key, value in manifest.items():
        if completion.get(key) != value:
            raise ValueError(f"evaluation completion mismatch: {root} {key}")
    if completion.get("completed") is not True:
        raise ValueError(f"evaluation not marked completed: {root}")
    if completion.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError(f"evaluation manifest hash mismatch: {root}")
    if completion.get("results_sha256") != file_sha256(results_path):
        raise ValueError(f"evaluation results hash mismatch: {root}")
    records = load_jsonl(results_path)
    validate_scores(records)
    if completion.get("exact_count") != sum(
        record["score"]["exact"] for record in records
    ):
        raise ValueError(f"evaluation exact count mismatch: {root}")
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split"], record["modality"])].append(record)
        groups[(record["split"], "all")].append(record)
    result = {}
    for (split, modality), values in sorted(groups.items()):
        result[f"{split}/{modality}"] = {
            "count": len(values),
            **{
                metric: sum(value["score"][metric] for value in values) / len(values)
                for metric in METRICS
            },
            "exact_count": sum(value["score"]["exact"] for value in values),
            "exact_rate": sum(value["score"]["exact"] for value in values)
            / len(values),
            "mean_seconds": sum(value["elapsed_seconds"] for value in values)
            / len(values),
        }
    return result


def official_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return record["task_id"], record["role"], record["modality"]


def paired(
    candidate_records: list[dict[str, Any]], base_records: list[dict[str, Any]]
) -> dict[str, Any]:
    candidates = {
        official_key(record): record
        for record in candidate_records
        if record["split"] == "official"
    }
    base = {official_key(record): record for record in base_records}
    if set(candidates) != set(base):
        raise ValueError("candidate and frozen base official keys differ")
    for key in base:
        if candidates[key]["grid_sha256"] != base[key]["grid_sha256"]:
            raise ValueError(f"candidate and base grid hashes differ: {key}")
    by_modality = {}
    for modality in ("text", "image", "both"):
        details = []
        wins = losses = ties = 0
        for key in sorted(key for key in base if key[2] == modality):
            delta = (
                candidates[key]["score"]["overall"] - base[key]["score"]["overall"]
            )
            wins += delta > 0
            losses += delta < 0
            ties += delta == 0
            details.append(
                {
                    "task_id": key[0],
                    "role": key[1],
                    "delta_overall": delta,
                    "base_exact": base[key]["score"]["exact"],
                    "candidate_exact": candidates[key]["score"]["exact"],
                }
            )
        by_modality[modality] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "details": details,
        }
    return by_modality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    expected_config_hash = file_sha256(args.config)
    if file_sha256(args.base_results) != config["experiment_027_base_results_sha256"]:
        raise ValueError("frozen Experiment 027 baseline SHA-256 mismatch")
    base_raw = load_jsonl(args.base_results)
    if len(base_raw) != 60:
        raise ValueError("frozen Experiment 027 must contain 60 responses")
    validate_scores(base_raw)
    base = [{**record, "split": "official"} for record in base_raw]
    report: dict[str, Any] = {
        "base": aggregate(base),
        "arms": {},
    }
    for arm in ARMS:
        training_completion_path = args.runs_root / arm / "completed.json"
        if not training_completion_path.exists():
            raise RuntimeError(f"missing training completion: {training_completion_path}")
        training_completion = json.loads(training_completion_path.read_text())
        if training_completion.get("arm") != arm or training_completion.get("pilot"):
            raise ValueError(f"training completion identity mismatch: {arm}")
        records = validate_evaluation_artifacts(
            args.runs_root / arm / "evaluation",
            arm,
            config,
            expected_config_hash,
            training_completion_path,
        )
        if len(records) != 84:
            raise ValueError(f"{arm} must contain 84 evaluation responses")
        report["arms"][arm] = {
            "training": training_completion,
            "aggregate": aggregate(records),
            "paired_vs_base": paired(records, base),
        }

    base_both = report["base"]["official/both"]
    language = report["arms"]["language_lora"]["aggregate"]
    multimodal = report["arms"]["multimodal_lora"]["aggregate"]
    all_official_both_exact_zero = (
        base_both["exact_rate"] == 0
        and language["official/both"]["exact_rate"] == 0
        and multimodal["official/both"]["exact_rate"] == 0
    )
    support = {}
    for arm_name, arm in (
        ("language_lora", language),
        ("multimodal_lora", multimodal),
    ):
        paired_both = report["arms"][arm_name]["paired_vs_base"]["both"]
        support[arm_name] = (
            arm["official/both"]["exact_rate"] > base_both["exact_rate"]
            or (
                all_official_both_exact_zero
                and arm["official/both"]["overall"] > base_both["overall"]
                and paired_both["wins"] > paired_both["losses"]
                and arm["synthetic_eval/all"]["exact_rate"] > 0
            )
        )
    visual_path_supported = (
        multimodal["official/image"]["overall"]
        > language["official/image"]["overall"]
        or multimodal["official/both"]["overall"]
        > language["official/both"]["overall"]
    )
    report["decision"] = {
        "perception_lora_supported": support,
        "multimodal_path_supported": visual_path_supported,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
