"""Rebuild Experiment 017's CPU-only structural novelty report."""

from __future__ import annotations

import json
from pathlib import Path

from arc_solver.structural_novelty import (
    ALL_FLAGS,
    analyze_target,
    jsonable,
    summarize_records,
)


ROOT = Path(__file__).resolve().parents[2]
LOO_ROOT = (
    ROOT
    / "kaggle_runs"
    / "nvarc_rslora_loo_l4"
    / "output_v001"
    / "experiment_016"
)
REAL_ROOT = (
    ROOT
    / "kaggle_runs"
    / "nvarc_single_pass_l4"
    / "output_v007"
    / "experiment_015"
)


def _load(path: Path):
    return json.loads(path.read_text())


def _performance(item: dict) -> dict:
    return {
        "top2": bool(item["top2"]),
        "truth_coverage": bool(item["truth_present"]),
        "best_cell_accuracy": float(item["best_candidate_cell_accuracy"]),
    }


def build_report() -> dict:
    loo_challenges = _load(LOO_ROOT / "solver_input" / "loo_challenges.json")
    loo_labels = _load(LOO_ROOT / "parent_finalizer_only" / "loo_labels.json")
    loo_metadata = _load(LOO_ROOT / "parent_finalizer_only" / "loo_metadata.json")
    loo_performance = _load(
        ROOT / "experiments" / "016_rslora_loo_validation" / "paired_loo_report.json"
    )
    fold_performance = {fold["fold_key"]: fold for fold in loo_performance["folds"]}
    metadata_by_fold = {fold["fold_key"]: fold for fold in loo_metadata["folds"]}

    loo_records = []
    for fold_key in sorted(loo_challenges):
        challenge = loo_challenges[fold_key]
        held_output = loo_labels[fold_key][0]
        analysis = analyze_target(
            {"input": challenge["test"][0]["input"], "output": held_output},
            challenge["train"],
        )
        fold = fold_performance[fold_key]
        loo_records.append(
            {
                "item_key": fold_key,
                **metadata_by_fold[fold_key],
                **analysis,
                "performance": {
                    "base": _performance(fold["base"]),
                    "rslora": _performance(fold["rslora"]),
                },
            }
        )

    real_challenges = _load(REAL_ROOT / "solver_input" / "challenges.json")
    real_labels = _load(REAL_ROOT / "parent_finalizer_only" / "labels.json")
    diagnostics = _load(
        ROOT
        / "experiments"
        / "015_nvarc_single_pass"
        / "candidate_diagnostics20_report.json"
    )
    diagnostic_items = {
        (item["task_id"], item["test_index"]): item
        for task in diagnostics["candidate_diagnostics"]["tasks"]
        for item in task["items"]
    }
    real_records = []
    for task_id in sorted(real_challenges):
        challenge = real_challenges[task_id]
        for test_index, test_item in enumerate(challenge["test"]):
            diagnostic = diagnostic_items[(task_id, test_index)]
            analysis = analyze_target(
                {
                    "input": test_item["input"],
                    "output": real_labels[task_id][test_index],
                },
                challenge["train"],
            )
            real_records.append(
                {
                    "item_key": f"{task_id}:{test_index}",
                    "task_id": task_id,
                    "test_index": test_index,
                    **analysis,
                    "performance": {
                        "rslora": {
                            "top2": bool(diagnostic["score_kgmon_top2"]),
                            "truth_coverage": bool(diagnostic["truth_present"]),
                            "best_cell_accuracy": float(
                                diagnostic["best_candidate_cell_accuracy"]
                            ),
                        }
                    },
                }
            )

    summaries = {
        "loo": summarize_records(loo_records, ["base", "rslora"]),
        "real_test": summarize_records(real_records, ["rslora"]),
    }
    comparison = {}
    for flag in ALL_FLAGS:
        loo_rate = summaries["loo"]["flag_rates"][flag]["rate"]
        real_rate = summaries["real_test"]["flag_rates"][flag]["rate"]
        comparison[flag] = {
            "loo_count": summaries["loo"]["flag_rates"][flag]["count"],
            "loo_denominator": len(loo_records),
            "loo_rate": loo_rate,
            "real_test_count": summaries["real_test"]["flag_rates"][flag]["count"],
            "real_test_denominator": len(real_records),
            "real_test_rate": real_rate,
            "real_minus_loo_rate": real_rate - loo_rate,
        }

    def overall(records: list[dict], arm: str) -> dict:
        rows = [record["performance"][arm] for record in records]
        return {
            "count": len(rows),
            "top2_count": sum(row["top2"] for row in rows),
            "top2_rate": sum(row["top2"] for row in rows) / len(rows),
            "truth_coverage_count": sum(row["truth_coverage"] for row in rows),
            "truth_coverage_rate": sum(row["truth_coverage"] for row in rows)
            / len(rows),
            "mean_best_cell_accuracy": sum(
                row["best_cell_accuracy"] for row in rows
            )
            / len(rows),
        }

    return jsonable(
        {
            "experiment": "017_structural_novelty_analysis",
            "hypothesis": (
                "LOO performance may overestimate real-test transfer because held "
                "training pairs are structurally less novel relative to their "
                "references than real-test items."
            ),
            "method": {
                "exact_pair_orbit": (
                    "A target pair matches when one shared D4 transform and one "
                    "globally consistent injective color mapping transform a "
                    "reference input and output into the target input and output."
                ),
                "exact_input_orbit": (
                    "The same test applied to input only; its mapping is independent "
                    "of the pair-orbit test."
                ),
                "color_assumption": (
                    "Exact orbit tests treat 0 exactly like every other permutable "
                    "color and permit partial mappings over observed source colors."
                ),
                "foreground_assumption": (
                    "Foreground cell counts and binary component shapes alone use "
                    "0 as background. Same-color components include color 0."
                ),
                "novelty_definition": (
                    "A signature is novel iff its exact value occurs in none of the "
                    "appropriate remaining training references."
                ),
                "statistics": (
                    "Counts and descriptive rates are primary. Phi is emitted only "
                    "when both binary margins vary; point-biserial Pearson r only "
                    "when flag and accuracy vary. Null means undefined. No p-values "
                    "or significance claims are made."
                ),
            },
            "sources": {
                "loo_challenges": str(
                    LOO_ROOT.relative_to(ROOT) / "solver_input" / "loo_challenges.json"
                ),
                "loo_labels": str(
                    LOO_ROOT.relative_to(ROOT)
                    / "parent_finalizer_only"
                    / "loo_labels.json"
                ),
                "loo_metadata": str(
                    LOO_ROOT.relative_to(ROOT)
                    / "parent_finalizer_only"
                    / "loo_metadata.json"
                ),
                "loo_performance": (
                    "experiments/016_rslora_loo_validation/paired_loo_report.json"
                ),
                "real_challenges": str(
                    REAL_ROOT.relative_to(ROOT) / "solver_input" / "challenges.json"
                ),
                "real_labels": str(
                    REAL_ROOT.relative_to(ROOT)
                    / "parent_finalizer_only"
                    / "labels.json"
                ),
                "real_performance": (
                    "experiments/015_nvarc_single_pass/"
                    "candidate_diagnostics20_report.json"
                ),
            },
            "overall_performance": {
                "loo_base": overall(loo_records, "base"),
                "loo_rslora": overall(loo_records, "rslora"),
                "real_test_rslora": overall(real_records, "rslora"),
            },
            "summaries": summaries,
            "loo_vs_real_test": comparison,
            "records": {"loo": loo_records, "real_test": real_records},
            "interpretation": {
                "verdict": "unresolved",
                "measured_facts": [
                    (
                        "Exact pair-orbit and input-orbit novelty were both "
                        "17/17 for LOO and 33/33 for real test, so these exact "
                        "measures do not distinguish the datasets."
                    ),
                    (
                        "Real test had higher novelty rates for input dimensions "
                        "(25/33 vs 9/17), input color count (14/33 vs 1/17), and "
                        "input nonzero binary shapes (32/33 vs 13/17)."
                    ),
                    (
                        "Dimension-relation novelty was lower on real test "
                        "(6/33) than LOO (4/17), while real-test RSLoRA top-2 was "
                        "0/6 for relation-novel versus 6/27 otherwise."
                    ),
                    (
                        "LOO RSLoRA achieved top-2 on 15/17 with truth coverage "
                        "17/17; real-test RSLoRA achieved top-2 on 6/33 with "
                        "truth coverage 8/33."
                    ),
                ],
                "hypotheses_and_limitations": [
                    (
                        "The measured facts are compatible with some coarse "
                        "signature shift, but exact orbit measures are saturated "
                        "and within-real performance directions are inconsistent; "
                        "therefore the proposed structural-novelty explanation is "
                        "neither supported nor contradicted by this sample."
                    ),
                    "These observational conditionals do not identify causality.",
                    "There are only 17 LOO folds and 33 real-test items.",
                    "Items from the same ARC task are not statistically independent.",
                    "Exact signatures intentionally do not measure semantic rule novelty.",
                    "The four LOO source tasks are a subset of the 20 real-test tasks.",
                ],
            },
        }
    )


if __name__ == "__main__":
    output = Path(__file__).with_name("analysis_report.json")
    output.write_text(json.dumps(build_report(), indent=2) + "\n")
    print(output)
