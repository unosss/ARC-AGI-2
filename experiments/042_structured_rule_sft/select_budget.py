"""Freeze one Experiment 042 budget from validation reports before test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="BUDGET=PATH",
        help="validation report; repeat for each trained budget",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a frozen selection")

    reports = []
    seen_budgets = set()
    for spec in args.report:
        budget_text, separator, path_text = spec.partition("=")
        if not separator:
            raise ValueError(f"invalid report specification: {spec}")
        budget = int(budget_text)
        path = Path(path_text)
        report = json.loads(path.read_text())
        if report.get("phase") != "validation" or report.get("budget") != budget:
            raise ValueError(f"report does not match validation budget {budget}")
        if budget in seen_budgets:
            raise ValueError(f"duplicate validation budget: {budget}")
        seen_budgets.add(budget)
        reports.append((budget, path, report))
    if not reports:
        raise ValueError("no validation reports")
    config = json.loads(args.config.read_text())
    expected_budgets = {int(value) for value in config["train_budgets"]}
    if seen_budgets != expected_budgets:
        raise ValueError(
            f"validation reports must cover exactly {sorted(expected_budgets)}"
        )
    config_hashes = {report["config_sha256"] for _, _, report in reports}
    challenge_hashes = {report["challenges_sha256"] for _, _, report in reports}
    label_hashes = {report["labels_sha256"] for _, _, report in reports}
    if len(config_hashes) != 1 or len(challenge_hashes) != 1 or len(label_hashes) != 1:
        raise ValueError("validation reports do not share data and config provenance")
    if next(iter(config_hashes)) != file_sha256(args.config):
        raise ValueError("validation reports do not match the supplied config")

    passing = [entry for entry in reports if entry[2]["confirmatory_gate_passed"]]
    if passing:
        selected = min(passing, key=lambda entry: entry[0])
        reason = "smallest_confirmatory_budget"
    else:
        selected = max(
            reports,
            key=lambda entry: (
                entry[2]["program_minus_direct"],
                entry[2]["program_exact"],
                -entry[0],
            ),
        )
        reason = "best_validation_delta_no_gate"
    payload = {
        "schema_version": 1,
        "experiment": 42,
        "selected_budget": selected[0],
        "selection_reason": reason,
        "validation_gate_passed": bool(selected[2]["confirmatory_gate_passed"]),
        "config_sha256": next(iter(config_hashes)),
        "validation_challenges_sha256": next(iter(challenge_hashes)),
        "validation_labels_sha256": next(iter(label_hashes)),
        "reports": {
            str(budget): {
                "path": str(path),
                "sha256": file_sha256(path),
                "program_minus_direct": report["program_minus_direct"],
                "confirmatory_gate_passed": report["confirmatory_gate_passed"],
            }
            for budget, path, report in sorted(reports)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
