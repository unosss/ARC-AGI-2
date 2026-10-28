"""Create the deterministic paired report for Experiment 027."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("reading", "objects", "relations", "overall")
EXPECTED_GRID_COUNT = 20
EXPECTED_RESPONSE_COUNT = 60


def read_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    keys = [(r["task_id"], r["role"], r["modality"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate record keys")
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != EXPECTED_RESPONSE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_RESPONSE_COUNT} responses, got {len(records)}"
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(record["modality"], record["role"])].append(record)
        paired[(record["task_id"], record["role"])][record["modality"]] = record

    aggregates: dict[str, Any] = {}
    for (modality, role), values in sorted(grouped.items()):
        aggregates[f"{modality}/{role}"] = {
            "count": len(values),
            **{
                metric: sum(v["score"][metric] for v in values) / len(values)
                for metric in METRICS
            },
            "structured_rate": sum(v["score"]["structured"] for v in values)
            / len(values),
            "exact_rate": sum(v["score"]["exact"] for v in values) / len(values),
            "mean_seconds": sum(v["elapsed_seconds"] for v in values) / len(values),
            "mean_output_tokens": sum(v["output_tokens"] for v in values)
            / len(values),
        }

    incomplete = [
        [task_id, role]
        for (task_id, role), arms in paired.items()
        if set(arms) != {"text", "image", "both"}
    ]
    if incomplete:
        raise ValueError(f"incomplete modality triplets: {incomplete}")
    if len(paired) != EXPECTED_GRID_COUNT:
        raise ValueError(f"expected {EXPECTED_GRID_COUNT} grids, got {len(paired)}")

    deltas = []
    wins = losses = ties = 0
    for (task_id, role), arms in sorted(paired.items()):
        delta = arms["both"]["score"]["overall"] - arms["text"]["score"]["overall"]
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
        deltas.append(
            {
                "task_id": task_id,
                "role": role,
                "both_minus_text_overall": delta,
                "text": arms["text"]["score"],
                "image": arms["image"]["score"],
                "both": arms["both"]["score"],
            }
        )

    modality_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        modality_all[record["modality"]].append(record)
    overall = {
        modality: {
            metric: sum(v["score"][metric] for v in values) / len(values)
            for metric in METRICS
        }
        for modality, values in sorted(modality_all.items())
    }
    decision = (
        overall["both"]["overall"] > overall["text"]["overall"]
        and overall["both"]["reading"] >= overall["text"]["reading"]
        and wins > losses
    )
    return {
        "response_count": len(records),
        "grid_count": len(paired),
        "aggregates_by_role": aggregates,
        "aggregates_all": overall,
        "paired_both_vs_text": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "details": deltas,
        },
        "decision_rule_passed": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(read_records(args.results))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
