"""Exploratory refinement of an immutable candidate payload from an earlier run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from arc_solver.qwen_staged_adapters import (
    activate_named_adapters,
    stage_specs_from_config,
)
from kaggle_runs.qwen_staged_adapters_runpod.worker import (
    build_global_model,
    train_task,
)
from kaggle_runs.qwen_iterative_refiner_runpod.worker import (
    adapt_task_refiner,
    file_digest,
    refine_payload,
    runtime_config,
    train_global_refiner,
    write_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = runtime_config(json.loads(args.config.read_text()))
    challenges = json.loads(args.challenges.read_text())
    task = challenges[args.task_id]
    source = json.loads(args.candidate.read_text())
    if source.get("task_id") != args.task_id:
        raise ValueError("cached candidate task id differs from --task-id")
    base = copy.deepcopy(source)
    for item in base["ranked_candidates"]:
        del item[2:]

    torch.manual_seed(config["seed"])
    model, structure, _ = build_global_model(
        "staged", config, args.config, args.model_dir, args.work
    )
    global_refiner, _ = train_global_refiner(
        model, structure, config, args.config, args.model_dir, args.work
    )
    task_refiner, adaptation = adapt_task_refiner(
        global_refiner, model, structure, task, args.task_id, config
    )
    _, _, late = stage_specs_from_config(model.config)
    train_task(
        model,
        structure,
        task,
        args.task_id,
        late,
        config,
        [],
        steps=config["task_local"]["steps"],
    )
    try:
        baseline = {
            **base,
            "arm": "baseline",
            "experiment_arm": "baseline",
            "stage_trajectories": [
                [
                    {"draft_rank": rank, "G0": item["grid"]}
                    for rank, item in enumerate(items)
                ]
                for items in base["ranked_candidates"]
            ],
        }
        global_payload = refine_payload(
            base,
            global_refiner,
            model,
            structure,
            task,
            args.task_id,
            "global_refiner",
            config,
        )
        task_payload = refine_payload(
            base,
            task_refiner,
            model,
            structure,
            task,
            args.task_id,
            "task_refiner",
            config,
        )
        task_payload["task_adaptation"] = adaptation
        for arm, payload in (
            ("baseline", baseline),
            ("global_refiner", global_payload),
            ("task_refiner", task_payload),
        ):
            destination = (
                args.output / "arms" / arm / "candidates" / f"{args.task_id}.json"
            )
            write_candidate(
                destination,
                payload,
                f"cached-source:{file_digest(args.candidate)}",
            )
    finally:
        model.delete_adapter("task")
        activate_named_adapters(model, ["perception", "interpretation"])


if __name__ == "__main__":
    main()
