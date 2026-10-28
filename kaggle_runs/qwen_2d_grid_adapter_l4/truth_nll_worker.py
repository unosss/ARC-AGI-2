"""Post-candidate, per-task-label NLL evaluator; never generates candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter
from arc_solver.qwen2d_augment import augment_task, color_permutation
from arc_solver.qwen2d_metadata import LayoutMetadata, build_layout
from arc_solver.qwen2d_runtime import (
    collate_token_rows,
    multi_view_nll,
    serialize_chatml_bounded,
    teacher_forced_metrics,
    train_fixed_steps,
)
from gpu_arm_worker import (
    check_deadline,
    find_training_challenges,
    load_base,
    lora_config,
    stable_seed,
    deadline_epoch,
    training_rows,
)


def score_views(
    model,
    structure,
    task,
    truths,
    arm,
    config,
):
    items = []
    for test_index, truth in enumerate(truths):
        canonical_seed = stable_seed(
            f"truth-canonical:{test_index}", config["seed"]
        )
        canonical_tokens, canonical_start, _ = serialize_chatml_bounded(
            task["train"],
            task["test"][test_index]["input"],
            truth,
            max_length=config["global_max_sequence_length"],
        )
        canonical_ids = torch.tensor([canonical_tokens], device="cuda")
        canonical_layout = LayoutMetadata.stack(
            [
                build_layout(
                    canonical_tokens, arm=arm, shuffle_seed=canonical_seed
                )
            ]
        )
        post = teacher_forced_metrics(
            model, structure, canonical_ids, canonical_layout, canonical_start
        )
        with model.disable_adapter():
            pre = teacher_forced_metrics(
                model, structure, canonical_ids, canonical_layout, canonical_start
            )
        views = []
        for permutation_index in range(config["task_local"]["decode_color_permutations"]):
            permutation = color_permutation(
                stable_seed(f"truth-nll:{test_index}:{permutation_index}", config["seed"])
            )
            task_with_truth = {
                "train": task["train"],
                "test": [
                    {"input": item["input"], "output": item_truth}
                    for item, item_truth in zip(task["test"], truths)
                ],
            }
            for geometry in range(config["task_local"]["decode_geometries"]):
                view_seed = stable_seed(
                    f"truth-order:{test_index}:{permutation_index}:{geometry}",
                    config["seed"],
                )
                augmented = augment_task(
                    task_with_truth,
                    geometry=geometry,
                    permutation=permutation,
                    order_seed=view_seed,
                    keep_test_outputs=True,
                )
                target = augmented["test"][test_index]["output"]
                tokens, start, _ = serialize_chatml_bounded(
                    augmented["train"],
                    augmented["test"][test_index]["input"],
                    target,
                    max_length=config["global_max_sequence_length"],
                )
                views.append(
                    (
                        torch.tensor([tokens], device="cuda"),
                        LayoutMetadata.stack(
                            [
                                build_layout(
                                    tokens, arm=arm, shuffle_seed=view_seed
                                )
                            ]
                        ),
                        start,
                    )
                )
        view_metric = multi_view_nll(
            model, structure, views, deadline_epoch=deadline_epoch()
        )
        with model.disable_adapter():
            pre_view_metric = multi_view_nll(
                model, structure, views, deadline_epoch=deadline_epoch()
            )
        items.append(
            {
                "canonical_nll": post["sum_nll"],
                "view16_nll": view_metric["mean_sum_nll"],
                "first_token_nll": post["first_token_nll"],
                "per_row_nll": post["per_row_nll"],
                "pre_task_lora": {
                    "canonical": pre,
                    "view16": pre_view_metric,
                },
                "post_task_lora": {
                    "canonical": post,
                    "view16": view_metric,
                },
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prepared = json.loads(args.prepared.read_text())
    from peft import PeftModel, get_peft_model

    base, _ = load_base(args.model_dir)
    global_root = args.work / "arms" / args.arm / "global" / "final"
    model = PeftModel.from_pretrained(base, global_root / "lora").merge_and_unload()
    structure = AdditiveStructureAdapter.load_arm(
        global_root / "structure", model.config, freeze=True
    ).cuda().to(torch.bfloat16)
    challenges = json.loads(find_training_challenges().read_text())
    task_ids = (
        prepared["manifest"]["negative_control_ids"]
        if args.arm in config["negative_controls"]
        else prepared["manifest"]["primary_ids"]
    )
    arm_root = args.work / "arms" / args.arm
    output = arm_root / "truth_nll.jsonl"
    lifecycle = arm_root / "truth_nll_lifecycle.jsonl"
    records_dir = arm_root / "truth_nll_records"
    lifecycle_dir = arm_root / "truth_nll_lifecycle_records"
    records_dir.mkdir(exist_ok=True)
    lifecycle_dir.mkdir(exist_ok=True)
    completed = {
        path.stem
        for path in records_dir.glob("*.json")
        if (lifecycle_dir / path.name).is_file()
    }
    for task_id in task_ids:
        check_deadline()
        if task_id in completed:
            continue
        torch.manual_seed(config["seed"])
        task_model = get_peft_model(model, lora_config(config["task_local"]))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in task_model.parameters() if parameter.requires_grad],
            lr=config["task_local"]["learning_rate"],
            weight_decay=0.0,
        )
        rows = list(training_rows(challenges[task_id], task_id, config))
        train_fixed_steps(
            task_model,
            structure,
            (
                collate_token_rows([row], arm=args.arm, shuffle_seed=config["seed"] + index)
                for index, row in enumerate(rows)
            ),
            optimizer=optimizer,
            steps=config["task_local"]["steps"],
            deadline_epoch=deadline_epoch(),
        )
        label_path = args.label_dir / f"{task_id}.json"
        truths = json.loads(label_path.read_text())
        record = {
            "arm": args.arm,
            "task_id": task_id,
            "items": score_views(
                task_model,
                structure,
                challenges[task_id],
                truths,
                args.arm,
                config,
            ),
            "optimizer_steps_after_label": 0,
            "dfs_calls_after_label": 0,
        }
        record_tmp = records_dir / f".{task_id}.tmp"
        record_tmp.write_text(json.dumps(record, sort_keys=True))
        os.replace(record_tmp, records_dir / f"{task_id}.json")
        events = [
            {"arm": args.arm, "task_id": task_id, "event": name}
            for name in ("training_completed", "label_opened", "nll_completed")
        ]
        lifecycle_tmp = lifecycle_dir / f".{task_id}.tmp"
        lifecycle_tmp.write_text(json.dumps(events, sort_keys=True))
        os.replace(lifecycle_tmp, lifecycle_dir / f"{task_id}.json")
        task_model.unload()
    rows = [
        json.loads((records_dir / f"{task_id}.json").read_text())
        for task_id in task_ids
    ]
    events = [
        item
        for task_id in task_ids
        for item in json.loads(
            (lifecycle_dir / f"{task_id}.json").read_text()
        )
    ]
    output_tmp = output.with_suffix(".tmp")
    output_tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    os.replace(output_tmp, output)
    lifecycle_tmp = lifecycle.with_suffix(".tmp")
    lifecycle_tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events)
    )
    os.replace(lifecycle_tmp, lifecycle)


if __name__ == "__main__":
    main()
