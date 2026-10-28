import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

from arc_solver.nvarc_loo import (
    aggregate_paired_report,
    build_loo_bundle,
    build_loo_solver_challenges,
    load_loo_labels,
    make_fold_key,
    parse_fold_key,
)
from kaggle_runs.nvarc_rslora_loo_l4.build_kernel import build_archive


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "kaggle_runs" / "nvarc_rslora_loo_l4"
REAL_SMOKE_FIXTURE = (
    ROOT
    / "kaggle_runs"
    / "nvarc_single_pass_l4"
    / "output_v005"
    / "experiment_015"
    / "solver_input"
    / "challenges.json"
)


def _pair(value):
    return {"input": [[value]], "output": [[value, value]]}


def test_fold_bundle_is_deterministic_unique_and_leakage_safe():
    challenges = {
        "bbbbbbbb": {
            "train": [_pair(1), _pair(2), _pair(3)],
            "test": [{"input": [[0]]}],
        },
        "aaaaaaaa": {
            "train": [_pair(4), _pair(5), _pair(6)],
            "test": [{"input": [[0]]}],
        },
    }
    first = build_loo_bundle(challenges, ["bbbbbbbb", "aaaaaaaa"])
    second = build_loo_bundle(challenges, ["bbbbbbbb", "aaaaaaaa"])
    assert first == second
    public, labels, metadata = first
    keys = list(public)
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys)) == 6
    assert all("_" not in key for key in keys)
    assert [fold["fold_key"] for fold in metadata["folds"]] == keys
    for key in keys:
        task_id, held_index = parse_fold_key(key)
        assert make_fold_key(task_id, held_index) == key
        fold = public[key]
        assert len(fold["train"]) == 2
        assert all("output" in pair for pair in fold["train"])
        assert fold["test"] == [
            {"input": challenges[task_id]["train"][held_index]["input"]}
        ]
        assert "output" not in fold["test"][0]
        held_bytes = json.dumps(labels[key][0], separators=(",", ":"))
        solver_bytes = json.dumps(fold, separators=(",", ":"))
        assert held_bytes not in solver_bytes


def test_solver_construction_and_late_label_loading_match():
    challenges = {
        "aaaaaaaa": {
            "train": [_pair(1), _pair(2), _pair(3)],
            "test": [{"input": [[0]]}],
        }
    }
    full_public, full_labels, full_metadata = build_loo_bundle(
        challenges, ["aaaaaaaa"]
    )
    public, metadata = build_loo_solver_challenges(challenges, ["aaaaaaaa"])
    assert public == full_public
    assert metadata == full_metadata
    assert load_loo_labels(challenges, metadata) == full_labels


def test_real_smoke_artifact_has_exactly_17_folds_when_available():
    if not REAL_SMOKE_FIXTURE.exists():
        return
    challenges = json.loads(REAL_SMOKE_FIXTURE.read_text())
    task_ids = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
    public, labels, metadata = build_loo_bundle(challenges, task_ids)
    assert {task: len(challenges[task]["train"]) for task in task_ids} == {
        "0934a4d8": 4,
        "36a08778": 6,
        "981571dc": 4,
        "aa4ec2a5": 3,
    }
    assert len(public) == len(labels) == metadata["fold_count"] == 17
    assert min(len(task["train"]) for task in public.values()) >= 2


def _oracle(items):
    return {
        "tasks": [
            {"task_id": fold_key, "items": [item]}
            for fold_key, item in items.items()
        ]
    }


def _item(present, top2, rank, accuracy, count):
    return {
        "truth_present": present,
        "score_kgmon_top2": top2,
        "score_kgmon_truth_rank": rank,
        "score_full_probmul_3_top2": top2,
        "score_full_probmul_3_truth_rank": rank,
        "best_candidate_cell_accuracy": accuracy,
        "unique_candidate_count": count,
    }


def test_paired_aggregation_counts_outcomes_and_nll_directions():
    metadata = {
        "folds": [
            {
                "fold_key": "aaaaaaaa~loo~000",
                "task_id": "aaaaaaaa",
                "held_train_index": 0,
                "remaining_train_pairs": 2,
            },
            {
                "fold_key": "aaaaaaaa~loo~001",
                "task_id": "aaaaaaaa",
                "held_train_index": 1,
                "remaining_train_pairs": 2,
            },
        ]
    }
    keys = [fold["fold_key"] for fold in metadata["folds"]]
    oracles = {
        "base": _oracle(
            {
                keys[0]: _item(False, False, None, 0.5, 3),
                keys[1]: _item(True, True, 1, 1.0, 2),
            }
        ),
        "rslora": _oracle(
            {
                keys[0]: _item(True, True, 2, 1.0, 4),
                keys[1]: _item(False, False, None, 0.75, 5),
            }
        ),
    }
    submissions = {
        arm: {
            key: [{"attempt_1": [[1]], "attempt_2": [[0]]}]
            for key in keys
        }
        for arm in ("base", "rslora")
    }
    telemetry = {
        arm: {
            key: {
                "decode_seconds": 10,
                "decode_cap_seconds": 1200,
                "hit_decode_cap": False,
                "training_steps": int(arm == "rslora"),
                "training_loss": None,
                "training_seconds": 0,
            }
            for key in keys
        }
        for arm in ("base", "rslora")
    }
    nll = [
        {
            "fold_key": keys[0],
            "canonical": {"delta_sum_nll": -1.0},
            "multi_view": {"delta_sum_nll": 0.0},
        },
        {
            "fold_key": keys[1],
            "canonical": {"delta_sum_nll": 2.0},
            "multi_view": {"delta_sum_nll": -3.0},
        },
    ]
    report = aggregate_paired_report(
        metadata, oracles, submissions, telemetry, nll
    )
    assert report["aggregate"]["paired_top2"] == {
        "rslora_win": 1,
        "base_win": 1,
        "tie": 0,
    }
    assert report["aggregate"]["nll"] == {
        "canonical": {"improve": 1, "tie": 0, "worsen": 1},
        "multi_view": {"improve": 1, "tie": 1, "worsen": 0},
    }
    assert report["aggregate"]["base_truth_coverage"] == 1
    assert report["aggregate"]["rslora_truth_coverage"] == 1


def test_config_and_kaggle_sources_are_scientifically_pinned():
    config = json.loads((RUN_DIR / "config.json").read_text())
    metadata = json.loads((RUN_DIR / "kernel-metadata.json").read_text())
    assert config["task_ids"] == [
        "0934a4d8",
        "36a08778",
        "981571dc",
        "aa4ec2a5",
    ]
    assert config["expected_fold_count"] == 17
    assert config["minimum_demos"] == 2
    assert config["n_train_aug"] == 16
    assert config["train_geometries"] == 8
    assert config["n_eval_aug"] == 2
    assert config["eval_geometries"] == 8
    assert config["decode_cap_seconds"] == 1200
    assert config["dfs_window_seconds"] == 540
    assert config["decode_batch"] == 4
    assert config["min_prob"] == 0.2
    assert config["epochs"] == 1
    assert config["lora"] == {
        "type": "RSLoRA",
        "rank": 256,
        "alpha": 32,
        "dropout": 0.0,
    }
    assert metadata["machine_shape"] == "NvidiaL4"
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == [
        "sorokin/pip-install-unsloth-flash-patch"
    ]
    assert metadata["model_sources"] == [
        "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
    ]
    assert metadata["docker_image"].endswith(
        "sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
    )


def test_archive_uses_dedicated_worker_and_packages_cpu_helpers():
    with zipfile.ZipFile(BytesIO(base64.b64decode(build_archive()))) as archive:
        names = set(archive.namelist())
        starter = archive.read("starter.py").decode()
    assert "nvarc_upstream_solver.py" in names
    assert "loo_worker.py" in names
    assert "arc_solver/nvarc_loo.py" in names
    assert "arc_solver/nvarc_eval.py" in names
    assert "from loo_worker import worker" in starter
    assert "arc_solver.py" not in names


def test_candidate_finalization_precedes_label_materialization_and_nll():
    runner = (RUN_DIR / "runner_template.py").read_text()
    base = runner.index('run_phase("base_candidates"')
    rslora = runner.index('run_phase("rslora_candidates"')
    labels = runner.index("labels = load_loo_labels")
    finalize = runner.index("for arm in (\"base\", \"rslora\")", labels)
    nll = runner.index('"nll",', finalize)
    assert base < rslora < labels < finalize < nll
    candidate_block = runner[base:labels]
    assert "ARC_LABEL_DIR" not in candidate_block
    assert "label_dir=" not in candidate_block


def test_nll_worker_orders_training_before_label_and_has_no_later_train_or_dfs():
    worker = (RUN_DIR / "loo_worker.py").read_text()
    function = worker[worker.index("def _run_nll(") : worker.index("\ndef worker(")]
    label_open = function.index("label_path.read_text()")
    assert function.index('"training_completed"') < label_open
    after_label = function[label_open:]
    assert "trainer.train" not in after_label
    assert "inference_turbo_dfs" not in after_label
    assert "token lengths changed" in after_label


def test_upstream_sources_remain_byte_pinned():
    expected = {
        "UPSTREAM.md": "4ef0ab51e14680ac6a22f19af33389ba92c04a68cceeb2f2deef17833bbb421f",
        "arc_decoder.py": "20ec4d4d31f5225cfd54d1ca2e92b73211800cdba413a1bec00c338ce95811a3",
        "arc_loader.py": "2c62526daba3fdd9da317e3dcb689acca8e24018e56e4c7f2c33ed15381ed8c6",
        "arc_solver.py": "2438f66910a1f3ebdfea4e166807c2ce9df534790129f6ad86ceb26088a52131",
        "starter.py": "fe4f1b27af75a2c32ead5b61e73135bcd7ee0ba86228e2aa608e9dfd6a243c69",
    }
    upstream = ROOT / "third_party" / "nvarc_single_pass"
    assert {
        name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
        for name in expected
    } == expected
