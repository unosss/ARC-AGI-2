import base64
from io import BytesIO
import json
from pathlib import Path
import zipfile

from arc_solver.nvarc_eval import (
    analyze_candidate_oracles,
    finalize_top2,
    load_finalizer_labels,
    write_solver_challenges,
)
from kaggle_runs.nvarc_single_pass_l4.build_kernel import build_archive


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "kaggle_runs" / "nvarc_single_pass_l4"


def test_labels_are_absent_from_solver_directory(tmp_path):
    challenges = {
        "abc": {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[3]]}],
        }
    }
    challenge_path = tmp_path / "challenges.json"
    solution_path = tmp_path / "solutions.json"
    challenge_path.write_text(json.dumps(challenges))
    solution_path.write_text(json.dumps({"abc": [[[3]]]}))
    solver_dir = tmp_path / "solver"
    finalizer_dir = tmp_path / "parent-only"

    public_path = write_solver_challenges(challenge_path, ["abc"], solver_dir)

    assert list(solver_dir.iterdir()) == [public_path]
    assert "output" not in json.loads(public_path.read_text())["abc"]["test"][0]
    assert not finalizer_dir.exists()

    labels = load_finalizer_labels(
        challenge_path, solution_path, ["abc"], finalizer_dir
    )
    assert labels == {"abc": [[[3]]]}
    assert not finalizer_dir.is_relative_to(solver_dir)


def test_top2_finalization_deduplicates_and_falls_back():
    challenges = {
        "abc": {
            "train": [],
            "test": [{"input": [[1, 2]]}, {"input": [[3]]}],
        }
    }
    selected = {
        "abc_0": [[[9]], [[9]], [[8]]],
        "abc_1": [[], "invalid"],
    }

    submission, counts = finalize_top2(challenges, selected)

    assert submission["abc"][0] == {
        "attempt_1": [[9]],
        "attempt_2": [[8]],
    }
    assert submission["abc"][1] == {
        "attempt_1": [[3]],
        "attempt_2": [[0]],
    }
    assert len(submission["abc"]) == len(challenges["abc"]["test"])
    assert counts["identity_fallbacks"] == 1
    assert counts["zero_fallbacks"] == 1


def test_candidate_oracles_cover_generation_selection_top2_and_multi_item():
    def scorer(field):
        def rank(samples):
            groups = {}
            for sample in samples.values():
                grid = tuple(map(tuple, sample["solution"]))
                groups.setdefault(grid, [sample[field], sample["solution"]])
            return [
                group[1]
                for group in sorted(
                    groups.values(), key=lambda group: group[0], reverse=True
                )
            ]

        return rank

    def sample(grid, kg, prob):
        return {"solution": grid, "kg": kg, "prob": prob}

    decoded = {
        "aaa_0": {
            "a": sample([[0, 0]], 30, 30),
            "b": sample([[1, 0]], 20, 10),
            "c": sample([[1, 1]], 10, 20),
            "duplicate": sample([[1, 1]], 10, 20),
            "invalid": sample([], 100, 100),
        },
        "aaa_1": {"a": sample([[2]], 10, 10)},
        "bbb_0": {
            "a": sample([[3, 0]], 10, 20),
            "b": sample([[9]], 20, 10),
        },
    }
    labels = {"aaa": [[[1, 1]], [[2]]], "bbb": [[[3, 3]]]}

    report = analyze_candidate_oracles(
        decoded,
        labels,
        {
            "score_kgmon": scorer("kg"),
            "score_full_probmul_3": scorer("prob"),
        },
    )

    aggregate = report["aggregate"]
    assert aggregate["item_count"] == 3
    assert aggregate["truth_present_items"] == 2
    assert aggregate["generation_miss_items"] == 1
    assert aggregate["score_kgmon_selection_miss_items"] == 1
    assert aggregate["score_kgmon_top2_items"] == 1
    assert aggregate["score_full_probmul_3_top2_items"] == 2
    assert aggregate["oracle_exact_tasks"] == 1
    assert aggregate["score_kgmon_top2_exact_tasks"] == 0
    assert aggregate["score_full_probmul_3_top2_exact_tasks"] == 1

    first = report["tasks"][0]["items"][0]
    assert first["unique_candidate_count"] == 3
    assert first["matching_shape_candidate_count"] == 3
    assert first["score_kgmon_truth_rank"] == 3
    assert first["score_full_probmul_3_truth_rank"] == 2
    assert first["best_candidate_cell_accuracy"] == 1.0
    generation_miss = report["tasks"][1]["items"][0]
    assert generation_miss["truth_present"] is False
    assert generation_miss["score_kgmon_truth_rank"] is None
    assert generation_miss["best_candidate_cell_accuracy"] == 0.5


def test_smoke_config_and_kaggle_sources_are_pinned():
    config = json.loads((RUN_DIR / "smoke-config.json").read_text())
    metadata = json.loads((RUN_DIR / "kernel-metadata.json").read_text())

    assert config["task_ids"] == [
        "0934a4d8",
        "36a08778",
        "981571dc",
        "aa4ec2a5",
    ]
    assert config["n_train_aug"] == 16
    assert config["n_eval_aug"] == 2
    assert config["task_cap_seconds"] == 1200
    assert config["lora_seed"] == 42
    assert config["train_aug_seed"] == 1
    assert config["eval_aug_seed"] == 2
    assert config["score_seed_offset"] == 0
    assert config["order"] == "sorted"
    assert config["nprocs"] == 4
    assert config["smoke_deadline_seconds"] == 55 * 60
    assert config["lora"] == {"type": "RSLoRA", "rank": 256}
    assert metadata["kernel_sources"] == [
        "sorokin/pip-install-unsloth-flash-patch"
    ]
    assert metadata["model_sources"] == [
        "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
    ]
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
    assert metadata["machine_shape"] == "NvidiaL4"
    assert metadata["enable_internet"] is False
    assert metadata["docker_image"].endswith(
        "sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
    )


def test_additional16_config_is_disjoint_and_keeps_science_settings():
    smoke = json.loads((RUN_DIR / "smoke-config.json").read_text())
    additional = json.loads((RUN_DIR / "additional16-config.json").read_text())

    assert len(additional["task_ids"]) == 16
    assert set(additional["task_ids"]).isdisjoint(smoke["task_ids"])
    for key in (
        "n_train_aug",
        "n_eval_aug",
        "task_cap_seconds",
        "lora_seed",
        "train_aug_seed",
        "eval_aug_seed",
        "score_seed_offset",
        "order",
        "nprocs",
        "selection",
        "lora",
    ):
        assert additional[key] == smoke[key]
    assert additional["report_filename"] == "additional16_report.json"


def test_diagnostic20_config_combines_prior_ids_and_keeps_science_settings():
    smoke = json.loads((RUN_DIR / "smoke-config.json").read_text())
    additional = json.loads((RUN_DIR / "additional16-config.json").read_text())
    diagnostic = json.loads((RUN_DIR / "diagnostic20-config.json").read_text())

    expected = sorted(smoke["task_ids"] + additional["task_ids"])
    assert diagnostic["task_ids"] == expected
    assert len(diagnostic["task_ids"]) == len(set(diagnostic["task_ids"])) == 20
    for key in (
        "n_train_aug",
        "n_eval_aug",
        "task_cap_seconds",
        "lora_seed",
        "train_aug_seed",
        "eval_aug_seed",
        "score_seed_offset",
        "order",
        "nprocs",
        "selection",
        "lora",
    ):
        assert diagnostic[key] == smoke[key]
    assert diagnostic["smoke_deadline_seconds"] == 4 * 60 * 60
    assert diagnostic["report_filename"] == "candidate_diagnostics20_report.json"


def test_packed_upstream_solver_does_not_collide_with_helper_package():
    with zipfile.ZipFile(BytesIO(base64.b64decode(build_archive()))) as archive:
        names = set(archive.namelist())
        starter = archive.read("starter.py").decode()

    assert "nvarc_arc_solver.py" in names
    assert "arc_solver.py" not in names
    assert "arc_solver/nvarc_eval.py" in names
    assert "from nvarc_arc_solver import worker" in starter


def test_runner_adds_offline_unsloth_kernel_to_worker_pythonpath():
    template = (RUN_DIR / "runner_template.py").read_text()

    assert 'search_roots = [INPUT_ROOT, Path("/kaggle/usr/lib")]' in template
    assert 'dependency_root = find_package_root("unsloth")' in template
    assert '"PYTHONPATH": os.pathsep.join((str(source_root), str(dependency_root)))' in template


def test_runner_opens_labels_only_after_solver_exit_and_never_passes_label_path():
    template = (RUN_DIR / "runner_template.py").read_text()

    process_wait = template.index("return_code = process.wait()")
    solution_open = template.index('solution_source = find_arc_file("evaluation_solutions")')
    assert process_wait < solution_open
    command_block = template[
        template.index("command = [") : template.index("with log_path.open")
    ]
    assert "public_path" in command_block
    assert "solution" not in command_block
    assert "label" not in command_block
