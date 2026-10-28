import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arc_solver.qwen2d_adapter import (
    AdditiveStructureAdapter,
    StructuralForward,
    assert_scale_zero_equivalence,
    hidden_size_from_config,
)
from arc_solver.qwen2d_augment import (
    apply_colors,
    augment_task,
    color_permutation,
    d4,
    inverse_d4,
    invert_prediction,
)
from arc_solver.qwen2d_experiment import (
    LabelLifecycle,
    collision_audit,
    fixed_token_budget,
    loss_token_count,
    make_synthetic_rows,
    make_task_manifest,
    paired_contrast,
)
from arc_solver.qwen2d_metadata import (
    FEATURE_NAMES,
    GeneratedGridState,
    LayoutMetadata,
    PINS,
    SegmentRole,
    assert_grids15_tokenizer,
    build_layout,
    serialize_grid,
)
from arc_solver.qwen2d_runtime import (
    batched_multi_view_nll,
    collate_token_rows,
    completion_labels,
    multi_view_nll,
    serialize_chatml,
)
from kaggle_runs.qwen_2d_grid_adapter_l4.build_kernel import build_archive
from kaggle_runs.qwen_2d_grid_adapter_l4.experiment_worker import _score_item
from kaggle_runs.qwen_2d_grid_adapter_l4 import gpu_arm_worker


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "kaggle_runs" / "qwen_2d_grid_adapter_l4"


def sequence(demo_in=((1, 2), (3, 4)), demo_out=((4, 3),), query=((5, 0),), answer=((9, 8),)):
    return [
        PINS.im_start,
        PINS.user,
        PINS.newline,
        *serialize_grid(demo_in),
        PINS.eos,
        PINS.im_start,
        PINS.assistant,
        PINS.newline,
        *serialize_grid(demo_out),
        PINS.eos,
        PINS.im_start,
        PINS.user,
        PINS.newline,
        *serialize_grid(query),
        PINS.eos,
        PINS.im_start,
        PINS.assistant,
        PINS.newline,
        *serialize_grid(answer),
        PINS.eos,
    ]


def test_layout_golden_full_chatml_and_padding():
    tokens = sequence() + [PINS.pad, PINS.pad]
    layout = build_layout(tokens)
    assert len(layout) == len(tokens)
    assert layout.role.tolist() == [
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4,
        0, 0,
    ]
    assert layout.row[3:8].tolist() == [1, 1, 0, 2, 2]
    assert layout.col[3:8].tolist() == [1, 2, 0, 1, 2]
    assert all(getattr(layout, name)[-2:].eq(0).all() for name in FEATURE_NAMES)


def test_parser_rejects_ragged_or_wrong_role_separator():
    with pytest.raises(ValueError, match="ragged"):
        build_layout([PINS.im_start, PINS.user, 10, 1, 2, 10, 3, PINS.eos])
    with pytest.raises(ValueError, match="followed"):
        build_layout([PINS.im_start, PINS.user, 1, PINS.eos])


class FakeTokenizer:
    vocab_size = 13
    pad_token_id = 13

    def __len__(self):
        return 16

    def encode(self, text, add_special_tokens=False):
        special = {
            "\n": [10],
            "<|im_start|>": [14],
            "<|im_start|>user\n": [14, 11, 10],
            "<|im_start|>assistant\n": [14, 12, 10],
            "<|im_end|>": [15],
        }
        return special.get(text, [int(text)] if text.isdigit() and len(text) == 1 else [])


def test_runtime_tokenizer_pins():
    assert_grids15_tokenizer(FakeTokenizer())
    bad = FakeTokenizer()
    bad.pad_token_id = 0
    with pytest.raises(RuntimeError, match="contract mismatch"):
        assert_grids15_tokenizer(bad)
    bad_base = FakeTokenizer()
    bad_base.vocab_size = 16
    with pytest.raises(RuntimeError, match="base_vocab_size"):
        assert_grids15_tokenizer(bad_base)


def test_upstream_chatml_token_boundaries_labels_and_metadata_align():
    train = [{"input": [[1, 2], [3, 4]], "output": [[5, 6]]}]
    tokens, answer_start = serialize_chatml(train, [[7], [8]], [[9, 0]])
    expected = [
        14, 11, 10, 1, 2, 10, 3, 4, 15,
        14, 12, 10, 5, 6, 15,
        14, 11, 10, 7, 10, 8, 15,
        14, 12, 10, 9, 0, 15,
    ]
    assert tokens == expected
    assert answer_start == 25

    layout = build_layout(tokens)
    assert layout.role.tolist() == (
        [SegmentRole.DEMO_INPUT] * 9
        + [SegmentRole.DEMO_OUTPUT] * 6
        + [SegmentRole.QUERY_INPUT] * 7
        + [SegmentRole.QUERY_OUTPUT] * 6
    )
    assert layout.grid.tolist() == [1] * 9 + [2] * 6 + [3] * 7 + [4] * 6
    assert layout.row[answer_start:].tolist() == [1, 1, 0]
    assert layout.col[answer_start:].tolist() == [1, 2, 0]

    labels = completion_labels(torch.tensor(tokens), layout)
    supervised = torch.nonzero(labels.ne(-100), as_tuple=False).flatten().tolist()
    assert supervised == [12, 13, 14, 25, 26, 27]
    assert labels[supervised].tolist() == [5, 6, 15, 9, 0, 15]


def test_component_features_are_color_and_d4_invariant_and_outputs_null():
    first = sequence(demo_in=((1, 1, 0), (0, 1, 2), (2, 2, 2)))
    recolored = sequence(demo_in=((7, 7, 0), (0, 7, 4), (4, 4, 4)))
    a, b = build_layout(first), build_layout(recolored)
    cell_mask = a.role == int(SegmentRole.DEMO_INPUT)
    cell_mask &= a.row > 0
    for name in ("component_size", "bbox_height", "bbox_width", "topology"):
        assert getattr(a, name)[cell_mask].tolist() == getattr(b, name)[cell_mask].tolist()
    assistant = (a.role == 2) | (a.role == 4)
    for name in FEATURE_NAMES[4:]:
        assert getattr(a, name)[assistant].eq(0).all()


def test_transformed_grid_metadata_tracks_new_coordinates():
    rotated = [list(row) for row in zip(*[[1, 2, 3], [4, 5, 6]][::-1])]
    tokens = [PINS.im_start, PINS.user, 10, *serialize_grid(rotated), PINS.eos]
    layout = build_layout(tokens, arm="rowcol")
    positions = [index for index, token in enumerate(tokens) if token in PINS.cells]
    assert [(layout.row[p].item(), layout.col[p].item()) for p in positions] == [
        (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)
    ]


def test_d4_color_roundtrip_and_metadata_alignment():
    grid = [[1, 0, 2], [1, 2, 2]]
    permutation = color_permutation(17)
    assert permutation[0] == 0
    assert sorted(permutation[1:]) == list(range(1, 10))
    for geometry in range(8):
        transformed = apply_colors(d4(grid, geometry), permutation)
        assert invert_prediction(
            transformed, geometry=geometry, permutation=permutation
        ) == grid
        tokens = [PINS.im_start, PINS.user, 10, *serialize_grid(transformed), PINS.eos]
        layout = build_layout(tokens)
        cells = [i for i, token in enumerate(tokens) if token in PINS.cells]
        assert len(cells) == len(grid) * len(grid[0])
        assert max(layout.row).item() == len(transformed)
        assert max(layout.col).item() == len(transformed[0])


def test_generated_state_coordinates_are_causal_and_clone_isolated():
    state = GeneratedGridState(grid_bucket=7)
    assert (state.metadata_for(3).row.item(), state.metadata_for(4).col.item()) == (1, 2)
    branch = state.clone()
    newline = branch.metadata_for(PINS.newline)
    assert newline.row.item() == newline.col.item() == 0
    next_cell = branch.metadata_for(2)
    assert (next_cell.row.item(), next_cell.col.item()) == (2, 1)
    assert state.row == 0 and state.col == 2
    eos = state.metadata_for(PINS.eos)
    assert eos.component.item() == 0


def test_shuffled_controls_are_deterministic_and_nontrivial():
    tokens = [
        PINS.im_start,
        PINS.user,
        10,
        *serialize_grid([[1, 1, 0], [2, 0, 3], [2, 2, 3]]),
        PINS.eos,
    ]
    normal = build_layout(tokens)
    shuffled_a = build_layout(tokens, arm="shuffled_rowcol", shuffle_seed=8)
    shuffled_b = build_layout(tokens, arm="shuffled_rowcol", shuffle_seed=8)
    assert torch.equal(shuffled_a.row, shuffled_b.row)
    assert not torch.equal(normal.row, shuffled_a.row) or not torch.equal(normal.col, shuffled_a.col)
    component = build_layout(tokens, arm="shuffled_component", shuffle_seed=9)
    assert sorted(component.component.tolist()) == sorted(normal.component.tolist())
    assert not torch.equal(component.component, normal.component)


class TinyCausal(torch.nn.Module):
    def __init__(self, hidden=12, vocab=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        return_dict=True,
        use_cache=False,
        **_,
    ):
        assert (input_ids is None) != (inputs_embeds is None)
        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        previous = (
            torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
            if past_key_values is None
            else past_key_values
        )
        cumulative = x.cumsum(1) + previous
        logits = self.head(cumulative)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        cache = cumulative[:, -1:] if use_cache else None
        return SimpleNamespace(logits=logits, loss=loss, past_key_values=cache)


def test_scale_zero_logits_and_nll_are_equivalent():
    torch.manual_seed(0)
    model = TinyCausal()
    adapter = AdditiveStructureAdapter(hidden_size_from_config(model.config))
    ids = torch.tensor([sequence()])
    layout = LayoutMetadata.stack([build_layout(ids[0])])
    errors = assert_scale_zero_equivalence(model, adapter, ids, layout)
    assert errors == {"max_abs_logit_error": 0.0, "abs_nll_error": 0.0}


def test_cached_and_uncached_structural_forward_match():
    torch.manual_seed(1)
    model = TinyCausal()
    adapter = AdditiveStructureAdapter(12)
    with torch.no_grad():
        adapter.gates["row"].fill_(0.5)
        adapter.gates["col"].fill_(-0.25)
    prefix = [
        PINS.im_start, PINS.user, 10, 1, 2, PINS.eos,
        PINS.im_start, PINS.assistant, 10,
    ]
    generated = [3, 4, 10, 5, 6]
    full_ids = torch.tensor([prefix + generated])
    full_layout = LayoutMetadata.stack([build_layout(full_ids[0], arm="rowcol")])
    forward = StructuralForward(model, adapter)
    full = forward(input_ids=full_ids, layout=full_layout, return_dict=True).logits
    prefix_ids = torch.tensor([prefix])
    prefix_layout = LayoutMetadata.stack([build_layout(prefix, arm="rowcol")])
    cached = forward(
        input_ids=prefix_ids, layout=prefix_layout, return_dict=True, use_cache=True
    )
    state = GeneratedGridState(grid_bucket=2)
    logits = []
    for token in generated:
        one_layout = LayoutMetadata.stack([state.metadata_for(token)])
        cached = forward(
            input_ids=torch.tensor([[token]]),
            layout=one_layout,
            past_key_values=cached.past_key_values,
            return_dict=True,
            use_cache=True,
        )
        logits.append(cached.logits[:, -1])
    assert torch.allclose(torch.stack(logits, 1), full[:, -len(generated) :], atol=1e-6)


def test_batched_multi_view_nll_matches_sequential_with_mixed_lengths():
    torch.manual_seed(2)
    model = TinyCausal()
    adapter = AdditiveStructureAdapter(12)
    with torch.no_grad():
        adapter.gates["row"].fill_(0.4)
        adapter.gates["col"].fill_(-0.3)
    tasks = [
        ([], [[1, 2]], [[3, 4]]),
        ([{"input": [[1]], "output": [[2]]}], [[3]], [[4], [5]]),
        ([], [[5], [6]], [[7]]),
        ([], [[8, 9]], [[1, 0]]),
    ]
    views = []
    for train, query, target in tasks:
        tokens, start = serialize_chatml(train, query, target)
        ids = torch.tensor([tokens])
        views.append((ids, LayoutMetadata.stack([build_layout(tokens)]), start))
    # Make the two equal-length rows carry different sidecar metadata. This
    # catches accidentally broadcasting one row's layout across the batch.
    changed = views[-1][1].row > 0
    views[-1][1].row[changed] = (views[-1][1].row[changed] % 30) + 1
    sequential = multi_view_nll(model, adapter, views)

    forward_shapes = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        forward_shapes.append(tuple(kwargs["inputs_embeds"].shape[:2]))
        return original_forward(*args, **kwargs)

    model.forward = recording_forward
    batched = batched_multi_view_nll(model, adapter, views, max_batch_size=2)
    assert sum(batch for batch, _ in forward_shapes) == len(views)
    assert sorted(width for batch, width in forward_shapes for _ in range(batch)) == sorted(
        ids.shape[1] for ids, _, _ in views
    )
    assert any(batch == 2 for batch, _ in forward_shapes)
    assert batched["view_count"] == sequential["view_count"]
    assert list(batched) == ["view_count", "mean_sum_nll", "views"]
    assert batched["mean_sum_nll"] == pytest.approx(
        sequential["mean_sum_nll"], abs=1e-6
    )
    for actual, expected in zip(batched["views"], sequential["views"]):
        assert list(actual) == [
            "sum_nll",
            "mean_nll",
            "first_token_nll",
            "per_row_nll",
            "token_count",
        ]
        assert actual["sum_nll"] == pytest.approx(expected["sum_nll"], abs=1e-6)
        assert actual["mean_nll"] == pytest.approx(expected["mean_nll"], abs=1e-6)
        assert actual["first_token_nll"] == pytest.approx(
            expected["first_token_nll"], abs=1e-6
        )
        assert actual["token_count"] == expected["token_count"]
        assert actual["per_row_nll"] == pytest.approx(
            expected["per_row_nll"], abs=1e-6
        )


def test_manifest_exclusion_stratification_and_hash():
    def task(size):
        grid = [[0] * size for _ in range(size)]
        return {"train": [{"input": grid, "output": grid}], "test": [{"input": grid}]}

    challenges = {f"{i:08x}": task(2 + i % 3) for i in range(140)}
    manifest = make_task_manifest(
        challenges, ["00000000"], seed=3, reserve_count=100, primary_count=24
    )
    assert "00000000" not in manifest["global_train_ids"]
    assert "00000000" not in manifest["reserved_ids"]
    assert len(manifest["reserved_ids"]) == 100
    assert len(manifest["primary_ids"]) == 24
    assert set(manifest["global_train_ids"]).isdisjoint(manifest["reserved_ids"])
    assert len(manifest["manifest_sha256"]) == 64


def test_manifest_builder_supports_preregistered_smoke_subset():
    def task():
        return {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }

    challenges = {f"{index:08x}": task() for index in range(12)}
    manifest = make_task_manifest(
        challenges,
        [],
        seed=3,
        reserve_count=10,
        primary_count=2,
        negative_count=1,
        replay_count=1,
    )
    assert len(manifest["primary_ids"]) == 2
    assert len(manifest["negative_control_ids"]) == 1
    assert len(manifest["replay_ids"]) == 1


def test_synthetic_split_budget_collisions_and_arm_identity():
    train, probe = make_synthetic_rows(seeds_per_family=10, seed=4)
    assert {row["family"] for row in train} == {row["family"] for row in probe}
    assert {row["seed"] for row in train}.isdisjoint({row["seed"] for row in probe})
    selected, manifest = fixed_token_budget(train, 1000)
    assert sum(row["loss_token_limit"] for row in selected) == 1000
    assert manifest["loss_tokens"] == 1000
    assert manifest == fixed_token_budget(train, 1000)[1]
    assert collision_audit(selected, probe, ["a"], ["b"])["passed"]


def test_loss_mask_matches_exact_budget_and_prompt_is_identical_across_arms():
    train = [{"input": [[1]], "output": [[2]]}]
    tokens, _ = serialize_chatml(train, [[3, 4]], [[5, 6]])
    row = {"input_ids": tokens, "loss_token_limit": 3}
    batches = {
        arm: collate_token_rows([row], arm=arm, shuffle_seed=99)
        for arm in ("text_control", "rowcol", "rowcol_component")
    }
    assert all(torch.equal(batch["input_ids"], batches["text_control"]["input_ids"]) for batch in batches.values())
    assert all(int((batch["labels"] != -100).sum()) == 3 for batch in batches.values())
    assert batches["text_control"]["layout"].row.eq(0).all()


def test_label_lifecycle_requires_all_candidate_hashes(tmp_path):
    lifecycle = LabelLifecycle(tmp_path, ["a", "b"], ["t1", "t2"])
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    for arm in ("a", "b"):
        for task in ("t1", "t2"):
            if (arm, task) != ("b", "t2"):
                lifecycle.record_candidates(arm, task, candidate)
    with pytest.raises(RuntimeError, match="labels remain sealed"):
        lifecycle.materialize_labels({"t1": [], "t2": []})
    lifecycle.record_candidates("b", "t2", candidate)
    path = lifecycle.materialize_labels({"t1": [], "t2": []})
    assert path.is_file()


def test_paired_statistics_report_discordant_ids_and_low_power():
    metrics = {
        "a": {"x": 0, "y": 1, "z": 0},
        "b": {"x": 1, "y": 0, "z": 0},
    }
    result = paired_contrast(metrics, "a", "b")
    assert result["wins"] == ["x"]
    assert result["losses"] == ["y"]
    assert result["ties"] == 1
    assert result["paired_randomization_p"] == 1.0
    assert result["low_power_warning"]


def test_report_item_aggregation_separates_generation_and_selection_miss():
    truth = [[1, 2]]
    ranked = [{"grid": [[0, 0]]}, {"grid": [[9, 9]]}, {"grid": truth}]
    selected_out = _score_item(ranked, truth)
    assert selected_out["truth_in_pool"]
    assert selected_out["truth_rank"] == 3
    assert selected_out["selection_miss"] and not selected_out["generation_miss"]
    absent = _score_item(ranked[:2], truth)
    assert absent["generation_miss"] and not absent["selection_miss"]
    assert absent["shape_accuracy"] == 1.0
    assert absent["unique_candidates"] == 2


def test_merged_task_inference_always_unmerges_on_failure(monkeypatch):
    calls = []

    class FakeTaskModel:
        def merge_adapter(self, *, safe_merge):
            calls.append(("merge", safe_merge))

        def unmerge_adapter(self):
            calls.append(("unmerge",))

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(gpu_arm_worker, "_evaluate_trained_task_impl", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        gpu_arm_worker.evaluate_trained_task(
            FakeTaskModel(),
            None,
            {},
            "task",
            "rowcol_component",
            {},
            pre={},
            post={},
            merge_for_inference=True,
        )
    assert calls == [("merge", True), ("unmerge",)]


def test_full_and_smoke_profiles_and_metadata_are_pinned():
    full = json.loads((RUN / "full-config.json").read_text())
    smoke = json.loads((RUN / "smoke-config.json").read_text())
    metadata = json.loads((RUN / "kernel-metadata.json").read_text())
    assert full["checkpoint"].endswith("/Transformers/bfloat16/1")
    assert full["arms"] == ["text_control", "rowcol", "rowcol_component"]
    assert full["global_loss_token_budget_per_arm"] == 1_500_000
    assert full["primary_tasks"] == 32 and full["minimum_primary_tasks"] == 24
    assert full["task_local"]["steps"] == 128
    assert full["task_local"]["rank"] == 256
    assert full["task_local"]["train_color_permutations"] * full["task_local"]["train_geometries"] == 128
    assert full["task_local"]["decode_color_permutations"] * full["task_local"]["decode_geometries"] == 16
    assert smoke["result_status"] == "smoke_only_never_report_as_final"
    assert smoke["global_loss_token_budget_per_arm"] < full["global_loss_token_budget_per_arm"]
    assert metadata["gpu_count"] == 4
    assert metadata["accelerator"] == "Nvidia L4 x4"
    assert metadata["model_sources"] == [full["checkpoint"]]


def test_monolith_contains_runtime_workers_and_no_solver_label_access():
    import base64
    from io import BytesIO
    import zipfile

    with zipfile.ZipFile(BytesIO(base64.b64decode(build_archive()))) as archive:
        names = set(archive.namelist())
        gpu_worker = archive.read("gpu_arm_worker.py").decode()
        nll_worker = archive.read("truth_nll_worker.py").decode()
    assert {
        "arc_solver/qwen2d_metadata.py",
        "arc_solver/qwen2d_adapter.py",
        "arc_solver/qwen2d_runtime.py",
        "gpu_arm_worker.py",
        "truth_nll_worker.py",
        "calibrate.py",
    } <= names
    assert "training_solutions" not in gpu_worker
    assert '"label_opened"' in nll_worker
    assert '"optimizer_steps_after_label": 0' in nll_worker


def test_checked_in_monolith_contains_current_bounded_corpus_preparation():
    monolith = (RUN / "qwen_2d_grid_adapter_l4.py").read_text()
    assert "from arc_solver.qwen2d_runtime import serialize_chatml_bounded" in monolith
    assert 'bounded["train"] = row.get("train", [])[dropped:]' in monolith
    assert '"min_prob":0.2' in monolith


def test_checked_in_monolith_embeds_current_qwen2d_sources():
    import ast
    import base64
    from io import BytesIO
    import zipfile

    tree = ast.parse((RUN / "qwen_2d_grid_adapter_l4.py").read_text())
    encoded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SOURCE_ARCHIVE"
            for target in node.targets
        )
    )
    with zipfile.ZipFile(BytesIO(base64.b64decode(encoded))) as archive:
        for relative in (
            "arc_solver/qwen2d_metadata.py",
            "arc_solver/qwen2d_runtime.py",
        ):
            assert archive.read(relative) == (ROOT / relative).read_bytes()
