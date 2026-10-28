import time
from types import SimpleNamespace

import pytest
import torch

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter
from arc_solver.qwen2d_metadata import GeneratedGridState, LayoutMetadata, PINS, build_layout
from arc_solver.qwen2d_runtime import (
    DFSBudget,
    batched_deterministic_cached_dfs,
    deterministic_cached_dfs,
)


def _logits(probabilities):
    remaining = 1.0 - sum(probabilities.values())
    unspecified = [token for token in range(16) if token not in probabilities]
    values = torch.full((16,), remaining / len(unspecified))
    for token, probability in probabilities.items():
        values[token] = probability
    return values.log()


def _prefix(marker):
    return [
        PINS.im_start,
        PINS.user,
        PINS.newline,
        marker,
        PINS.eos,
        PINS.im_start,
        PINS.assistant,
        PINS.newline,
    ]


class ScriptedBatchModel(torch.nn.Module):
    def __init__(self, scripts):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(16), freeze=True)
        self.scripts = scripts
        self.token_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, past_key_values=None, use_cache=True, **_):
        ids = inputs_embeds.argmax(dim=-1)
        self.token_calls.append(ids.detach().cpu().clone())
        history = ids if past_key_values is None else torch.cat((past_key_values[0], ids), dim=1)
        rows = []
        for row in history:
            marker = int(row[3])
            suffix = tuple(int(token) for token in row[len(_prefix(marker)) :] if token != PINS.pad)
            rows.append(self.scripts[marker].get(suffix, _logits({0: 0.1})))
        logits = torch.stack(rows).to(inputs_embeds.device)[:, None, :].expand(
            len(rows), ids.shape[1], -1
        )
        return SimpleNamespace(
            logits=logits,
            past_key_values=(history,) if use_cache else None,
        )


class RecordingAdapter(AdditiveStructureAdapter):
    def __init__(self):
        super().__init__(16)
        self.layouts = []

    def forward(self, layout, *, scale=1.0):
        self.layouts.append(
            LayoutMetadata(**{name: value.detach().cpu().clone() for name, value in layout.as_dict().items()})
        )
        return super().forward(layout, scale=scale)


def _inputs(markers, arms):
    prefixes = [_prefix(marker) for marker in markers]
    ids = torch.tensor(prefixes)
    layouts = LayoutMetadata.stack(
        [build_layout(prefix, arm=arm, shuffle_seed=100 + lane) for lane, (prefix, arm) in enumerate(zip(prefixes, arms))]
    )
    return ids, layouts


def _batch_run(model, markers, budgets, arms=None, adapter=None):
    arms = arms or ["rowcol_component"] * len(markers)
    ids, layouts = _inputs(markers, arms)
    return batched_deterministic_cached_dfs(
        model,
        adapter or AdditiveStructureAdapter(16),
        ids,
        layouts,
        grid_buckets=[marker + 1 for marker in markers],
        budgets=budgets,
        arms=arms,
        shuffle_seeds=[100 + marker for marker in markers],
        scale=0.0,
    )


def _single_run(scripts, marker, budget, arm="rowcol_component"):
    ids, layout = _inputs([marker], [arm])
    return deterministic_cached_dfs(
        ScriptedBatchModel(scripts),
        AdditiveStructureAdapter(16),
        ids,
        layout,
        grid_bucket=marker + 1,
        budget=budget,
        scale=0.0,
        generated_arm=arm,
        generated_shuffle_seed=100 + marker,
    )


def _scripts(markers):
    return {
        marker: {
            (): _logits({1: 0.55, 2: 0.3}),
            (1,): _logits({PINS.eos: 0.8}),
            (2,): _logits({PINS.eos: 0.8}),
        }
        for marker in markers
    }


def test_batch_size_one_has_exact_logical_parity():
    scripts = _scripts([0])
    budget = DFSBudget(50, 50, 2, max_new_tokens=2, min_prob=0.2)
    expected = _single_run(scripts, 0, budget)
    actual = _batch_run(ScriptedBatchModel(scripts), [0], [budget])

    assert actual.results == [expected]
    assert actual.telemetry.physical_batch_size == 1
    assert actual.telemetry.physical_forward_tokens == expected.model_forward_tokens
    assert actual.telemetry.active_lanes_per_call == (1, 1, 1)


def test_four_lanes_match_independent_dfs_and_stop_independently():
    markers = [0, 1, 2, 3]
    scripts = _scripts(markers)
    prefix_length = len(_prefix(0))
    budgets = [
        DFSBudget(0, 50, 2, max_new_tokens=2, min_prob=0.2),
        DFSBudget(50, prefix_length, 2, max_new_tokens=2, min_prob=0.2),
        DFSBudget(50, 50, 1, max_new_tokens=2, min_prob=0.2),
        DFSBudget(
            50,
            50,
            2,
            max_new_tokens=2,
            min_prob=0.2,
            deadline_epoch=time.time() - 1,
        ),
    ]
    expected = [
        _single_run(scripts, marker, budget)
        for marker, budget in zip(markers, budgets)
    ]
    adapter = RecordingAdapter()
    model = ScriptedBatchModel(scripts)
    actual = _batch_run(model, markers, budgets, adapter=adapter)

    assert actual.results == expected
    assert [result.stopped_by for result in actual.results] == [
        "node_budget",
        "forward_token_budget",
        "candidate_budget",
        "run_deadline",
    ]
    assert actual.telemetry.active_lanes_per_call == (4, 1)
    assert actual.telemetry.physical_forward_tokens == 4 * prefix_length + 4
    for ids, layout in zip(model.token_calls[1:], adapter.layouts[1:]):
        pad = ids.eq(PINS.pad)
        assert pad.any()
        for values in layout.as_dict().values():
            assert torch.all(values[pad] == 0)


@pytest.mark.parametrize(
    "arm",
    [
        "text_control",
        "rowcol",
        "rowcol_component",
        "shuffled_rowcol",
        "shuffled_component",
    ],
)
def test_generated_metadata_is_correct_for_every_arm(arm):
    scripts = _scripts([0])
    adapter = RecordingAdapter()
    budget = DFSBudget(20, 20, 1, max_new_tokens=2, min_prob=0.2)
    _batch_run(
        ScriptedBatchModel(scripts),
        [0],
        [budget],
        arms=[arm],
        adapter=adapter,
    )

    expected_state = GeneratedGridState(
        grid_bucket=1, arm=arm, shuffle_seed=100
    )
    expected = expected_state.metadata_for(1)
    generated = adapter.layouts[1]
    for name in expected.as_dict():
        assert torch.equal(getattr(generated, name)[0], getattr(expected, name))


def test_rejects_oversized_or_mismatched_batches():
    scripts = _scripts([0, 1, 2, 3])
    budget = DFSBudget(10, 20, 1)
    ids, layouts = _inputs([0, 1, 2, 3], ["rowcol"] * 4)
    with pytest.raises(ValueError, match="exceeds limit"):
        batched_deterministic_cached_dfs(
            ScriptedBatchModel(scripts),
            AdditiveStructureAdapter(16),
            ids,
            layouts,
            grid_buckets=[1, 2, 3, 4],
            budgets=[budget] * 4,
            arms=["rowcol"] * 4,
            shuffle_seeds=[None] * 4,
            max_physical_batch=3,
        )
    with pytest.raises(ValueError, match="must match batch"):
        batched_deterministic_cached_dfs(
            ScriptedBatchModel(scripts),
            AdditiveStructureAdapter(16),
            ids,
            layouts,
            grid_buckets=[1],
            budgets=[budget] * 4,
            arms=["rowcol"] * 4,
            shuffle_seeds=[None] * 4,
        )


def test_per_lane_time_stop_and_unsupported_mutable_cache():
    scripts = _scripts([0])
    timed_out = DFSBudget(
        10, 20, 1, emergency_seconds=0.0, deadline_epoch=time.time() + 60
    )
    result = _batch_run(ScriptedBatchModel(scripts), [0], [timed_out])
    assert result.results[0].stopped_by == "emergency_time_cap"
    assert result.results[0].nodes_used == 0

    class ListCacheModel(ScriptedBatchModel):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            output.past_key_values = []
            return output

    with pytest.raises(TypeError, match="unsupported mutable cache type"):
        _batch_run(
            ListCacheModel(scripts),
            [0],
            [DFSBudget(10, 20, 1)],
        )


def test_dynamic_cache_is_restored_before_batched_siblings():
    transformers = pytest.importorskip("transformers")
    DynamicCache = transformers.DynamicCache

    class DynamicBatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(16), freeze=True)
            self.child_cache_lengths = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, past_key_values=None, use_cache=True, **_):
            ids = inputs_embeds.argmax(dim=-1)
            cache = DynamicCache() if past_key_values is None else past_key_values
            before = cache.get_seq_length()
            if ids.shape[1] == 1:
                self.child_cache_lengths.append((ids.flatten().tolist(), before))
            states = ids.to(dtype=torch.float32)[:, None, :, None]
            cache.update(states, states, layer_idx=0)
            probabilities = {1: 0.45, 2: 0.45} if before == 0 else {PINS.eos: 0.9}
            row = _logits(probabilities)
            logits = row.view(1, 1, -1).expand(ids.shape[0], ids.shape[1], -1)
            return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)

    model = DynamicBatchModel()
    budget = DFSBudget(20, 30, 2, max_new_tokens=2, min_prob=0.2)
    result = _batch_run(model, [0, 1], [budget, budget])
    prefix_length = len(_prefix(0))

    assert [
        [candidate["tokens"] for candidate in item.candidates]
        for item in result.results
    ] == [
        [[1, PINS.eos], [2, PINS.eos]],
        [[1, PINS.eos], [2, PINS.eos]],
    ]
    assert model.child_cache_lengths == [
        ([1, 1], prefix_length),
        ([2, 2], prefix_length),
    ]


def test_dynamic_cache_inactive_lane_is_padded_and_restored():
    transformers = pytest.importorskip("transformers")
    DynamicCache = transformers.DynamicCache

    class UnevenDynamicBatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(16), freeze=True)
            self.markers = None
            self.child_calls = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, past_key_values=None, use_cache=True, **_):
            ids = inputs_embeds.argmax(dim=-1)
            cache = DynamicCache() if past_key_values is None else past_key_values
            before = cache.get_seq_length()
            if past_key_values is None:
                self.markers = ids[:, 3].tolist()
            else:
                self.child_calls.append((ids.flatten().tolist(), before))
            states = ids.to(dtype=torch.float32)[:, None, :, None]
            cache.update(states, states, layer_idx=0)
            rows = []
            for marker in self.markers:
                if before == 0 and marker == 0:
                    rows.append(_logits({1: 0.45, 2: 0.45}))
                else:
                    rows.append(_logits({PINS.eos: 0.9}))
            logits = torch.stack(rows)[:, None, :].expand(ids.shape[0], ids.shape[1], -1)
            return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)

    model = UnevenDynamicBatchModel()
    budget = DFSBudget(20, 30, 2, max_new_tokens=2, min_prob=0.2)
    result = _batch_run(model, [0, 1], [budget, budget])
    prefix_length = len(_prefix(0))

    assert [item["tokens"] for item in result.results[0].candidates] == [
        [1, PINS.eos],
        [2, PINS.eos],
    ]
    # EOS is not legal before a grid row starts, so this lane has no branch.
    assert result.results[1].candidates == []
    assert model.child_calls == [
        ([1, PINS.pad], prefix_length),
        ([2, PINS.pad], prefix_length),
    ]
