import math
from types import SimpleNamespace

import pytest
import torch

from arc_solver.qwen2d_adapter import AdditiveStructureAdapter
from arc_solver.qwen2d_metadata import GeneratedGridState, LayoutMetadata, PINS, build_layout
from arc_solver.qwen2d_runtime import DFSBudget, deterministic_cached_dfs


PREFIX = [PINS.im_start, PINS.assistant, PINS.newline]


def _logits(probabilities):
    remaining = 1.0 - sum(probabilities.values())
    unspecified = [token for token in range(16) if token not in probabilities]
    assert remaining > 0 and unspecified
    values = torch.full((16,), remaining / len(unspecified))
    for token, probability in probabilities.items():
        values[token] = probability
    return values.log()


class ScriptedLegacyModel(torch.nn.Module):
    def __init__(self, scripts):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(16), freeze=True)
        self.scripts = scripts

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds,
        past_key_values=None,
        return_dict=True,
        use_cache=True,
        **_,
    ):
        del return_dict
        ids = inputs_embeds.argmax(dim=-1)
        history = ids if past_key_values is None else torch.cat((past_key_values[0], ids), dim=1)
        suffix = tuple(int(token) for token in history[0, len(PREFIX) :])
        logits = self.scripts.get(suffix, _logits({0: 0.1}))
        output_logits = logits.to(inputs_embeds.device).view(1, 1, -1).expand(
            1, inputs_embeds.shape[1], -1
        )
        return SimpleNamespace(
            logits=output_logits,
            past_key_values=(history,) if use_cache else None,
        )


def _run(model, *, budget):
    prefix_ids = torch.tensor([PREFIX])
    prefix_layout = LayoutMetadata.stack([build_layout(PREFIX, arm="rowcol")])
    adapter = AdditiveStructureAdapter(16)
    return deterministic_cached_dfs(
        model,
        adapter,
        prefix_ids,
        prefix_layout,
        grid_bucket=1,
        budget=budget,
        scale=0.0,
    )


def _reference_dfs(scripts, min_prob):
    cutoff = -math.log(min_prob)
    found = []
    nodes = 0

    def visit(tokens, score, state):
        nonlocal nodes
        log_probs = torch.log_softmax(scripts[tokens], dim=-1)
        ranked = sorted(
            state.allowed_tokens(), key=lambda token: (-float(log_probs[token]), token)
        )
        for token in ranked:
            child_score = score - float(log_probs[token])
            if not child_score < cutoff:
                continue
            nodes += 1
            child_state = state.clone()
            child_state.metadata_for(token)
            child = tokens + (token,)
            if token == PINS.eos:
                found.append({"tokens": list(child), "nll": child_score})
            elif len(child) < 2:
                visit(child, child_score, child_state)

    visit((), 0.0, GeneratedGridState(grid_bucket=1))
    return sorted(found, key=lambda item: (item["nll"], item["tokens"])), nodes


def test_scripted_logits_match_strict_cumulative_nll_reference():
    scripts = {
        (): _logits({1: 0.6, 2: 0.3}),
        (1,): _logits({PINS.eos: 0.5, PINS.newline: 0.3}),
        (2,): _logits({PINS.eos: 0.5, PINS.newline: 0.3}),
    }
    expected_candidates, expected_nodes = _reference_dfs(scripts, min_prob=0.2)
    result = _run(
        ScriptedLegacyModel(scripts),
        budget=DFSBudget(
            max_nodes=100,
            max_model_forward_tokens=100,
            max_candidates=10,
            max_new_tokens=2,
            min_prob=0.2,
        ),
    )
    assert result.candidates == expected_candidates
    assert result.nodes_used == expected_nodes == 3
    assert [item["tokens"] for item in result.candidates] == [[1, PINS.eos]]


def test_cumulative_nll_equal_to_cutoff_is_pruned():
    root = _logits({1: 0.6, 2: 0.3})
    exact_probability = math.exp(float(torch.log_softmax(root, dim=-1)[1]))
    result = _run(
        ScriptedLegacyModel({(): root}),
        budget=DFSBudget(
            max_nodes=10,
            max_model_forward_tokens=20,
            max_candidates=1,
            max_new_tokens=1,
            min_prob=exact_probability,
        ),
    )
    assert result.nodes_used == 0
    assert result.candidates == []


def test_rectangular_grammar_skips_ragged_eos_without_spending_candidate():
    scripts = {
        (): _logits({1: 0.99}),
        (1,): _logits({2: 0.99}),
        (1, 2): _logits({PINS.newline: 0.99}),
        (1, 2, PINS.newline): _logits({3: 0.99}),
        # EOS is highest but illegal until this second row reaches width two.
        (1, 2, PINS.newline, 3): _logits({PINS.eos: 0.6, 4: 0.39}),
        (1, 2, PINS.newline, 3, 4): _logits({PINS.eos: 0.99}),
    }
    result = _run(
        ScriptedLegacyModel(scripts),
        budget=DFSBudget(
            max_nodes=100,
            max_model_forward_tokens=100,
            max_candidates=1,
            max_new_tokens=6,
            min_prob=0.2,
        ),
    )
    assert [item["tokens"] for item in result.candidates] == [
        [1, 2, PINS.newline, 3, 4, PINS.eos]
    ]
    assert result.nodes_used == 6
    assert result.stopped_by == "candidate_budget"


def test_generated_grid_state_enforces_rectangles_bounds_and_metadata():
    state = GeneratedGridState(grid_bucket=4, max_dim=3)
    metadata = [state.metadata_for(token) for token in (1, 2, PINS.newline, 3)]
    assert [(item.row.item(), item.col.item()) for item in metadata] == [
        (1, 1),
        (1, 2),
        (0, 0),
        (2, 1),
    ]
    with pytest.raises(ValueError, match="invalid generated grid"):
        state.metadata_for(PINS.eos)
    with pytest.raises(ValueError, match="invalid generated grid"):
        state.metadata_for(PINS.newline)
    state.metadata_for(4)
    with pytest.raises(ValueError, match="invalid generated grid"):
        state.metadata_for(5)

    width_limit = GeneratedGridState(grid_bucket=1)
    for _ in range(30):
        width_limit.metadata_for(0)
    with pytest.raises(ValueError, match="invalid generated grid"):
        width_limit.metadata_for(0)
    width_limit.metadata_for(PINS.eos)

    height_limit = GeneratedGridState(grid_bucket=1)
    for row in range(30):
        height_limit.metadata_for(0)
        if row < 29:
            height_limit.metadata_for(PINS.newline)
    with pytest.raises(ValueError, match="invalid generated grid"):
        height_limit.metadata_for(PINS.newline)
    height_limit.metadata_for(PINS.eos)


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
def test_generated_metadata_matches_completed_serialization(arm):
    seed = 731
    prefix = [
        PINS.im_start,
        PINS.user,
        PINS.newline,
        1,
        PINS.eos,
        PINS.im_start,
        PINS.assistant,
        PINS.newline,
    ]
    generated = [2, 3, PINS.newline, 4, 5, PINS.eos]
    complete = build_layout(prefix + generated, arm=arm, shuffle_seed=seed)
    state = GeneratedGridState(
        grid_bucket=2, arm=arm, shuffle_seed=seed
    )
    pieces = [state.metadata_for(token) for token in generated]
    for name in complete.as_dict():
        assert torch.equal(
            torch.cat([getattr(piece, name) for piece in pieces]),
            getattr(complete, name)[len(prefix):],
        )


def test_unsupported_mutable_cache_is_rejected():
    class ListCacheModel(ScriptedLegacyModel):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            output.past_key_values = []
            return output

    with pytest.raises(TypeError, match="unsupported mutable cache type"):
        _run(
            ListCacheModel({(): _logits({1: 0.9})}),
            budget=DFSBudget(10, 20, 1, max_new_tokens=2),
        )


def test_dynamic_cache_is_cropped_before_root_sibling():
    transformers = pytest.importorskip("transformers")
    DynamicCache = transformers.DynamicCache

    class DynamicCacheModel(torch.nn.Module):
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
                self.child_cache_lengths.append((int(ids.item()), before))
            states = ids.to(dtype=torch.float32)[:, None, :, None]
            cache.update(states, states, layer_idx=0)
            probabilities = (
                {1: 0.45, 2: 0.45}
                if before == 0
                else {PINS.eos: 0.9}
            )
            logits = _logits(probabilities).view(1, 1, -1).expand(
                1, ids.shape[1], -1
            )
            return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)

    model = DynamicCacheModel()
    result = _run(
        model,
        budget=DFSBudget(
            max_nodes=20,
            max_model_forward_tokens=20,
            max_candidates=2,
            max_new_tokens=2,
            min_prob=0.2,
        ),
    )
    assert [item["tokens"] for item in result.candidates] == [
        [1, PINS.eos],
        [2, PINS.eos],
    ]
    assert model.child_cache_lengths == [(1, len(PREFIX)), (2, len(PREFIX))]
