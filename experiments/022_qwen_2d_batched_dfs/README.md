# Experiment 022: metadata-aware batched Turbo DFS

Status: Phase A completed; failed the preregistered parity and throughput
criteria. No result-bearing paired ablation was started.

## Phase A result

The full benchmark ran on Kaggle L4 with task `484b58aa`, test item 0, all 16
views, the original per-view budgets, and three timing repetitions.

- Sequential median: **747.68 s**
- Batched median: **396.27 s**
- Median speedup: **1.887x**
- Required speedup: **2.5x**
- Real-Qwen batch-size-one parity: exact
- Batch-four candidate parity: 15/16 views
- Batch-four logical counters: 15/16 views
- Largest cumulative NLL difference: **0.11048** (view 15)
- Held-out test outputs opened: no
- ARC accuracy interpretation: forbidden

The batched implementation reduced decode wall time by about 47%, but did not
meet the preregistered throughput threshold. More importantly, batching changed
floating-point logits slightly; over long candidate sequences those differences
accumulated in NLL. View 15 crossed a constrained-search decision boundary and
therefore explored a different candidate set and logical amount of work.

The smoke run had exact parity at 16 nodes/view, demonstrating that the cache
and metadata path works mechanically. The full run shows why the smoke could
not establish semantic equivalence at the real 4,096-node budget.

Artifacts:

- Smoke:
  `kaggle_runs/qwen_2d_batched_dfs_l4/outputs/smoke_v001/experiment_022/phase_a_report.json`
- Full:
  `kaggle_runs/qwen_2d_batched_dfs_l4/outputs/full_v002/experiment_022/phase_a_report.json`

Per the stopping rule, Experiment 022 does not proceed to a result-bearing
paired accuracy ablation. Batched DFS can still be treated as a distinct search
algorithm, as in public NVARC, but not as an algebraically identical replacement
for Experiment 021's sequential DFS.

## Why this is a separate experiment

Experiment 021 showed that the 2D adapter fits on an L4 after activation
checkpointing, but its result-bearing protocol projected to 202.5 hours for the
24-task fallback. A major implementation difference from public NVARC was that
all 16 augmented decode views were searched one at a time. Public NVARC groups
equal-length views and advances up to four DFS lanes in one model call.

Experiment 021 remains unchanged. Experiment 022 first restores that batching
property while preserving token-aligned structural metadata and per-view search
semantics.

## Phase A question

Can four equal-length augmented views share cached model forwards without
changing each view's constrained DFS result, metadata, or logical budget?

This phase is an engine validation and throughput experiment. It is not an ARC
accuracy experiment and does not open held-out test outputs.

## Batched search contract

- Maximum physical batch size: 4.
- Prefixes in one batch must have exactly equal token length.
- The 16 views are grouped in public-NVARC geometry classes:
  `(identity, rotate180)`, `(rotate90, rotate270)`,
  `(transpose, transpose+rotate180)`, and
  `(transpose+rotate90, transpose+rotate270)`, with both color permutations in
  each group.
- Each lane retains its own cumulative NLL, rectangular grid state, candidate
  queue, row/column shuffle seed, candidates, and stop reason.
- The strict cumulative threshold remains `score < -log(0.2)`.
- Each lane retains 4,096 nodes, 12,288 logical model-forward tokens, at most
  128 candidates, and at most 931 generated tokens.
- Inactive lanes receive pad token 13 and all-null sidecar metadata. Padding is
  measured as physical work but is not charged to any lane's logical budget.
- A shared mutable `DynamicCache` is cropped back to the parent sequence length
  after every recursive child batch.
- Generated component features remain null. No output shape, component, or
  held-out label enters search.

## Required parity checks

1. Batch size 1 exactly matches the existing single-view DFS for candidates,
   NLLs, nodes, forward tokens, and stop reason.
2. Four scripted lanes exactly match four independent single-view runs.
3. Candidate, node, forward-token, and exhaustion stops can occur independently.
4. Every inactive lane has pad token and all-zero metadata.
5. DynamicCache is restored before every sibling expansion.
6. Causal generated metadata matches completed serialization for all five arms.
7. All 16 real view specifications occur exactly once and no batch mixes prefix
   lengths.

## Phase A GPU criteria

- The exact grids15 checkpoint and tokenizer contract must pass.
- A real-Qwen batch-1 run must match the existing engine.
- Batched and independent runs use identical view definitions and per-view
  logical budgets.
- Report logical work separately from physical batch calls and padding.
- Median batched decode throughput must improve by at least 2.5× on the same L4
  and selected label-blind prompts.
- Any candidate difference near the probability threshold is recorded; it is
  never silently treated as speed-only equivalence.

## Result-bearing follow-up

Batching alone cannot make the Experiment 021 protocol fit. The v6 calibration
already gives a lower bound above 38 hours before decoding and rescoring.
Therefore Experiment 022 does not silently reduce tasks, views, nodes, LoRA
steps, controls, or global tokens.

If Phase A succeeds, a result-bearing protocol must be preregistered separately
with:

- batched DFS and batched teacher-forced rescoring;
- immutable reusable global-arm checkpoints;
- persisted task adapters so sealed-label NLL does not retrain them;
- fixed arm/task shards across multiple Kaggle runs;
- candidate hashes from every shard before any held-out label is opened;
- a complete measured runtime projection for every shard.

If those conditions cannot fit available compute, the paired accuracy
experiment remains unrun rather than being replaced by an underpowered toy.

## Artifacts

- `preregistered_manifest.json`: immutable Phase A conditions.
- `arc_solver/qwen2d_runtime.py`: single-view and batched DFS engines.
- `tests/test_qwen2d_batched_dfs.py`: semantic/cache/metadata parity tests.
- `kaggle_runs/qwen_2d_batched_dfs_l4/`: Phase A L4 package.
