# Experiment 021: explicit 2D metadata for NVARC Qwen

Status: GPU calibration completed; preregistered full run was rejected as
infeasible before any result-bearing arm or held-out-label phase.

## Scientific question

Does token-aligned 2D structure improve the deployed NVARC pipeline based on the
exact Kaggle model source
`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`?

The upstream recipe used synthetic examples derived from ARC-AGI-2 evaluation.
Its absolute evaluation score is therefore not a clean estimate of unseen-task
generalization. Paired differences between arms hold the checkpoint, examples,
tokens, seeds, task LoRA, decode views, and search budgets fixed and isolate this
intervention better, but not perfectly: checkpoint contamination can interact
with the adapter.

## Preregistered hypotheses and decisions

- H1: `rowcol` improves primary task-level top-2 exact over `text_control`.
- H2: `rowcol_component` improves task-level top-2 exact over `rowcol`.
- Mechanistic support requires improved held-out perception probes and either
  better truth-in-pool or better truth rank. Lower teacher-forced NLL alone is
  insufficient.
- A hypothesis is supported only if its paired exact difference is positive,
  its 95% task bootstrap interval excludes zero, and the paired randomization
  p-value is at most 0.05. With 24–32 tasks, all conclusions will explicitly be
  described as low-power.
- A shuffled control matching or exceeding the corresponding true-metadata arm
  blocks a causal interpretation.
- Synthetic-probe gains will never be reported as ARC solve gains.

These rules were written before a GPU run. Final held-out outputs remain sealed
until every arm/task candidate pool is materialized and SHA-256 recorded.

## Run log

- Smoke v2 stopped at preflight and exposed the real tokenizer contract:
  every role segment begins with `im_start=14`, while Hugging Face reports a
  13-token base vocabulary and 16 total tokens.
- Smoke v3 passed tokenizer preflight but stopped because the pinned Kaggle
  Transformers build requires the deprecated-compatible `torch_dtype`
  argument rather than `dtype`.
- Smoke v4 completed all five arms on four concurrent CUDA workers. Scale-zero
  errors were exactly zero, replay hashes were deterministic, labels opened
  only after candidate hashes existed, truth-NLL recorded no post-label
  optimization/DFS, and the final report passed
  `jsonschema+internal_strict`. This run is operational evidence only; its
  two-task/rank-8 measurements are not ARC accuracy evidence.
- Full v5 was launched from the fixed result-bearing profile. Its calibration
  stopped before its first optimizer step: an 8,192-token rank-64 forward used
  21.48 GiB of PyTorch allocations on a 22.03 GiB L4 and failed on a further
  74 MiB allocation. No arm or label phase ran. Activation checkpointing was
  enabled afterward without changing data, budgets, trainable parameters, or
  objective; the next calibration must still select 32, use the preregistered
  24-task fallback, or abort before arm training.
- Full v6 completed calibration after activation checkpointing. Peak allocation
  was 11.95 GiB for rank-64 global training and 17.65 GiB for rank-256 task
  training, so memory was no longer the blocker. The measured 24-task
  projection was 729,015 seconds (202.5 hours), and the 32-task projection was
  832,945 seconds (231.4 hours), versus 41,451 seconds remaining. The gate
  correctly aborted before any arm ran or label opened. Unit measurements were
  116.79 seconds per 34,410 global loss tokens, 146.10 seconds per 16 task
  steps, 56.82 seconds per 512-node DFS view, 4.66 seconds per NLL view, and
  1.94 seconds per 1.49-GiB checkpoint.

The runtime result identifies a design bottleneck rather than an accuracy
result. Experiment 021 executes 16 DFS views sequentially; at the measured
rate, 4,096 nodes per view projects to roughly two hours per test item before
rescoring. Upstream NVARC instead batches four augmented views in Turbo DFS and
uses task/search wall-clock caps. A result-bearing successor must implement
metadata-aware batched DFS and recalibrate before fixing a new budget. Silently
reducing this experiment's nodes, tasks, views, or token budget would answer a
different question and is not treated as Experiment 021.

## Arms

All five arms allocate the same adapter parameter tensors and execute the same
forward path. This is allocated-parameter matching, not effective-capacity
matching: padding id 0 fixes the `text_control` contribution to zero.

1. `text_control`: all-null sidecar (allocated-parameter-matched null path).
2. `rowcol`: row, column, segment role, and grid bucket.
3. `rowcol_component`: row/column plus input-grid components.
4. `shuffled_rowcol`: negative control on 12 preregistered primary tasks.
5. `shuffled_component`: negative control on the same subset.

Components are 4-connected, same-color, non-background sets. Their features
include deterministic membership, size, bbox height/width, and a
color/permutation-invariant topology bucket. All assistant/output component
features are null. No output shape or component is taken from truth.
Color augmentation fixes background color 0 and permutes only colors 1–9 so
the component decomposition remains consistent with the displayed grid.

## Architecture

`arc_solver/qwen2d_metadata.py` parses the native grids15 ChatML stream.
It asserts cell ids `0..9`, newline `10`, user `11`, assistant `12`, pad `13`,
`im_start` `14`, and EOS `15` at runtime. Hugging Face reports 13 base tokens
and 16 total tokens. Metadata is a sidecar and does not add tokens. Padding is
null.

`arc_solver/qwen2d_adapter.py` reads hidden size from model config and adds
learned feature embeddings behind zero-initialized scalar gates:

`inputs_embeds = embed_tokens(input_ids) + delta`

The wrapped model never receives `input_ids` together with `inputs_embeds`.
Native position ids and 1D RoPE are unchanged. A mandatory scale-zero test
compares logits and NLL against the unwrapped model.

`arc_solver/qwen2d_runtime.py` is the common call path for global training,
task-local training, teacher-forced NLL, multi-view scoring, initial DFS prefix,
and every cached one-token DFS update. Generated cells receive coordinates from
a branch-local causal state; generated components remain null. Search equality
is based on nodes and model-forward tokens, with time only as an emergency cap.
As in upstream Turbo DFS, branches are also pruned unless cumulative sequence
probability remains strictly above `0.2`; only rectangular 1–30 × 1–30 grids
can consume the candidate budget.

Unsloth's trainer is not used because its `inputs_embeds` behavior is not
assumed safe. The run uses standard Transformers plus PEFT and fails if the
common structural forward cannot be used.

## Data and label lifecycle

Only official ARC-AGI-2 **training split train pairs** enter global training.
The reviewed 10 IDs and Experiment 013 phase-1 20 IDs are excluded. Seed
`2102101` deterministically reserves 120 eligible tasks; the rest form ARC LOO
global rows. Primary tasks are chosen before arm execution by round-robin over
label-blind strata based on train-pair dimensions, input component counts, and
train output-size relation.

The global corpus combines ARC train-pair LOO rows with deterministic families:
reconstruction, indexed cell, row/column extraction, transpose, D4,
translation, component count/bbox/membership, and copy/recolor selected
component. Every fifth family-local seed is probe-only. Manifests persist row
order, counts, hashes, and task/pair collision audits.

The full profile uses exactly 1,500,000 loss-bearing assistant tokens per arm.
Prompt rows and sequence lengths are identical across arms. The final partial
row is label-masked, not text-truncated.

## Training and evaluation

Global training uses a common rank-64 RSLoRA plus the trainable structural
adapter. Each arm checkpoint is saved separately. For every primary task, that
arm's global adapter is frozen, and a fresh task-local rank-256, alpha-32
RSLoRA is initialized. No task state carries over. Each task receives exactly
128 optimizer steps from 16 color permutations × 8 D4 views.

Before task adaptation, held-out probes report exact, valid-grid, and
teacher-forced NLL by reading/coordinate, D4/translation, and component family.
Primary evaluation uses 32 tasks if measured calibration projects at most
11.5 hours; otherwise the first preregistered 24 are used. Each task has 16
decode views and 8 rescore views. Each view receives 4,096 DFS nodes and a
12,288 total model-forward-token ceiling
(up to 8,192 prompt tokens plus 4,096 cached tokens), at most 128 candidates,
and a 900-second emergency cap. The upstream `score_kgmon` top-2 rule is
preserved.

Four fixed tasks are rerun twice with identical seeds for nondeterminism
diagnostics; repeats are not independent observations.

Metrics include task/item top-2 exact, truth in pool and rank, generation versus
selection miss, shape/cell accuracy, unique candidates, nodes/tokens used,
canonical and 16-view truth NLL, first-token and per-row NLL, pre/post task-LoRA
NLL, and probe metrics. Reports include B−A and C−B discordant task IDs,
bootstrap intervals, and paired randomization tests.

## Profiles and expected compute

`full-config.json` is the only result-bearing profile:

- 4 × NVIDIA L4, bf16, offline exact checkpoint;
- five global arms × 1.5M loss tokens (main arms scheduled first);
- 32 primary tasks × 3 arms, plus 12 tasks × 2 negative controls;
- 128 task-local steps, 16 decode views, 8 rescore views;
- deterministic DFS budgets above;
- calibration gate at 41,400 seconds and hard emergency cap at 42,600 seconds.

If calibration exceeds the gate, primary evaluation drops once to the
preregistered 24-task prefix before any arm runs. It does not reduce per-task
training or DFS budgets. Calibration measures longest-first global rows, a
label-blind longest selected task/test prompt, real checkpoint I/O, rank-64
and rank-256 peak CUDA memory, DFS, and teacher-forced NLL, then conservatively
projects rescoring, probes, and duplicate truth-NLL work. It aborts before arm
execution if the 24-task projection cannot fit.
`smoke-config.json` is explicitly non-inferential.

Global state is checkpointed every 64 optimizer steps and task state after each
candidate pool. Only the newest global optimizer checkpoint is retained.
Resume checks an arm/config/corpus/model fingerprint; task candidates are
atomically materialized before being reused.
Outputs are written under immutable `output_vNNN` directories.

## Files

- `preregistered_manifest.json`: exclusions and deterministic split rule.
- `report_schema.json`: required result fields.
- `kaggle_runs/qwen_2d_grid_adapter_l4/`: pinned configs, build script,
  generated monolithic script, and kernel metadata.
- `tests/test_qwen2d_grid_adapter.py`: parser, metadata, controls, parity,
  lifecycle, budget, statistics, and pin tests.
