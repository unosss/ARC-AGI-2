# Experiment 034: staged Qwen adapters

## Question and architecture

Does separating deterministic grid perception from object-level interpretation
reduce adapter interference? All arms use the same frozen Qwen checkpoint,
tokenizer, 2D metadata path, adapter parameterization, optimizer settings,
augmentations, task-local adaptation, decode/search limits, and label-blind
task panel. Only adapter staging and corpus order differ.

The synthetic families are frozen in `arc_solver/qwen2d_experiment.py`.
Perception contains reconstruction, indexed-cell, row/column extraction,
component count, bounding box, and membership. Interpretation contains
transpose, D4, translation, component copy/recolor, and ARC leave-one-out
transformations from the global-training partition. Stage 1 has exactly
600,000 loss-bearing tokens and stage 2 exactly 900,000. Cycling and a final
`loss_token_limit` make budgets exact. Synthetic probes are never cycled into
training and remain partitioned by stage.

The three primary arms are:

1. `staged_late`: train a perception-only adapter in the early third, then
   freeze it and train an interpretation-only adapter in the middle third.
   Each task receives a fresh task adapter in the late third.
2. `mixed_late`: use the same early/middle parameter counts and stage budgets,
   but train both global bands on the same deterministic balanced mixture of
   perception and interpretation rows. Its task adapter remains in the late
   third. This isolates objective specialization.
3. `staged_early_collision`: reuse the exact frozen global checkpoint from
   `staged_late`, but place the fresh task adapter in the early third, on the
   same modules as the perception adapter. Rank, layer count, data, and 128
   task steps match `staged_late`. This isolates problem-specific placement.

Adapters have separate parameters and optimizer states and remain unmerged.
The frozen early adapter stays active while the middle adapter learns; both
frozen global adapters stay active while a task adapter learns. Tensor digests
must prove that inactive adapters do not change. Arm initialization, row order,
seeds, checkpoint cadence, and total loss-bearing tokens are matched. The mixed
corpus is deterministically interleaved before fixed-token cycling.

## Blind panel and controls

The panel contains 32 tasks. At setup time, collect the task IDs explicitly
used by prior experiments from their preregistered manifests and fixed
`task_ids.json`/protocol panels, union them with the exclusions in
`preregistered_manifest.json`, and intersect that set with the mounted
challenge IDs. Run `make_task_manifest` with seed `34034`, reserve count 100,
and primary count 32. Its strata use only published training pairs; test
outputs and prior scores are forbidden. The resulting IDs and manifest hash
must be written before any arm starts.

Eight controls are selected from those 32 by the lowest SHA-256 of
`experiment-034-control:<task_id>`, with task ID as the tie-breaker. Controls
are therefore paired tasks, not an additional exposed panel. They receive the
adapter-order ablation and shuffled-family checks fixed in the run manifest;
control outcomes are diagnostic and are not pooled into the primary
three-arm test.

## Label lifecycle

Only challenge train pairs and test inputs may be mounted during selection,
training, decoding, and candidate ranking. Every arm/task candidate file and
all probe predictions are finalized and SHA-256 recorded. `LabelLifecycle`
must then verify the complete 3 x 32 candidate-hash matrix before a parent-only
evaluator may materialize labels. No optimizer step, decoding, candidate
replacement, threshold change, or task removal is allowed after labels open.
Failures remain failures; missing tasks are reported.

## Outcomes and decision rules

The primary metric is paired task-level top-2 exact accuracy on all 32 tasks.
The primary contrasts, fixed in order, are `staged_late - mixed_late` and
`staged_late - staged_early_collision`. Report mean paired differences,
task-bootstrap 95% intervals, exact/randomized paired p-values, wins, losses,
and ties. Holm correction applies across the two primary contrasts. A claim
requires a positive difference, corrected p <= 0.05, and a 95% interval
excluding zero.

Secondary metrics are top-1 task exact, item exact, cell accuracy, truth in
candidate pool, truth rank conditional on presence, generation versus
selection misses, held-out probe exact/cell accuracy by stage and family,
teacher-forced NLL, runtime, peak memory, and per-task discordance. Secondary
metrics are descriptive and cannot rescue a failed primary endpoint.

## Calibration gate

A label-free calibration run must complete tokenizer/scale-zero checks,
one optimizer step for each stage and composition mode, deterministic replay,
probe isolation and collision audits, and one decode without OOM or NaN. Using
measured throughput and peak allocation, the full 3 x 32 run plus eight-task
controls must project within the declared compute window with at least 25%
time and memory headroom. Calibration may tune batch size and gradient
accumulation while preserving architecture, examples, 600k/900k loss-token
budgets, and effective optimizer updates. Otherwise the experiment aborts
before primary training; there is no reduced-task fallback.

## Limits

Thirty-two paired tasks have low power and come from one available ARC source.
Public checkpoints may contain related tasks. Synthetic family boundaries are
an operational taxonomy, not proof of internal modules. Late composition
changes activation context and need not isolate semantic knowledge perfectly.
Top-2 exact depends on the fixed search and ranking pipeline. Probe gains,
NLL, cell accuracy, and controls do not constitute ARC solves, and no result
estimates private leaderboard performance.
