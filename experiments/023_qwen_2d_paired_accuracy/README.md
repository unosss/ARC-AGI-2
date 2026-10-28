# Experiment 023: Qwen 2D paired accuracy

Status: completed on Kaggle L4.

## Result

The full two-arm run completed in **11,025 seconds (3.06 hours)**.

- `text_control`: **8/8 tasks**, 10/10 test items top-2 exact.
- `rowcol_component`: **8/8 tasks**, 10/10 test items top-2 exact.
- Paired difference: **0 tasks**.
- Metadata wins/losses: **0/0**, with eight ties.
- Truth in generated pool: 10/10 for both arms.
- Candidate files for every task and arm existed before labels were opened.

Therefore this experiment does **not** show an accuracy improvement from the 2D
sidecar. It also does not show a regression at top-2 exact. The comparison hit a
ceiling: the control solved every selected task, so these eight tasks cannot
measure positive headroom.

There was a small rank difference below the primary metric: on task `17b866bd`
item 0, the control ranked truth first while the metadata arm ranked it second.
All other metadata truths ranked first. This is not an exact-score improvement.

The absolute 100% must not be interpreted as clean ARC generalization. The
starting grids15 checkpoint is known to have used ARC-AGI-2
evaluation-derived synthesis, and the selected tasks may be especially easy or
checkpoint-familiar. A follow-up intended to detect improvement must select
tasks where the fixed control actually fails, without using the metadata
arm's results or labels to make that selection.

Artifacts:

- `kaggle_runs/qwen_2d_paired_accuracy_l4/outputs/full_v003/experiment_023_output/report.json`
- Kaggle kernel version 3:
  `unosss/qwen-2d-paired-accuracy-l4x4`

## Question

Does the trained row/column/component sidecar improve top-2 ARC accuracy over
the same Qwen/RSLoRA pipeline with a null sidecar?

## Comparison

Only two arms are run:

- `text_control`: allocated adapter path, all metadata null.
- `rowcol_component`: causal row, column, grid and input-component metadata.

Both arms use the same checkpoint, global corpus and loss-token budget, task
IDs, optimizer settings, task-local steps, augmentations, batched 16-view DFS,
candidate rescoring, and top-2 selector. They run concurrently on separate L4s.

This is a directional eight-task comparison, not a significance claim. Unlike
Experiment 021, it omits probes, shuffled controls, replays, and post-label
truth-NLL retraining. Candidate files for both arms are completed before the
official training solutions are opened. Exact-match evaluation itself requires
no model execution.

## Fixed tasks

`0692e18c`, `0f63c0b9`, `12422b43`, `14754a24`, `15663ba9`,
`17b80ad2`, `17b866bd`, `1be83260`.

They are the first eight label-blind, stratified primary tasks from Experiment
021's previously fixed manifest. The 30 manually reviewed/earlier-evaluation
tasks remain excluded.

## Decision

The metadata arm is directionally better only if it has more task-level top-2
exact solves. Item exact, truth-in-pool, generation/selection misses, runtime,
and per-task discordance are also reported. Ties are reported as ties.
