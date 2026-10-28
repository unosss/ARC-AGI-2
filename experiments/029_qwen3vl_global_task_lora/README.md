# Experiment 029: Global Perception + Task LoRA

Status: implementation in progress.

## Preregistration amendment

Before any task adapter training, candidate generation, or solution access, the
exhaustive sequence preflight measured 256/6,272 augmented supervised examples
above the original 4,096-token limit; the maximum was 4,150. The fixed limit
was therefore increased to 4,352 tokens on 2026-10-24. No task, augmentation,
optimizer step, target token, beam, or generation token was removed. The
preflight remains fail-closed and must report zero truncations before training.

## Question

Does a shared grid-perception adaptation improve ARC task solving after the
model receives a fresh LoRA trained only on that task's demonstrations?

The experiment separates two stages:

1. global perception adaptation from Experiment 028;
2. task-time training on the current ARC task.

## Fixed paired comparison

- `base_task_lora`: unadapted Qwen3-VL plus a fresh task LoRA.
- `perception_task_lora`: the frozen Experiment 028 language perception
  adapter and the same fresh task LoRA active additively.

The global adapter is never merged into quantized weights and never receives a
gradient during task training. Every arm/task starts from identical task-LoRA
initialization. No state is carried between tasks.

The fixed 16-task hard set is inherited from Experiment 024. It was selected
before this intervention and has headroom. It is deliberately difficult and
is not an unbiased estimate of ARC population accuracy.

## Task-time training

For each task, exactly 128 optimizer updates are performed:

- 16 deterministic color permutations, preserving color 0;
- all 8 D4 rotations/reflections;
- one update for every color/geometry combination;
- every train pair contributes to that update and losses are averaged.

Input is the rendered ARC grid plus its numeric JSON. The target is only the
output grid JSON. Task LoRA uses rank 64, alpha 32, learning rate `1e-4`,
NF4 base weights and bf16 compute.

## Generation and label boundary

Each test input is decoded with deterministic two-beam search. At most two
unique rectangular grids with colors 0–9 and dimensions 1–30 are retained.
Raw text, token counts, parsed grids, train reconstructions, adapter hashes,
runtime and memory are persisted.

Workers receive challenge data only. The finalizer refuses to open the
solution file until all 32 arm/task candidate files exist and match their
manifests. Test outputs are then used only for post-hoc scoring.

Primary metric is task-level top-2 exact. Secondary metrics are item exact,
cell accuracy, valid-grid rate, train reconstruction, paired wins/losses,
runtime and memory.

## Feasibility gate

Both arms run the complete 128-step pilot on `135a2760`. The projected
sequential runtime for all 32 arm/task runs, with a 25% margin, must be at most
28,800 seconds, and each peak allocation must be at most 22 GiB. Failure aborts
the full experiment rather than reducing tasks, steps, transforms, beams or
generation length.

## Run

The pinned challenge and solution files are the 1,016-task merged artifacts
recorded by Experiment 024; only the preregistered 16 IDs are read.

```bash
bash experiments/029_qwen3vl_global_task_lora/runpod_setup.sh
python experiments/029_qwen3vl_global_task_lora/run.py \
  --config experiments/029_qwen3vl_global_task_lora/config.json \
  --challenge /path/to/merged_challenges.json \
  --solutions /path/to/merged_solutions.json \
  --global-adapter /root/experiment_028/runs/language_lora/adapter \
  --output-root experiments/029_qwen3vl_global_task_lora/runpod_output
```

The orchestrator first runs the real-model adapter integrity probe and both
complete pilot arms. It proceeds through all tasks only if both feasibility
gates pass. Resume accepts artifacts only when code, config, data and adapter
fingerprints still match; stale or partial state fails closed.

## Result

The full paired run completed in 7 h 55 min 52 s on one RTX 4090.

- `base_task_lora`: task top-2 0/16, item top-2 0/28, mean best-cell
  accuracy 0.4399.
- `perception_task_lora`: task top-2 0/16, item top-2 0/28, mean best-cell
  accuracy 0.5621.
- Paired task result: 0 wins, 0 losses, 16 ties.
- Valid-item rate was 0.9286 for both arms.
- Peak allocated memory was 14.46 GiB for base and 14.59 GiB for perception.

Thus the frozen perception checkpoint improved the secondary cell-level
similarity by 0.1222, but produced no exact item or task rescue. This rejects a
claim that this particular perception checkpoint plus additive task LoRA is
sufficient to improve exact ARC solving under the fixed budget.

Failure staging also shows that task-time fitting itself remains incomplete:
base reconstructed every train pair on 8/16 tasks and perception on 7/16.
Among those fitted tasks, neither arm generalized to an exact test output.
Per-task details and raw metrics are in `runpod_output/REPORT.md` and
`runpod_output/report.json`.

