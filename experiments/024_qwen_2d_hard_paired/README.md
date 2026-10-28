# Experiment 024: Qwen 2D hard-set paired comparison

Status: completed on one RunPod RTX 4090.

This repeats Experiment 023 on the fixed 16-task set used by Experiment 015.
The set was fixed before the 2D adapter existed; the public NVARC reproduction
solved 2/16 tasks, so it has measurable headroom. All 16 tasks are evaluated,
not only failures.

The control and `rowcol_component` arms use identical data and budgets. All
target task IDs are removed from global training. Primary outcomes are:

- control and metadata task-level top-2 exact;
- metadata-only rescues;
- control-only regressions;
- net paired difference.

This is a deliberately hard, enriched set and is not an unbiased estimate of
overall ARC accuracy.

## RunPod result

- `text_control`: 2/16 task-level top-2 exact (12.5%; 3 exact test items).
- `rowcol_component`: 2/16 task-level top-2 exact (12.5%; 4 exact test items).
- Paired metadata wins: 0. Paired metadata losses: 0. Net task difference: 0.
- Both arms solved `135a2760` and `1818057f`; all 16 task-level outcomes tied.
- End-to-end runtime: 14,899.87 seconds (4h 08m 19.87s).

Artifacts are in `runpod_output/`, including the report, preparation
manifests, all 32 task candidate payloads, arm completion hashes, and concise
worker logs.

The RunPod adaptation is reproducible from
`kaggle_runs/qwen_2d_hard_paired_runpod/`. It merged the pod's ARC 2026
training challenge/solution files with the exact 16 historical payloads and
labels, ran both arms sequentially on GPU 0, and preserved restart checkpoints
and completed candidates on the pod. The accepted production configuration
was unchanged: task-local merging and rescore batching remained disabled,
task-local training used 128 steps, decoding used 16 views, rescoring used 8
sequential views, all DFS budgets were retained, and each arm used exactly
1,500,000 global loss tokens. Only the operational emergency deadline was
expanded from 41,400 seconds to 604,800 seconds for RunPod.

This hard set is deliberately enriched and is not an unbiased estimate of
population ARC accuracy.
