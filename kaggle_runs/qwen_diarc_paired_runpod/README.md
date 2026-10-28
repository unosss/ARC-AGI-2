# Experiment 036 RunPod runner

This runner performs a paired, label-blind comparison on the exact
`sorokin/qwen3_4b_grids15_sft139` base checkpoint:

- `base`: fresh base plus a fresh task-local RSLoRA adapter.
- `diarc`: fresh base, frozen released
  `yyxdnmd/DiARC-adapters/qwen3-4b/arc-agi-2` adapter loaded as `diarc`, plus
  the same fresh task-local adapter. `diarc` and `task` are active additively,
  while only `task` is trainable.

Both workers use `text_control` metadata and a frozen, zero-scale structure
sidecar. They share task rows, seeds, optimizer settings, DFS limits, and
rescoring. Each arm runs in a separate process from a fresh base load; the
worker explicitly deletes and unloads adapters before exit.

## Modes

```bash
python kaggle_runs/qwen_diarc_paired_runpod/run_experiment.py \
  --smoke --challenges challenges.json --solutions solutions.json \
  --work /workspace/exp036-smoke --output /workspace/exp036-smoke-output

python kaggle_runs/qwen_diarc_paired_runpod/run_experiment.py \
  --calibrate --challenges challenges.json --solutions solutions.json \
  --model-dir /models/qwen3_4b_grids15_sft139 \
  --work /workspace/exp036-cal --output /workspace/exp036-cal-output

python kaggle_runs/qwen_diarc_paired_runpod/run_experiment.py \
  --hard-exploratory --challenges challenges.json --solutions solutions.json \
  --model-dir /models/qwen3_4b_grids15_sft139 \
  --work /workspace/exp036-hard --output /workspace/exp036-hard-output
```

`--hard-exploratory` defaults to task `195c6913`. `--adapter` may point at an
offline snapshot of the registered DiARC adapter. The smoke mode is CPU-only
and never imports or loads a model.

The parent writes a challenge-only worker file. Worker commands never contain
the solutions path. The parent verifies both candidate files and immutable
SHA-256 markers before its first solutions read, then writes `report.json`.
