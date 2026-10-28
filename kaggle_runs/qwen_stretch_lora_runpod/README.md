# Experiment 037: capped row/column stretch LoRA

Paired comparison on the same frozen Qwen3-4B + DiARC stack:

- `control`: 16 color permutations × 8 D4 views.
- `stretch`: the same 128 views, cycling through identity, row, column, and
  row+column repetition during task-local LoRA training.

ARC grids cannot exceed 30×30. Each selected axis position is duplicated once;
when literal 2× scaling would exceed 30, evenly spaced positions are duplicated
until the axis reaches 30. Decoding and rescoring remain canonical and identical
between arms, so this experiment tests training augmentation only.

```bash
PYTHONPATH=. python kaggle_runs/qwen_stretch_lora_runpod/run_experiment.py \
  --hard-exploratory \
  --challenges /path/to/arc-agi_training_challenges.json \
  --solutions /path/to/arc-agi_training_solutions.json \
  --model-dir /path/to/qwen3_4b_grids15_sft139 \
  --adapter /path/to/diarc-adapter \
  --work /tmp/experiment_037_work \
  --output /tmp/experiment_037_output
```
