# Experiment 034 RunPod package

This package runs the preregistered stage-separated Qwen comparison. The
worker receives only ARC challenges (training demonstrations and test inputs);
only the parent process accepts a solution path, after all 96 primary and 16
diagnostic-control candidate files have been hashed.

## Requirements

- One NVIDIA GPU with at least 24 GB memory
- A local grids15 Qwen checkpoint
- ARC challenge and solution JSON files
- Repository root available on `PYTHONPATH`

```bash
cd /home/unos/kaggle/ARC-AGI-2
bash kaggle_runs/qwen_staged_adapters_runpod/setup.sh
```

Set paths once:

```bash
PKG=kaggle_runs/qwen_staged_adapters_runpod
CHALLENGES=/data/arc-agi_training_challenges.json
SOLUTIONS=/secure/arc-agi_training_solutions.json
MODEL=/models/qwen3_4b_grids15
WORK=/workspace/experiment_034
OUTPUT=/workspace/experiment_034_output
```

Prepare the deterministic blind panel and five fixed-token corpora. This step
does not read a solution file:

```bash
python "$PKG/run_experiment.py" \
  --repo "$PWD" --challenges "$CHALLENGES" --work "$WORK" --prepare-only
```

Run the production-rank calibration and 25% safety gate:

```bash
python "$PKG/run_experiment.py" \
  --repo "$PWD" --challenges "$CHALLENGES" --model "$MODEL" \
  --work "$WORK" --calibrate-only
```

Run the first two panel tasks without opening labels:

```bash
python "$PKG/run_experiment.py" \
  --repo "$PWD" --challenges "$CHALLENGES" --model "$MODEL" \
  --work "$WORK" --pilot
```

Run all 32 tasks in all three primary arms, plus the two fixed controls on
eight hash-selected tasks, and evaluate:

```bash
python "$PKG/run_experiment.py" \
  --repo "$PWD" --challenges "$CHALLENGES" --solutions "$SOLUTIONS" \
  --model "$MODEL" --work "$WORK" --output "$OUTPUT" --full
```

Every global-stage checkpoint, candidate, and candidate hash is atomic.
Rerunning the same command resumes completed work and rejects changed config,
challenge, corpus, or candidate artifacts. Keep the solution file outside
`WORK`; the worker command has no solution argument.

The perception stage also trains the additive row/column/component metadata
adapter. It is frozen for interpretation and task training. The controls swap
the perception/interpretation layer bands or train interpretation against a
deterministically rotated target assignment; neither control enters the
primary three-arm hypothesis test.
