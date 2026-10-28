# Experiment 035 RunPod runner

This package compares the existing staged-Qwen DFS (`baseline`) with a
globally trained three-stage 2-D refiner and the same refiner adapted per task.
The worker accepts challenge data only and rejects any `test[].output` field.
The parent opens the solution file only after all arm/task candidate files have
been hashed.

The Qwen model and structural adapters use the Experiment 034 staged loader.
Qwen is frozen while `Qwen2DRefiner` is trained. Refiner hidden states come
from `StructuralForward(..., output_hidden_states=True)` over demonstrations
and the query, never the target. Task adaptation holds out each known training
pair, generates its top-two DFS drafts using only the remaining pairs, and
only then attaches the held output for loss.

## Offline smoke

No model is loaded or downloaded:

```bash
python kaggle_runs/qwen_iterative_refiner_runpod/run_experiment.py --smoke
```

## RunPod

```bash
PKG=kaggle_runs/qwen_iterative_refiner_runpod
CHALLENGES=/data/arc-agi_training_challenges.json
SOLUTIONS=/secure/arc-agi_training_solutions.json
MODEL=/models/qwen3_4b_grids15
WORK=/workspace/experiment_035
OUTPUT=/workspace/experiment_035_output

python "$PKG/run_experiment.py" --repo "$PWD" \
  --challenges "$CHALLENGES" --work "$WORK" --prepare-only

python "$PKG/run_experiment.py" --repo "$PWD" \
  --challenges "$CHALLENGES" --model "$MODEL" --work "$WORK" --calibrate-only

python "$PKG/run_experiment.py" --repo "$PWD" \
  --challenges "$CHALLENGES" --model "$MODEL" --work "$WORK" --pilot

python "$PKG/run_experiment.py" --repo "$PWD" \
  --challenges "$CHALLENGES" --solutions "$SOLUTIONS" --model "$MODEL" \
  --work "$WORK" --output "$OUTPUT" --full
```

Preparation, staged-Qwen checkpoints, global-refiner checkpoints, candidates,
and candidate hashes are restart-safe. A changed config, challenge, corpus,
model path, worker, or global-refiner checkpoint causes a stale-artifact error
instead of silently reusing incompatible work.

Each candidate payload retains the established `ranked_candidates` list-of-
lists schema. Refined candidates additionally record G0-G3 trajectory
references, confidence/edit/gate diagnostics, timing, source votes, and the
existing multi-view Qwen rescore.
