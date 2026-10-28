# Experiment 035 results

## Runtime gate

RTX A6000 calibration passed.

- Measured calibration wall time: 62.61 s
- Peak allocated GPU memory: 11.35 GB
- Peak with 25% margin: 14.19 GB of 47.70 GB
- Projected 16-task runtime: 2,122 s (35.4 min)
- Projected runtime with 25% margin: 2,653 s (44.2 min)

The calibration verified label-blind prompts, frozen Qwen during refiner
training, hidden-state extraction, and all three refinement stages.

## Two-task pilot

Both pilot tasks were already solved exactly by the baseline top two, so this
panel had a ceiling effect. The unguarded refiner produced wrong shapes. The
decoder was consequently changed to retain the draft shape unless the shape
head is at least 95% confident and to apply a cell edit only when categorical
confidence multiplied by the update gate reaches 0.35.

## Difficult saved candidate

Task `195c6913` used the immutable staged-Qwen candidates from Experiment 034.
The two test items began at 93.78% and 91.56% best-cell accuracy.

- G1: unchanged
- G2: unchanged
- G3 global refiner: 90.44% and 60.00%
- G3 task-adapted refiner: 93.67% and 84.33%
- Exact solutions: 0/2 for every arm
- Final top-two best-cell accuracy remained 93.78% and 91.56%, because G0 was
  retained in the merged pool and correctly outranked destructive revisions.

## Conclusion

The infrastructure works and conservative pooling prevents regression, but the
trained correction state is not yet useful. It mostly leaves strong drafts
unchanged for two stages and makes harmful edits in the third. Experiment 035
therefore does **not** justify a larger run in its current form. The next
iteration should train edit/no-edit decisions directly from real out-of-fold
solver errors instead of relying mainly on generic target corruption.
