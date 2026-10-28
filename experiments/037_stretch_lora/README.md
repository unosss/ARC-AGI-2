# Experiment 037: anisotropic stretch augmentation

## Hypothesis

Repeating rows and columns in both sides of each visible training pair teaches
the task-local LoRA that the transformation should be stable under changes in
sampling density. This may reduce dependence on absolute token positions.

## Paired design

Both arms use the released DiARC adapter, rank-256 task RSLoRA, 128 training
steps, identical color/D4 schedules, identical decoding, and identical
rescoring. The only changed variable is the training-grid stretch.

The difficult, previously exposed task `195c6913` is used as an exploratory
test. Its training grids are 20×20 and 25×25 while test inputs are 30×30.
Consequently, capped repetition expands selected training axes to 30 rather
than creating illegal 40×40 or 50×50 grids.

Primary metrics are exact match and truth-in-pool. Top-1 and oracle best-top-2
cell accuracy are retained only as diagnostics.
