# Experiment 040: true outer-LOO augmentation gate

## Status: not a valid augmentation gate

Removing one demonstration can remove evidence required to identify the task's
rule. This protocol measures robustness when only three pairs are available;
it does not determine whether an augmentation helps under the official
condition where all four pairs are given. The measurements below remain useful
as a three-example diagnostic, but must not be used to accept or reject the
augmentation.

Each of the four published train pairs in `20270e3b` was removed completely in
turn. A fresh task LoRA was adapted on only the other three pairs, then decoded
the excluded canonical input. The excluded output was unavailable to every
training-row constructor and opened only for fold scoring.

## Result

- Repeated-original control: 2/4 exact in pool and 2/4 top-2 exact.
- Row-2× augmentation: 1/4 exact in pool and 1/4 top-2 exact.
- Column-2× augmentation: 1/4 exact in pool and 1/4 top-2 exact.
- Both-axes-2× augmentation: 1/4 exact in pool and 1/4 top-2 exact.

All four modes solved fold 2. Only the control also solved fold 0. None of the
augmentations rescued a fold missed by control.

Mean best-top-2 cell accuracy was 50.0% for control, 48.2% for rows, 25.0% for
columns, and 48.7% for both axes.

Teacher-forced mean sum NLL ranked rows (12.47) and both axes (13.61) ahead of
control (15.44), despite their worse exact generation. NLL is therefore not a
safe augmentation gate for this task; held-fold exact generation must remain
primary.

## Interpretation

The earlier full-train gains remain valid observations under the normal
condition where all four published train outputs are available. This outer-LOO
result only shows that repetition did not help when one quarter of the task
evidence was removed. No augmentation decision follows from that result.
