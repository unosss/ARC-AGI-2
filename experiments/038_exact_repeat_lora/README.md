# Experiment 038: exact row/column repetition

## Correction

Experiment 037 used a non-uniform capped warp and did not test the intended
hypothesis. Experiment 038 rejects incompatible tasks instead of modifying the
operation.

## Fixed comparison

- Exposed hard task: `20270e3b`
- Maximum axis among all train inputs and outputs: 13
- Every repeated grid therefore remains within the 30×30 model contract.
- Both arms receive the same unchanged 128 color/D4 training views first.
- Matched control: repeat those 128 unchanged views.
- Treatment: add 128 views alternating exact row-2× and column-2× repetition.
- Decode and rescore only the original task with identical budgets.

The task was already known to be unsolved in Experiment 024; one truth appeared
at rank 5 and the other was absent from the candidate pool.
