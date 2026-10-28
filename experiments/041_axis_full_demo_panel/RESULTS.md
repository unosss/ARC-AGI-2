# Experiment 041: full-demo exact-axis panel

Two previously exposed hard tasks were selected because every complete train
pair remains legal after exact 2× row or column repetition. Every arm used all
published train pairs. Candidate files for both tasks and both arms were sealed
before test solutions were opened.

Both arms used 256 mixed training rows and identical seeds, model, DiARC
adapter, search, and rescore budgets:

- Control: 128 original views plus 128 repeated original views.
- Treatment: 128 original views plus 64 exact row-2× and 64 exact column-2×
  views.

## Result

- Truth-in-pool: control 0/3 items; treatment 1/3.
- Item top-2 exact: control 0/3; treatment 1/3.
- Whole-task top-2 exact: 0/2 for both arms.
- Mean best-top-2 cell accuracy: 21.81% → 56.64% (+34.84 points).

On `20270e3b`, treatment recovered item 1 exactly at rank 2; control did not
generate truth. Item 0 remained wrong in both arms.

On `136b0064`, neither arm generated truth, but best-top-2 cell accuracy
improved from 65.41% to 69.92%.

## Interpretation

Under the actual condition where every train demonstration is available, exact
single-axis repetition produced a positive paired result on both compatible
hard tasks: one exact candidate rescue and a partial improvement on the other
task. This restores the Experiment 038 finding under a mixed sample schedule
and a second task.

The panel contains only two exposed tasks, so it supports retaining exact
single-axis repetition as a candidate augmentation; it is not a population
accuracy estimate. Both-axes repetition was not included.
