# Experiment 039 result

## Status: superseded ordered-schedule run

This run presented all identity views, then all row-2× views, then all
column-2× views, and finally all both-axes-2× views. It therefore confounded
sample content with optimization order. The corrected runner deterministically
shuffles the same sample multiset; its output is stored under `results_mixed/`.

## Result

Neither the 512-step repeated-original control nor the exact scale-lattice arm
generated either exact truth.

- Item 0 best-top-2 cell accuracy improved from 82% to 94%.
- Item 1 remained at 0% because the selected candidates had the wrong shape.
- Mean best-top-2 cell accuracy improved from 41% to 47%, a gain of 6 points.
- Truth-in-pool remained 0/2 in both arms.
- Candidate count increased from 13 to 16.
- Whole-task and item-level exact scores remained zero.

Canonical post-LoRA LOO sum NLL was already saturated: 0.000062 for control
and 0.000059 for the scale lattice. Training took 406 seconds and 726 seconds
respectively; peak allocated memory was 20.48 GB and 20.51 GB.

## Interpretation

The additional exact representations still supplied useful signal for item 0,
but increasing the schedule to 512 steps did not reproduce Experiment 038's
exact candidates. Experiment 038 used a different fixed seed and a 256-step
identity-plus-axis schedule, so this cross-experiment difference cannot be
assigned solely to the both-axes examples.

The result rejects the simple strategy of continually adding every available
view at a fixed learning rate. Canonical demonstration fit is already complete,
while candidate generation remains sensitive to the augmentation schedule and
initialization. The strongest observed configuration remains Experiment 038's
256-step exact row/column schedule.
