# Experiment 041 expanded full-demo panel

The original two hard tasks were supplemented with ten tasks selected without
test labels from previously fixed evaluation/exclusion sets. Every selected
task has train inputs and outputs whose axes remain at most 30 after exact
single-axis 2× repetition.

## Additional ten tasks

- Control: 10/10 item top-2 exact, 10/10 truths in pool, 100% mean best-top-2
  cell accuracy.
- Exact row/column repetition: 9/10 item top-2 exact, 10/10 truths in pool,
  99.8% mean best-top-2 cell accuracy.
- The only regression was `fcc82909`: control ranked truth second, while the
  treatment ranked it third and placed a 98% candidate in the top two.

The other nine tasks had no top-2 regression. Eight were top-1 exact in both
arms; `3bd292e8` was rank 2 in both arms.

## Combined twelve-task panel

Across the original two tasks and the ten-task expansion, there are 13 test
items:

- Truth-in-pool: control 10/13; treatment 11/13.
- Item top-2 exact: 10/13 in both arms.
- Whole-task top-2 exact: control 10/12; treatment 9/12.
- Mean best-top-2 cell accuracy: control 81.96%; treatment 89.84%.

## Interpretation

Exact single-axis repetition improves candidate generation on the difficult
compatible cases: it adds one truth to the combined pool and raises mean cell
accuracy by about 7.9 points. It does not improve net submitted exact accuracy
because one easy task's truth moves from rank 2 to rank 3.

The augmentation should therefore be retained as an additional candidate
generator, not used as a replacement for the control adapter. Combining the
control and repetition candidate pools avoids the observed regression while
preserving the added hard-task candidate.
