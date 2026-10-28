# Experiment 038 result

## Validity

This run implements uniform repetition exactly. Every row or every column in
an augmented grid is duplicated once; no partial or capped warp is used.
Task `20270e3b` is compatible because every train input and output axis is at
most 13 cells.

Both arms first received the same 128 unchanged views. The matched control
received another 128 unchanged views, while the treatment received 64 exact
row-2× and 64 exact column-2× views. Decode and rescore budgets were identical.

## Result

The exact-repeat arm recovered both truths in its candidate pool. The matched
control recovered only one.

- Item 0: matched control found the exact truth at rank 4 with 0% top-2 cell
  accuracy; exact repetition moved truth to rank 3 and top-2 cell accuracy to
  64%.
- Item 1: matched control missed truth and had 0% top-2 cell accuracy; exact
  repetition found truth at rank 2 and reached 100% top-2 cell accuracy.

Aggregate changes:

- Truth-in-pool: 1/2 → 2/2
- Item top-2 exact: 0/2 → 1/2
- Mean best-top-2 cell accuracy: 0% → 82%
- Whole-task top-2 exact: 0 → 0, because item 0 remained at rank 3
- Candidate count: 13 → 18

Canonical post-LoRA LOO sum NLL was 0.00013 for the repeated-original control
and 0.00793 for exact repetition. Both fit the canonical demonstrations closely;
the exact-repeat arm generalized better despite the slightly higher canonical
training diagnostic.

Training took 204 seconds for control and 267 seconds for exact repetition.
Both peaked at 20.48 GB allocated GPU memory.

## Conclusion

Unlike the invalid capped-warp run, exact uniform row/column repetition produced
a clear positive result on this exposed hard task: it added the previously
missing truth, promoted that item to rank 2, and moved the other truth from rank
4 to rank 3. It did not fully solve the task because the first truth remained
just outside the two-answer submission limit.

This is one exposed compatible task, so it is evidence that the augmentation is
useful, not yet an accuracy estimate. The next validation target should be a
small fixed panel of tasks whose complete train pairs remain legal after exact
2× repetition.
