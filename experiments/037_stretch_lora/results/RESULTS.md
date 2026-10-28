# Experiment 037 result

## Status: invalid for the intended hypothesis

This run did not implement uniform 2× repetition. Because task `195c6913`
already has 20×20 and 25×25 training grids, the runner duplicated only selected
rows and columns to stay under the 30×30 parser limit. That non-uniform warp is
not the requested transformation and can change geometry-dependent rules.
Consequently, the measurements below describe a failed capped-warp variant and
provide no evidence for or against exact row/column repetition.

## Setup

Both arms used Qwen3-4B grids15, the released DiARC adapter, and a rank-256
task RSLoRA for 128 steps on the exposed hard task `195c6913`. Control used the
existing color and D4 augmentation. Stretch replaced three quarters of those
training grids with capped row, column, or two-axis repetitions. Decoding was
identical.

## Result

Neither arm produced an exact output, and truth was absent from every candidate
pool.

| Metric | Control | Stretch | Delta |
|---|---:|---:|---:|
| Mean top-1 cell accuracy | 92.11% | 92.06% | -0.06 pp |
| Mean best-top-2 cell accuracy | 92.11% | 92.33% | +0.22 pp |
| Mean pool-best cell accuracy | 94.72% | 92.33% | -2.39 pp |
| Canonical post-LoRA LOO sum NLL | 0.342 | 0.859 | +0.518 |
| Generated candidates | 14 | 3 | -11 |
| DFS nodes used | 16,675 | 8,735 | -7,940 |

Per test item:

- Item 0: control top-1 92.67%, pool-best 97.78% at rank 7; stretch produced
  one candidate at 92.56%.
- Item 1: control top-1/top-2 91.56% and pool-best 91.67%; stretch top-1 was
  91.56% and its second candidate reached 92.11%.

Stretch increased mean training sequence length from 3,458 to 4,453 tokens and
training time from 507 to 653 seconds. Peak allocated memory remained 20.56 GB.

## Conclusion

The capped non-uniform warp failed, but this result must not be used to reject
uniform 2× repetition. A valid follow-up must select tasks whose relevant grids
remain at most 30×30 after exact duplication, preserve the full original
training schedule, and add exact-repeat examples under a matched control.
