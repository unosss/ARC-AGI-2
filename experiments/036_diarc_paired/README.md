# Experiment 036: DiARC paired checkpoint test

## Question

On the same `sorokin/qwen3_4b_grids15_sft139` checkpoint and task-local
adaptation/search pipeline, does the released DiARC adapter improve the
hard-exploratory ARC task?

The sole intervention is the frozen, active DiARC standard-LoRA adapter in the
`diarc` arm. Its registered configuration is rank 256 over
q/k/v/o/gate/up/down/embed/lm_head. The `base` arm starts from a separate fresh
base load. Both arms add a fresh rank-256 RSLoRA `task` adapter, train only that
adapter for 128 matched rows (16 color permutations by 8 geometries), decode
one color permutation by eight geometries, rescore eight views, and use batch-4
DFS with 4,096 nodes, 12,288 model-forward tokens, and 128 candidates per
view. The structure route is retained as `text_control` with a frozen
zero-scale sidecar.

## Lifecycle

The default hard task is `195c6913`. Each arm executes in its own process,
loads a fresh base, and explicitly unloads all adapters. Workers receive only
training pairs and test inputs. They emit pre/post task leave-one-out NLL,
ranked candidates with votes and ranks, timing, peak allocated GPU memory, and
an immutable SHA-256 marker.

Only after the complete 2-by-1 candidate hash barrier does the parent open the
solutions file. It reports top-1/top-2 exact, truth-in-pool, truth rank,
generation and selection misses, best top-2 cell accuracy, ranked votes, and
DiARC-minus-base aggregate deltas.

`--smoke` validates this lifecycle on CPU without loading a model.
`--calibrate` runs both arms on one task with two adaptation rows and reduced
decode/search budgets. `--hard-exploratory` uses every frozen production
budget. Calibration results are operational only and do not replace the hard
comparison.

## Interpretation

This is an exploratory paired result on one deliberately selected task, not an
accuracy estimate or confirmatory test. A difference can reflect interaction
between DiARC and task adaptation/search; it does not isolate pretraining
knowledge.
