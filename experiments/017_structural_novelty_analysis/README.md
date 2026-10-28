# Experiment 017: CPU-only structural novelty analysis

## Question and verdict

Prior hypothesis: the 17-fold leave-one-out (LOO) result may overestimate
real-test transfer because held training pairs are structurally less novel than
real-test items relative to their available demonstrations.

**Verdict: unresolved.** Some coarse signatures shift in the predicted
direction, but the exact orbit measures are saturated in both datasets and the
within-real-test performance associations are inconsistent. These descriptive
data neither support nor contradict structural novelty as the explanation for
the performance gap.

No model, Kaggle run, scorer, or tuning parameter was changed or rerun.

## Reproduction

```bash
PYTHONPATH=. python experiments/017_structural_novelty_analysis/analyze.py
```

The script is deterministic, CPU-only, and writes `analysis_report.json`.

## Definitions and assumptions

- **Exact pair orbit:** a target input/output pair is obtainable from one
  reference pair by one shared D4 transform and one globally consistent
  injective color mapping across both grids.
- **Exact input orbit:** the same test on inputs only.
- D4 includes identity, 90/180/270 degree rotations, and each rotation after
  reflection. Rectangular dimensions may swap.
- Every observed source color is mapped consistently; the map may be partial
  over the universal ARC palette, but it must be injective. Repeated colors
  constrain the same map.
- In exact orbit tests, **0 is an ordinary permutable color**, matching the
  upstream all-color augmentation assumption.
- Only foreground cell counts and nonzero binary component shapes treat
  **0 as background**. Same-color 4-connected components include 0.
- Dimensions are orientation-normalized as `(min(height,width),
  max(height,width))`.
- Binary component shapes are translation- and D4-normalized and compared as
  multisets. Dimension relation is the pair of reduced output/input ratios for
  the short and long orientation-normalized dimensions.
- A signature is novel only when its exact value appears in none of the
  appropriate remaining training references.

## Measured facts

The observed performance joined to each item was:

- LOO Base: top-2 16/17, truth coverage 16/17, mean best-cell accuracy
  0.995981.
- LOO RSLoRA: top-2 15/17, truth coverage 17/17, mean best-cell accuracy 1.0.
- Real-test RSLoRA (`score_kgmon`): top-2 6/33, truth coverage 8/33, mean
  best-cell accuracy 0.758317.

Novelty counts (LOO versus real test):

- exact pair orbit: 17/17 versus 33/33;
- exact input orbit: 17/17 versus 33/33;
- input dimensions: 9/17 versus 25/33;
- output dimensions: 13/17 versus 27/33;
- input color count: 1/17 versus 14/33;
- output color count: 1/17 versus 11/33;
- input foreground cell count: 13/17 versus 30/33;
- output foreground cell count: 13/17 versus 30/33;
- input same-color component size multiset: 17/17 versus 33/33;
- output same-color component size multiset: 17/17 versus 31/33;
- input nonzero binary shape multiset: 13/17 versus 32/33;
- output nonzero binary shape multiset: 13/17 versus 30/33;
- output/input dimension relation: 4/17 versus 6/33.

The largest dataset shifts were input color-count novelty (+0.365 real minus
LOO), output color-count novelty (+0.275), input-dimension novelty (+0.228),
and input binary-shape novelty (+0.205). Exact pair/input orbit and input
same-color-component novelty had no variance and therefore cannot explain the
gap in this sample.

Within real test, dimension-relation-novel items had RSLoRA top-2 0/6, truth
coverage 1/6, and mean best-cell accuracy 0.507371, versus 6/27, 7/27, and
0.814082 for non-novel items. The point-biserial correlation with cell accuracy
was -0.354743 and top-2 phi was -0.222222. These are descriptive only. Other
flags were mixed: for example, input-color-count novelty had top-2 1/14 versus
5/19, but input-dimension novelty had 5/25 versus 1/8.

Within LOO, RSLoRA truth coverage and best-cell accuracy were constant (17/17
and 1.0), so correlations for those outcomes are undefined. Input-dimension
novelty had top-2 7/9 versus 8/8 (phi -0.344265), but this is only two misses.

## Concrete examples

Each line gives a novel example and, when one exists, a non-novel comparator:

- pair orbit: `0934a4d8~loo~000` and `0934a4d8:0` are novel; no non-novel item.
- input orbit: `0934a4d8~loo~000` and `0934a4d8:0` are novel; no non-novel item.
- input dimensions: novel `36a08778~loo~000`, `135a2760:0`; non-novel
  `0934a4d8~loo~000`, `0934a4d8:0`.
- output dimensions: novel `0934a4d8~loo~000`, `0934a4d8:0`; non-novel
  `981571dc~loo~000`, `142ca369:0`.
- input color count: novel `981571dc~loo~003`, `13e47133:1`; non-novel
  `0934a4d8~loo~000`, `0934a4d8:0`.
- output color count: novel `981571dc~loo~003`, `13e47133:1`; non-novel
  `0934a4d8~loo~000`, `0934a4d8:0`.
- input foreground count: novel `36a08778~loo~000`, `135a2760:0`; non-novel
  `0934a4d8~loo~000`, `0934a4d8:0`.
- output foreground count: novel `0934a4d8~loo~000`, `0934a4d8:0`; non-novel
  `981571dc~loo~000`, `269e22fb:0`.
- input same-color components: `0934a4d8~loo~000` and `0934a4d8:0` are novel;
  no non-novel item.
- output same-color components: novel `0934a4d8~loo~000`, `0934a4d8:0`;
  real-test non-novel `269e22fb:0` (no LOO non-novel item).
- input binary shapes: novel `36a08778~loo~000`, `135a2760:0`; non-novel
  `0934a4d8~loo~000`, `0934a4d8:0`.
- output binary shapes: novel `0934a4d8~loo~000`, `0934a4d8:0`; non-novel
  `981571dc~loo~000`, `269e22fb:0`.
- dimension relation: novel `0934a4d8~loo~000`, `20270e3b:0`; non-novel
  `36a08778~loo~000`, `135a2760:0`.

The complete per-item signatures, orbit witnesses, flags, conditionals,
contingency counts, correlations, and up to five examples/counterexamples per
category are in `analysis_report.json`.

## Interpretation and limitations

Measured fact: real-test items are more often novel on several coarse
input/color/shape signatures. Measured fact: exact D4+color pair/input novelty
is 100% in both groups. Measured fact: only dimension-relation novelty shows a
notable negative real-test cell-accuracy association, while several other
directions are weak or reversed.

Hypothesis: coarse structural shift may contribute to the LOO/real gap, but
these data cannot isolate it from task difficulty, semantic rule novelty,
generation behavior, or task selection.

Limitations:

- 17 LOO folds and 33 real-test items are small denominators.
- Multiple items from one task are dependent; no item-level p-values or
  significance claims are made.
- LOO uses four source tasks, which are a subset of the 20 real-test tasks.
- Exact signatures can be too strict (saturation) and do not encode semantic
  transformation rules.
- All comparisons are observational and post hoc; conditioning does not imply
  causality.
