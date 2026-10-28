# Experiment 033: Candidate coverage and selection

## Result

The blind 20-task panel produced:

- top-1 exact: **15/20 (75%)**
- full candidate-set oracle: **17/20 (85%)**
- top-1 cell accuracy: **98.73%**
- train-exact unique candidates: **162 total**, mean **8.1/task**
- invalid generation: **0/20**

Among the five top-1 failures, two contained an exact test candidate but ranked
it incorrectly, while three contained no exact candidate:

- solved at rank 1: 15
- selection error: 2
- generation error: 3

Thus the observed residual failures were not explained by one bottleneck alone.
Candidate coverage was the larger exact-match failure class (3/5), but ranking
accounted for 2/5.

## Oracle depth

| Candidate budget | Exact tasks |
|---:|---:|
| 1 | 15/20 |
| 2 | 15/20 |
| 3 | 16/20 |
| 4 | 16/20 |
| 5 | 16/20 |
| 8 | 16/20 |
| 10 | 16/20 |
| 16 / full set | 17/20 |

The curve is not a smooth improvement from adding candidates. The first
selection error is recovered at rank 3 (`28a6681f`); the second only at rank 12
(`a6f40cea`). Therefore a small top-k increase would recover one task, while the
last covered task requires retaining a very low-prior representation.

## Independent-run contribution

- r1 oracle: 16/20
- r2 oracle: 15/20
- union oracle: 17/20
- solved by both runs: 14
- r1 only: 2 (`247ef758`, `28a6681f`)
- r2 only: 1 (`78332cb0`)
- neither: 3 (`88e364bc`, `de809cff`, `da515329`)

A second independent run added one task over the stronger single run, from
16/20 to 17/20. It generated useful diversity, but repeated sampling alone did
not remove the three shared blind spots.

## Selection errors

### `28a6681f`: destination-cell ordering

The correct candidates were ranks 3 and 5. Rank 1 and the exact candidate agree
on the main representation: color-1 mass is conserved, fixed colored cells form
receptacles, and the mass settles into supported slots. They differ in how
competing slots are ordered on the larger test geometry.

Rank 1 commits all five cells to the narrow right vessel. The exact
generalization globally orders stable slots by effective depth and places three
cells in the right pocket plus two in the lower left concavity. The rank-1
prediction differs in only four of 100 cells. The ranking favored the simpler,
independently recurring “deepest receptacle, bottom-up” description over the
correct candidate's more parameterized effective-depth and narrow-pocket
tie-break.

### `a6f40cea`: periodic phase through an occluder

The correct candidate was rank 12. All high-ranked candidates inferred the
same broad structure—foreign contours hidden by a color-3 rectangular
occluder—and all reconstructed the solid-color geometry. The decisive
difference was phase representation for a period-2 edge:

- ranks 1–11 mostly advanced phase over emitted/visible contour cells or used
  conservative endpoint completion;
- rank 12 fit each rectangle side independently in global row/column
  coordinates and advanced parity through masked coordinates.

Only the latter matched the test, correcting five of 80 cells relative to rank
1. The frozen prior penalized it because independent side phases and global
masked-coordinate parity looked more parameterized and less local. This is a
genuine representation-selection error, not a missing geometry or rasterization
mistake.

## Generation errors

### `88e364bc`

All 12 unique candidates agree on the broad mechanism: decode a direction from
the 1-to-2 legend and move each color-4 marker toward its color-associated
track. Every candidate missed the same two cells across 800 test cells; the
second test output was exact.

The failing marker starts at zero-based `(3, 11)`. Every candidate moves it
diagonally through `(4, 12)` to `(8, 16)`, while the correct output stops at
`(4, 12)`. The next diagonal step would cut the corner beside the color-5 wall
at `(4, 13)`. All candidates test only whether the next diagonal destination is
occupied; none represents the swept path (a grid “supercover”) in which either
side-adjacent wall also blocks diagonal motion.

This is a narrow candidate-generation failure. Direction, marker ownership, and
the task family were extracted correctly, and the raster matches each supplied
execution rule. The absent distinction is discrete endpoint occupancy versus
continuous/supercover collision during a diagonal step.

### `de809cff`

Only two of 16 raw hypotheses reconstructed every train output. Both inferred
dense overlapping rectangles, zero-valued defects, and 3×3 repairs centered by
color 8. The closer eligible candidate still differed on 27 of 900 cells,
confined to zero-based rows 10–12 and columns 21–29.

The correct scaffold contains three independently bounded rectangles:

- color 6: rows 1–10, columns 8–29;
- color 1: rows 5–19, columns 1–20;
- color 6: rows 19–29, columns 4–25.

The eligible candidates treated the long zero region after the first
rectangle's boundary as a defect or continuation of another panel. The missing
distinction was that the two color-6 regions are separate entities with
independent finite endpoints, and that zeros beyond an endpoint are background,
not holes.

A rejected hypothesis came within four test cells but failed train exactness at
rectangle endpoints. Therefore the abstract repair operator was available, but
no candidate combined it with correct entity extraction. Mechanistically this
is an entity-extraction failure inside the broader `generation_error` class,
not an execution/materialization bug.

### `da515329`

Seven train-exact candidates represented the pattern as clockwise or
counterclockwise spirals, alternating square rings, or transformed exemplar
masks. The closest train-exact candidate differed on 84 of 676 cells. A
train-inexact rejected candidate came slightly closer at 74 cells, so no
generated execution captured the test rule.

Entity extraction was successful: the candidates consistently identified the
missing center and four cardinal arms. What they omitted was the role of arm
length `k` as a **local corner-shape parameter**. In the intended rule, each
arm continues as a clockwise strand, but for `k > 2` a sharp turn is decomposed
into an approximately `2, k−2, k` staircase/dogleg; the other arms are its
90-degree rotations. The smaller train cases collapse zero-length segments and
therefore resemble ordinary `+2` or `+4` spirals, while the `k=4` test exposes
the distinction.

The generated set offered fixed-increment spirals, rings, and whole-mask
scaling, but no hypothesis in which the observed arm length controls corner
offset and strand separation. This is a candidate-generation failure after
correct entity extraction, not a rasterization/materialization error.

## What “different representations” meant here

The alternatives were not merely different prose. A candidate was retained
only when its executable semantics could diverge on the test. The two selection
errors show two concrete axes:

1. **Entity/relation ordering** — `28a6681f` represents the destination either
   as one selected vessel or as a global set of stable slots ranked across
   multiple concavities.
2. **Coordinate system/state evolution** — `a6f40cea` represents alternating
   color phase either along reconstructed visible/emitted contour steps or in
   global coordinates continuing through an occluded interval.

Both alternatives fit all visible train pairs exactly, yet produce different
test grids. This is the operational meaning of a train-equivalent
representation choice in this experiment.

## Interpretation

The main result is a **75% top-1 / 85% oracle** split. The 10-point gap proves
that selection matters, but the remaining 15-point gap to perfect oracle shows
that ranking alone is insufficient.

Candidate breadth was also not monotonically protective. Selection-error tasks
had 11 and 12 unique candidates, while generation failures had 2, 7, and 12.
The issue is therefore semantic coverage, not candidate count by itself.

The panel is small: Wilson 95% intervals are approximately 53–89% for top-1
and 64–95% for full oracle. The result should be treated as a mechanistic
diagnostic on this fixed panel, not as an ARC leaderboard estimate.

## Consequence for the next experiment

Do not begin LoRA training from these results yet. The next discriminating test
is targeted candidate intervention on the five failures:

1. expose explicit counterfactual axes without labels (global-vs-local slot
   ordering, mask-skipping-vs-global phase, alternative ownership, layer order,
   and continuation schedule);
2. require deterministic execution of each axis;
3. test whether the exact candidate enters the set and where it ranks.

If explicit axes recover the three generation failures, the bottleneck is
proposal/search. If they do not, the failure lies in constructing or executing
the representation itself. Only then is there a defined training target.
