# Experiment 031 analysis

## Result

Two independently elicited representations were frozen for each of five
tasks. A fresh executor then received only one representation artifact, not the
raw ARC grids.

- Representation-only executions exact: 7/10
- Tasks for which both independent runs were exact: 3/5
- Micro cell accuracy: 98.4185%
- Controlled `0d87d2a6` contact intervention: exact
- Controlled overlap intervention: 12 wrong cells

This is evidence that the externalized representations often contain enough
information to execute the transformation. It is not evidence that the same
symbols literally exist in hidden neural activations.

## A concrete definition of task representation

For this analysis, a representation is the five-part structure

`Z = (E, R, I, O, A)`

- `E` — entities extracted from cells;
- `R` — relations and equivalence classes over those entities;
- `I` — information treated as invariant, preserved, or discarded;
- `O` — the operator that constructs an output from the represented input;
- `A` — unresolved alternatives that agree on the training pairs.

Two descriptions are semantically the same when these five parts agree even
if their nouns differ. Calling the same object a `region` or a `compartment`
is not a different representation. Changing translation equivalence to
rotation/reflection equivalence, changing the reflection reference frame, or
changing overlap to overlap-or-contact is a different representation.

## What was constructed in each task

### `72207abc`: sparse grid to indexed sequence

- `E`: one active row, an ordered seed word, and zero background;
- `R`: seed item `n mod k` corresponds to column
  `T(n) = n(n+1)/2`;
- `I`: absolute color meaning and gap lengths are discarded; order and grid
  bounds are preserved;
- `O`: periodically emit seed colors at triangular-number columns;
- `A`: finite lookup alternatives remain formally possible but add no
  explanatory structure.

All three independent accounts were semantically equivalent and both transfer
runs were exact.

### `7039b2d7`: pixel grid to quotient grid

- `E`: complete divider rows/columns and maximal non-divider intervals;
- `R`: each row-interval × column-interval pair is one compartment;
- `I`: interval thickness, absolute divider position, and input dimensions
  are discarded; interval order and compartment color are preserved;
- `O`: map each nonempty compartment to one output cell;
- `A`: representative-cell copying and uniform-color aggregation are
  observationally equivalent because all observed compartments are uniform.

All accounts agreed that a boundary divider creates no empty outside
compartment. Both transfer runs were exact.

### `ad38a9d0`: components to shape classes

- `E`: maximal 4-connected components;
- `R`: normalized component shape maps to a target color;
- `I`: all accounts discard translation, but they disagree on whether rotation
  and reflection also belong to the same shape class;
- `O`: preserve component cells and replace their color by the class color;
- `A`: translation-only, rotation-invariant, and full D4 equivalence are not
  separated by the observed orientations.

Both transfer runs were exact on this test, but the unseen symmetry
generalization is not stable. One run's D4 claim is stronger than its evidence.

### `bf32578f`: boundary topology versus row intervals

One run represented the grid as a supplied half-boundary, its reflected wall,
and the strict enclosed interior. The other represented each active row as a
source interval around an endpoint column.

This changes three substantive parts of `Z`:

- the reference frame: canvas center versus endpoint column;
- the preserved structure: closed-wall topology versus independent row runs;
- the operator: strict interior fill versus interval arithmetic.

The interval account used endpoint column 3 as the reflection axis and filled
through column 5. The correct width-8 reflection is about 3.5 and fills only
through column 4. This caused exactly two extra cells at `(2,5)` and `(5,5)`.
Training rows involved in the output had one-cell-wide boundary runs, so the
two operators were indistinguishable there. One transfer run was exact and one
was not.

### `0d87d2a6`: guide-component incidence graph

- `E`: border marker pairs, their guide axes, and red connected components;
- `R`: guide/component incidence;
- `I`: area, aspect ratio, order, and number of intersected guides are
  discarded; incidence and component extent are preserved;
- `O`: draw guides and recolor every selected component in full;
- `A`: select by cell overlap only, or by overlap plus 4-neighbor contact.

Both new runs independently identified this exact ambiguity, then chose the
minimal overlap predicate. Both transfer runs reproduced the same 12-cell
error as Experiment 030. Thus the representation was stable but wrong under a
train-underdetermined choice.

## Causal intervention

The two `0d87d2a6` intervention artifacts kept output dimensions, axes,
component rectangles, colors, and construction order fixed. Only the
selection relation changed:

- overlap selects six components;
- overlap-or-contact additionally selects `C0`, rows 0–3 and columns 9–11.

Fresh executors produced outputs differing at exactly those 12 cells. The
contact condition exactly matched the label; the overlap condition missed
exactly those cells.

This establishes behavioral causal sensitivity of the executor to the
externalized selection representation. It does not yet establish that:

- hidden activations use the same symbols;
- the abstract relation rather than the enumerated selected-component list was
  executed;
- the representation is necessary when the raw grid is also available;
- the behavior generalizes to newly generated counterfactual layouts.

## Main conclusion

“Constructing a representation” is not merely narrating visible shapes. In the
successful stable cases, the model changed the computational domain:

- cells became sequence positions;
- line pixels became interval partitions and a quotient grid;
- colored pixels became component shape classes;
- boundary pixels became a reflected enclosure;
- markers and objects became an incidence graph.

The important unresolved problem is no longer only whether such abstractions
can be produced. It is how to choose among abstractions that are exactly
equivalent on the few training pairs. Both observed failures came from this
selection problem: boundary semantics in `bf32578f` and contact semantics in
`0d87d2a6`.

## Required next test before training

The current intervention includes an enumerated selected-component set. A
stronger causal test should provide only entity coordinates and the abstract
relation, then use newly generated layouts for which the executor must compute
membership itself. It should cross overlap/contact, center/object-relative
reflection axes, and translation/D4 shape equivalence. This would distinguish
execution of a reusable relation from copying a precomputed selection list.

