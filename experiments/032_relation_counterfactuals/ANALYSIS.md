# Experiment 032 analysis

## Result

- Exact conditions: 6/6
- Exact counterfactual cases: 14/14
- Micro cell accuracy: 100%
- All three paired interventions changed exactly the expected cells

No execution artifact contained selected component IDs supplied by the
generator, precomputed relation flags, normalized query signatures,
precomputed output intervals, or output grids.

## What was computed

### Guide-component incidence

The executors received guide segments and component rectangles only.

- `overlap` independently constructed cell sets and selected components with a
  nonempty intersection.
- `contact` additionally computed Manhattan-distance-one edge contact while
  rejecting diagonal-only contact.

Across three layouts, the paired predictions differed on exactly 8, 8, and 10
cells. These supports exactly matched the generated semantic difference.

### Reflection reference and boundary semantics

The executors received source intervals, grid width, and an anchor, but no
target intervals.

- `center_strict` computed `[u+1, W-2-u]`.
- `endpoint_interval` computed `[u+1, 2a-l]`.

Across two layouts, the paired outputs differed on exactly 3 and 6 cells,
matching the expected support. This directly reproduces the kind of
train-equivalent geometric split observed in `bf32578f`, now without a
precomputed target.

### Shape equivalence

The executors received prototype and query coordinates only.

- `translation` subtracted minimum row and column and required exact equality.
- `d4` generated rotation/reflection transforms, normalized them, and tested
  set equality.

Across two layouts, the outputs differed on exactly 10 and 11 cells, matching
all rotated/reflected instances and leaving translation-only mismatches
unchanged in the other condition.

## What this establishes

Given an explicit task representation, the model can:

1. instantiate abstract entities from coordinate parameters;
2. calculate nontrivial spatial relations rather than copy membership lists;
3. execute different equivalence relations and reference frames;
4. localize the output effect exactly to the cells implied by the changed
   relation.

Therefore the Experiment 031 intervention result was not solely caused by an
enumerated `selected_components` shortcut. The same model can execute the
underlying reusable relations on unseen synthetic layouts.

## Combined interpretation with Experiment 031

The combined evidence separates three stages:

1. **Representation construction:** convert cells into entities and candidate
   relations.
2. **Representation selection:** choose which train-consistent relation is
   intended.
3. **Execution:** apply the chosen relation to test entities.

Experiment 032 shows no observed execution failure when the relation is
specified: 14/14 cases were exact. Experiment 031's two failures instead arose
from selecting among train-equivalent representations:

- endpoint-column versus center/closed-boundary reflection;
- overlap-only versus overlap-or-contact.

For these tested frontier-model examples, the evidence points more strongly to
representation selection than to inability to perform the spatial operation.

## What remains unproven

- The cases are synthetic and the rule is supplied in language.
- Entity extraction from raw grids was bypassed in Experiment 032.
- The panel covers only three relation families.
- This does not identify hidden neural activation structure.
- It does not show that a smaller Qwen model can construct or execute the same
  representations.

## Next research target

Before training, the next useful experiment is no longer another execution
test. It is a candidate-generation and selection audit over a larger blind ARC
panel:

- record multiple semantically distinct train-exact representations;
- generate counterfactual consequences for each;
- measure whether the intended relation is present in the candidate set;
- separate “correct candidate absent” from “correct candidate generated but
  ranked incorrectly.”

That directly answers whether model capacity is limiting representational
imagination, selection, or both.

