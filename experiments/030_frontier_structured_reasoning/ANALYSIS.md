# Experiment 030 results

All ten trace files were completed and hashed before the answer key was opened.

## Accuracy

- Task exact: 9/10
- Item exact: 9/10
- Macro mean item cell accuracy: 99.7913%
- Micro cell accuracy: 99.2243%
- Only failure: `0d87d2a6`, with 12 wrong cells in an otherwise correct 23x25 grid

The task result is identical to the earlier one-shot blind run. The new result
adds falsifiable records of what was observed, proposed, rejected, and applied.

## Recurrent successful strategy

Across the successful tasks, the externally stated reasoning follows the same
general process:

1. Construct task-specific entities rather than using one fixed parse:
   components, frames, divider runs, half-boundaries, marker-defined axes, or
   repeated columns.
2. Establish explicit input/output correspondences.
3. Form competing rules at different abstraction levels.
4. Materialize a complete predicted output for every train input.
5. Reject candidates using concrete cell-level contradictions.
6. When multiple rules remain train-equivalent, prefer the rule with the
   clearest geometric correspondence and fewer special cases.
7. Apply the selected parameterized relation to the test grid.

Examples include rejecting area-only classification using equal-area shapes,
rejecting divider-count arithmetic using a boundary divider, and rejecting a
constant-width fill using the varying shoulders of a reflected outline.

## Failure analysis

For `0d87d2a6`, the selected rule was:

> Draw marker-defined row/column guides and recolor every red component that
> intersects a guide.

It reconstructs every train output exactly. On the test, it leaves the red
rectangle at rows 0–3 and columns 9–11 unchanged. The label recolors that
entire 4x3 rectangle, producing exactly 12 mismatches. The rectangle does not
overlap the row-4 guide, but touches it along an edge.

The missed distinction is therefore:

- proposed: recolor components whose cells overlap a guide;
- label behavior: also recolor a component reached by/touching a guide.

The train pairs did not force this distinction through the recorded
intersection examples. Solving the test requires selecting the intended
physical semantics of a propagating line, not merely the weakest
train-consistent predicate.

## Implication

The trace supports a more specific account than “the large model sees the
grid.” It dynamically constructs an abstraction suited to each task, proposes
counterfactual alternatives, and uses exact train outputs as constraints.
Its remaining error comes from choosing among train-equivalent
generalizations, not from malformed output or broad perceptual failure.

