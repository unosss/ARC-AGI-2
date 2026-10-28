# Experiment 032: Relation counterfactual execution

## Purpose

Experiment 031 showed that changing an externalized selection representation
changed the output locally. Its intervention artifacts also contained an
enumerated `selected_components` list, so an executor could copy that list
without evaluating the abstract relation.

Experiment 032 removes all precomputed memberships and output intervals. A
fresh executor receives only:

- entity coordinates;
- one abstract relation or equivalence rule;
- a generic construction procedure.

It must compute relation membership and construct outputs on synthetic
counterfactual layouts not used in ARC training pairs. No model training is
performed.

## Fixed families and conditions

### Incidence

- `overlap`: select a component iff its cell set intersects a guide.
- `contact`: select a component iff it intersects a guide or has a cell at
  Manhattan distance one from a guide cell.

The probes include overlap, edge-contact-only, diagonal-only, and separated
components.

### Reflection

- `center_strict`: for each source run `[l,u]` in a width `W` row, fill
  `[u+1, W-2-u]`.
- `endpoint_interval`: for anchor `a`, fill `[u+1, 2a-l]`.

Single-cell runs can make these rules observationally identical. Multi-cell
runs expose the difference without providing the resulting interval.

### Shape equivalence

- `translation`: match a component to a prototype after translation only.
- `d4`: match after any translation, 90-degree rotation, or reflection.

The probe mixes translated, rotated, and reflected instances. Unmatched
components retain their source color.

## Boundary

Executors may read only this protocol and one condition artifact under
`artifacts/`. They must not read the generator, answer key, paired condition,
other execution, prior experiments, or raw ARC data.

Artifacts must not contain:

- selected component IDs;
- precomputed overlap/contact flags;
- precomputed normalized shape signatures or class IDs for query components;
- precomputed output intervals;
- literal output grids.

All six executions are frozen and hashed before grading opens
`private_answer_key.json`.

## Measurements

- exact output per condition;
- exact relation membership implied by the output;
- whether paired conditions differ only at cells entailed by the relation
  change;
- family-level success across more than one counterfactual case where
  applicable.

This tests reusable execution of an external symbolic relation. It still does
not inspect or identify hidden neural representations.

