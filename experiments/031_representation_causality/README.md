# Experiment 031: Task representation causality

## Question

Experiment 030 produced task-specific verbal abstractions, but one trace per
task cannot establish that those abstractions support the answer. This
experiment asks:

1. Does independent elicitation recover the same representation?
2. Can a fresh executor solve from that representation without seeing the raw
   ARC grids?
3. Does changing one semantically critical relation change the prediction in
   the expected way?

This experiment performs no model training.

## Fixed task panel

| Task | Representation family under investigation |
| --- | --- |
| `72207abc` | sequence and index set |
| `ad38a9d0` | connected component and normalized shape |
| `7039b2d7` | divider-induced interval partition |
| `bf32578f` | boundary, reflection, and enclosed region |
| `0d87d2a6` | guide-to-component contact relation |

The panel was fixed before new elicitation. It deliberately includes the sole
Experiment 030 failure.

## Label boundary

Representation producers may read only:

- this protocol;
- the corresponding challenge under
  `outputs/assistant_blind10_20261014/challenges/`.

They must not read Experiment 030 traces or predictions, the answer key,
canvases, reports, other ARC data, or search by task ID.

Representation-only executors may read only:

- this protocol;
- one frozen representation artifact.

They must not read the original challenge, Experiment 030, the answer key, or
other representation runs. All producer and executor outputs are frozen before
grading reads any test label.

## Representation artifact

Each producer writes one JSON artifact with:

- `vocabulary`: task-specific entity and relation types;
- `extraction`: deterministic definitions for deriving them from cells;
- `instances`: symbolic encoding of every train pair and test input;
- `invariances`: properties intentionally discarded or preserved;
- `candidate_models`: at least two transformations where plausible;
- `train_validation`: concrete evidence and contradictions;
- `selected_model`;
- `remaining_ambiguities`;
- `execution_spec`: enough information for a fresh executor to construct the
  test output without access to raw grids.

The artifact must not contain a literal final test output grid. This makes the
second stage a real representation-to-output transfer rather than file copying.

## Conditions

### R0: Existing raw-grid baseline

Experiment 030 is the one-run raw-grid baseline. It is not rerun or exposed to
new solvers.

### R1/R2: Independent representation elicitation

Two fresh producers independently create an artifact for each fixed task. With
the Experiment 030 trace, this gives three independently stated
representations per task for stability analysis.

### T1/T2: Representation-only transfer

A fresh executor receives one R1 or R2 artifact and produces the test grid
without the raw challenge. Exact output is evidence that the externalized
representation is sufficient for execution. Failure localizes information
lost between cells and the symbolic encoding.

### I-overlap/I-contact: Controlled semantic intervention

For `0d87d2a6`, two artifacts will be identical except for the critical
component-selection relation:

- overlap: a component is selected only if it shares a cell with a guide;
- contact: a component is selected if it overlaps or is 4-neighbor adjacent
  to a guide.

Fresh executors receive the two artifacts separately. A prediction difference
confined to the disputed component demonstrates causal control by the stated
relation. Which intervention matches the label is checked only after freezing.

## Primary measurements

- **Vocabulary stability:** agreement on entity/relation families across
  independent runs.
- **Instance stability:** agreement on extracted task instances.
- **Model stability:** agreement on selected transformation and unresolved
  ambiguity.
- **Transfer exactness:** exact task and cell accuracy from representation
  alone.
- **Intervention fidelity:** whether the changed relation produces the
  predicted localized output change.

## Interpretation limits

This evaluates externally materialized task representations, not hidden neural
activations. A positive result supports the narrower claim that the model can
construct and use a task-specific symbolic bottleneck. It does not prove that
the same symbols are literally encoded inside the network.

