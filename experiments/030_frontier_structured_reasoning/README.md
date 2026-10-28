# Experiment 030: Frontier structured reasoning replay

## Purpose

Re-solve the exact ten blind ARC training tasks used on 2026-10-14 while
persisting falsifiable intermediate reasoning artifacts. This does not claim
access to hidden model activations or private chain-of-thought.

## Fixed tasks

`42f14c03`, `72207abc`, `0d87d2a6`, `bf32578f`, `7ee1c6ea`,
`ad38a9d0`, `7039b2d7`, `b27ca6d3`, `f8a8fe49`, `fcc82909`

## Label boundary

Each solver may read only its corresponding file under
`outputs/assistant_blind10_20261014/challenges/`. It may not read the previous
prediction, answer key, score canvas, another ARC dataset, or search by task
ID. All ten trace files must exist before grading opens `answer_key.json`.

## Persisted trace

Each `traces/<task_id>.json` contains:

1. per-grid observations;
2. explicit input/output correspondences;
3. at least two candidate rule hypotheses where plausible;
4. predicted output for every train input under each serious hypothesis;
5. exact train-pair validation and concrete contradictions;
6. revisions or hypothesis rejection;
7. selected rule and why it survived;
8. test construction summary and final test grids.

The trace records externally stated, testable reasoning products. It should not
contain generic prose presented as inaccessible internal reasoning.

