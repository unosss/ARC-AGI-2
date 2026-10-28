# Experiment 019: Hybrid deterministic 2D solver

## Scope

This experiment is a reusable, CPU-only vertical slice for deterministic 2D
perception and explicit operations. It is deliberately specialized to
occlusion-reconstruction tasks. It is **not** a learned CNN/U-Net and it uses no
Qwen model. A learned model may later rank the structured hypotheses emitted
here, but it must not replace the train-all exact gate.

Production code lives in `arc_solver/hybrid2d/`. It contains no task ID,
test-answer, pinned bounding box, or dependency on `nvarc_stage_diagnosis.py`.

## Design

1. `Scene2D` finds maximal connected same-color components whose bounding box is
   completely solid. It scores large, two-dimensional, globally unusual
   rectangles without assigning a special meaning to any color.
2. The relation grammar enumerates destination-to-source affine D4 maps
   `x -> A*x+b`.
3. Broad self-relations must have zero visible mismatch and at least 35% visible
   support. Identity is excluded.
4. Global witnesses are combined cell-wise only when they agree. A conflict
   causes abstention.
5. If global witnesses leave cells unknown, axis-swapping maps may fill them.
   Their complete source window must be visible and their one-cell target/source
   context frame must have zero mismatch. The preferred policy requires two
   witnesses to agree on the complete prediction.
6. Every policy is applied to every train pair. A policy survives only if one of
   its candidates exactly equals every train output. All surviving programs are
   retained in diagnostics, and at most two unique test predictions are emitted.

Ranking uses global coverage, witness agreement, frame support, broad support,
and program description length. It does not use a task identifier.

## Pre-implementation numerical check

The generic selector was checked on all four `0934a4d8` demonstrations before
production code was written:

- inferred masks: `[14:23,25:29]`, `[2:6,21:26]`,
  `[19:22,12:19]`, and `[7:11,26:30]`;
- inferred color: 8 in every demonstration;
- zero-error broad witnesses:
  `reflect_rows+(31,0)`, `reflect_columns+(0,31)`, and
  `rotate_180+(31,31)`;
- reconstructed train cells: 36/36, 20/20, 21/21, and 16/16.

No train tie was resolved from test truth. Three policy variants remain
train-exact and are recorded in the report. On the test input, global
reflections recover 19/27 cells. Two independently enumerated axis-swapping
witnesses agree on one completion:

- `transpose+(0,0)`, source `[0:3,14:23]`;
- `rotate_90+(0,31)`, source `[0:3,9:18]`.

The second source window is the reflected equivalent of the first. Both have 17
zero-error frame cells.

## Leakage boundary

The core API is:

```python
solve(train_pairs, test_input)
```

It has no test-output argument. `load_challenges` removes accidental test
outputs. The CLI writes predictions before opening the optional labels file;
`evaluate_posthoc` only scores already-materialized predictions. Tests include a
mapping that raises if the solver tries to read a test `output`.

## Running

Focused task:

```bash
python -m arc_solver.hybrid2d CHALLENGES.json \
  --task-ids 0934a4d8 \
  --labels LABELS.json \
  --report experiments/019_hybrid_2d_solver/0934a4d8_report.json \
  --submission experiments/019_hybrid_2d_solver/0934a4d8_submission.json
```

`CHALLENGES.json` may instead be a directory containing `challenges.json` or
`challenge.json`. Omit `--labels` for a strictly label-free production run.

Tests:

```bash
python -m pytest -q tests/test_hybrid2d.py
```

## Results

`0934a4d8_report.json` records:

- train fit: 4/4 exact for each of three retained programs;
- unique candidates: 1;
- test result after post-hoc label loading: top-1 exact and top-2 exact;
- inferred test mask: `[14:23,0:3]`, color 8;
- global coverage: 19/27, followed by the two local witnesses above;
- label-free runtime: about 3.3 seconds on the local CPU.

`diagnostic20_report.json` is the requested NVARC diagnostic-20 run. Inference
completed before labels were opened:

- 20 tasks / 33 test items;
- train-all fit: 1/20 tasks;
- predictions: 1/33 items, with 32 abstentions;
- post-hoc exact: 1/33 items and 1/20 tasks at both top-1 and top-2;
- the exact task is `0934a4d8`;
- label-free runtime: about 11.8 seconds.

This low broad-suite coverage is expected and intentional: phase 1 implements
only solid-rectangle occlusion reconstruction. It does not claim to solve the
other transformation families in the diagnostic set. A full 120-task run was
not performed because it would measure unsupported task families rather than
the vertical slice's target capability.
