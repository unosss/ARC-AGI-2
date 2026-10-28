# Experiment 033: Candidate coverage and selection

## Question

For blind ARC tasks, does the model fail because the intended representation
is absent from its candidate set, or because a generated correct candidate is
ranked below another train-consistent candidate?

No model training is performed.

## Fixed panel

Twenty tasks are selected deterministically from the local held-out challenge
and solution pair. The ten tasks previously solved by the frontier model in
Experiment 030 and tasks singled out for manual rule analysis are excluded
before selection. Merely appearing in a small-model evaluation manifest does
not count as frontier-model exposure, because those manifests collectively
enumerate the entire available source. The selected IDs are stored in
`task_ids.json`.

Each solver may read only its corresponding label-free file under
`challenges/` and this protocol. It must not read:

- `private_answer_key.json`;
- the source solution file;
- previous predictions, traces, reports, or canvases;
- the other independent generation run;
- other ARC data or task-ID search results.

## Independent candidate generation

Each task receives two fresh generation runs, `r1` and `r2`. Each run targets
eight semantically distinct hypotheses, or fewer when it cannot justify eight
serious alternatives.

Every candidate records:

- a five-part representation `Z = (E, R, I, O, A)`;
- a canonical semantic signature;
- a deterministic natural-language execution specification;
- full predicted outputs for every train input;
- explicit train exactness and contradictions;
- full predicted outputs for every test input;
- a within-run prior rank and rationale.

Candidates that knowingly fail a train pair may be retained only as rejected
diagnostics and are not eligible for coverage.

## Blind aggregation and ranking

After both generation runs finish, one fresh aggregator per task:

1. checks the supplied train predictions against visible train labels;
2. removes non-exact candidates;
3. deduplicates semantic equivalents across both runs;
4. preserves distinct train-equivalent generalizations;
5. ranks candidates without seeing test labels.

The frozen ranking criteria, in order, are:

1. exact reconstruction of every train output;
2. explicit input/output correspondence with no task-specific exceptions;
3. lower description and parameter complexity;
4. causal locality and preservation of irrelevant structure;
5. common geometric/object semantics;
6. independent recurrence across `r1` and `r2`.

The aggregator writes `ranked/<task_id>.json` with complete test predictions
for every unique eligible candidate. All ranked artifacts are hashed before
grading reads labels.

## Measurements

- `top1`: exact test accuracy of the frozen rank-1 candidate;
- `oracle@K`: fraction of tasks with at least one exact candidate among the
  first K unique candidates;
- full-set oracle coverage;
- candidate-count and semantic-duplication distributions;
- failure decomposition:
  - `solved_top1`;
  - `selection_error` — an exact candidate exists but was ranked below 1;
  - `generation_error` — no generated candidate predicts the test exactly;
  - `invalid_generation` — no candidate reconstructs all train outputs.

`oracle@K` measures whether increasing candidate breadth can help. The gap
between oracle and top-1 measures the selection problem.

## Scope

The tasks and model may have appeared in public pretraining. This experiment
measures the current model's observable candidate-generation behavior on a
fixed blind protocol; it is not an estimate of private leaderboard accuracy.

