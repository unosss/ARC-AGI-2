# Artifact schemas

## Generator file

Required top-level keys:

`task_id`, `run`, `candidates`.

Each candidate contains:

- `local_id`;
- `status`: `eligible` or `rejected`;
- `representation` with `entities`, `relations`, `invariances`, `operator`,
  and `alternatives`;
- `semantic_signature`;
- `execution_spec`;
- `predicted_train_outputs`;
- `train_exact`;
- `contradictions`;
- `test`;
- `prior_rank`;
- `rank_rationale`.

`test` is a list of predicted output grids, in challenge test order.

## Ranked file

Required top-level keys:

`task_id`, `source_runs`, `verification`, `deduplication`, `ranking`.

Each `ranking` entry contains:

- `rank`;
- `canonical_id`;
- `source_candidates`;
- `semantic_signature`;
- `representation`;
- `execution_spec`;
- `test`;
- `ranking_rationale`.

Only candidates whose supplied train outputs exactly equal all visible train
outputs may appear in `ranking`.

