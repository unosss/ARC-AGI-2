# Trace schema

Each task trace is JSON with keys:

`task_id`, `observations`, `correspondences`, `hypotheses`,
`revision_history`, `selected_rule`, `test_construction`, `test`.

Each hypothesis records `id`, `rule`, `predicted_train_outputs`,
`train_exact`, `contradictions`, and `status`.
