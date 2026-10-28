# Artifact schemas

## Producer artifact

Required top-level keys:

`task_id`, `run`, `vocabulary`, `extraction`, `instances`, `invariances`,
`candidate_models`, `train_validation`, `selected_model`,
`remaining_ambiguities`, `execution_spec`.

`execution_spec` must contain:

- output dimensions or a deterministic way to derive them;
- all test-input entities and relations needed for execution;
- an ordered construction procedure;
- coordinate conventions;
- collision and boundary behavior.

It must not contain the literal final test output grid.

## Executor artifact

Required top-level keys:

`task_id`, `source_representation`, `representation_parse`,
`execution_steps`, `unavailable_information`, `test`.

`test` contains the predicted test output grids.

