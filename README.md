# ARC-AGI-2 experiments

Experimental solvers and controlled studies for ARC-AGI-2. The repository
contains the implementation, evaluation protocols, tests, and compact results;
downloaded foundation models, LoRA weights, checkpoints, and large generated
artifacts are intentionally excluded.

## Layout

- `arc_solver/`: reusable inference and representation components
- `experiments/`: numbered experiments and their protocols
- `tests/`: unit and regression tests
- `docs/`: technical notes
- `data/`: compact task data used by local evaluations
- `canvases/`: visual analyses

## Latest study

[Experiment 042](experiments/042_structured_rule_sft/) compares two matched
Qwen3-VL-4B LoRA objectives using text-serialized grids:

1. direct query-grid generation;
2. generation of an executable `transform(grid)` program.

On 64 held-out synthetic rules, direct generation remained stronger at every
completed budget. See the
[results](experiments/042_structured_rule_sft/RESULTS.md) for the exact
protocol, limitations, and measurements.

## Notes

The experiments are research prototypes rather than a single production
solver. Each numbered directory documents its own assumptions and commands.
Large model files must be downloaded separately using the model identifiers
recorded in the relevant configuration.
