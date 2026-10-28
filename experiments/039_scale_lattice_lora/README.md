# Experiment 039: exact integer scale lattice

This experiment tests whether adding every legal 2× axis combination improves
over merely increasing task-LoRA steps.

- Exposed hard task: `20270e3b`
- Treatment cycles: identity, row-2×, column-2×, both-axes-2×
- Control cycles: identity repeated four times
- 128 color/D4 views per cycle; 512 steps per arm
- All mode/color/geometry tuples are deterministically shuffled together
- No clipping, padding, partial repetition, or transformed inference inputs
- Identical DiARC base, rank-256 task adapter, decode, and rescore budgets
