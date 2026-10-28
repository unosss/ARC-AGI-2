# Experiment 039: exact scale lattice

The treatment trains once on each legal exact integer scale for task
`20270e3b`: identity, row-2×, column-2×, and both-axes-2×. Each mode receives
the same 16-color × 8-D4 schedule, for 512 total steps.

The matched control also uses 512 steps but repeats the identity schedule four
times. A fixed task-seeded shuffle mixes all four latent mode slots across the
entire optimization trajectory; the matched control follows the identical
slot/color/geometry order. Decode and rescore settings are identical. All train
grids are at most 13 cells per axis, so the treatment never clips, partially
repeats, pads, or otherwise changes the requested transformations.
