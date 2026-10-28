# Experiment 039 corrected mixed-schedule result

## Correction

The same 512-sample multiset from the ordered run was shuffled with a fixed
task-seeded permutation. Both arms followed the same latent
mode/color/geometry order. Scale modes therefore appeared throughout training
instead of forming four separate stages.

## Result

Neither arm generated an exact truth.

- Repeated-original control mean best-top-2 cell accuracy: 41%
- Mixed scale-lattice mean best-top-2 cell accuracy: 0%
- Truth-in-pool: 0/2 for both arms
- Candidate count: 20 for control and 11 for the scale lattice
- Canonical post-LoRA LOO sum NLL: 0.000085 for control and 0.00645 for the
  scale lattice

The scale arm's candidate shapes were wrong for both items, so cell accuracy
was scored as zero.

## Interpretation

Deterministic mixing removes the explicit four-stage curriculum, but it does
not recover Experiment 038's 256-step result. Relative to the ordered
scale-lattice run, mixing reduced mean best-top-2 cell accuracy from 47% to 0%
and reduced candidate count from 16 to 11.

This demonstrates substantial optimizer-path sensitivity. Merely shuffling
single-example SGD updates does not make the four representations contribute
simultaneously: each example still changes the weights before the next one.
A stricter order-neutral test would accumulate one gradient from each of
identity, row-2×, column-2×, and both-axes-2×, average those four gradients,
and perform one optimizer update.
