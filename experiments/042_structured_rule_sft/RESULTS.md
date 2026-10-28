# Experiment 042 results

## Question

Does supervising an executable rule program generalize better than supervising
the output grid directly?

Both arms used Qwen3-VL-4B-Instruct with the same language-layer QLoRA
configuration and the same ordered episodes. Prompts contained three complete
demonstration pairs plus one query input, serialized as text. This was not a
vision-input experiment and did not use the NVARC 3.2M-example ARC SFT.

Programs were accepted only when they passed the AST sandbox, replayed all
three demonstrations exactly, and produced a valid query grid. The 64-rule
validation set was disjoint by source ID. Exact match was the primary metric.

## Validation results

- 64 training rules: direct 1/64 exact and 47.50% micro cell accuracy;
  program 0/64 exact, 0.00% micro cell accuracy, and 0/64 demo replay.
- 2,000 training rules: direct 3/64 exact and 49.82% micro cell accuracy;
  program 1/64 exact, 1.81% micro cell accuracy, and 1/64 demo replay.
- 5,000 training rules: direct 4/64 exact and 66.35% micro cell accuracy.
  Program evaluation was stopped; demo replay was 0/47 at that point.

At 2,000 rules, the paired program-minus-direct exact difference was -3.125
percentage points (one program win, three losses, 60 ties; two-sided paired
exact p=0.625; task-bootstrap 95% interval [-9.375, 3.125] points).

The 5,000-rule program evaluation was stopped at 47/64 on request. Of those 47
generations, 33 passed static program validation, 13 exhausted the 4,096-token
generation limit, and none replayed all demonstrations. No partial-set query
accuracy is reported.

## Runtime

On one RTX A6000 48GB:

- 2,000-rule training: 2h30m direct, 2h57m program
- 5,000-rule training: 6h16m direct, 7h23m program
- peak training memory in the 64-rule pilot: about 33GB
- 5,000-rule direct validation generation: 94m for 64 episodes

## Conclusion

This implementation does not support the hypothesis that emitting a free-form
Python rule program is a better intermediate representation. More data improved
direct generation from 1 to 4 exact matches and substantially improved its
cell accuracy. Program learning remained bottlenecked by exact demonstration
replay, while generation was slower and frequently ran to the token limit.

This does not show that all structured intermediate representations fail.
It specifically rejects this combination of unrestricted program targets,
one-epoch QLoRA, three demonstrations, and greedy generation at the tested
scales. A future study should use a constrained DSL or separately supervise
program induction and execution before spending more compute on this arm.

The sealed test split was never opened.
