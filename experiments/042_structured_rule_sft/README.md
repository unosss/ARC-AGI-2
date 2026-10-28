# Experiment 042: structured-rule SFT

## Question

On the same Qwen3-VL checkpoint, examples, LoRA parameterization, optimizer
budget, and held-out synthetic rules, does supervising an executable rule
improve exact ARC transfer over supervising only the query output grid?

This is a small scaling experiment, not a 100k/3.2M training run. Its purpose
is to test the learning objective before committing substantial compute.

## Public source data

The corpus builder consumes the two public NVARC releases:

- `sorokin/nvarc-synthetic-puzzles`: accepted input/output pairs;
- `sorokin/nvarc-artifacts-puzzles`: generated inputs, candidate output grids,
  and the Python programs that produced those grids.

It does not infer a rule label from grid differences. Synthetic pair paths are
`<split>/<puzzle_name1>/<puzzle_name1>_<puzzle_name2>.json`; these two IDs join
directly to the corresponding columns in the public `outputs/*.parquet`
artifacts. For each accepted puzzle, candidate programs are run in a
resource-limited, AST-restricted subprocess. A program becomes a teacher only
if it reproduces every accepted pair exactly.

## Leakage-safe split

The unit of splitting is the two source puzzle IDs, not an augmented view or
an individual pair. Each source ID is assigned independently by seeded hash;
a mixed puzzle is retained only when both source IDs have the same assignment.
Identical canonical programs crossing splits are removed. This prevents one
component description from appearing in train and test through different
mixtures. Validation and test additionally forbid reuse of a source ID across
episodes, so paired statistics operate on source-disjoint rules. Each episode
has three demonstrations and one query pair.

Test challenges contain demonstrations and the query input only. Query outputs
and canonical programs are written under a separate `--labels-dir`, which must
not be mounted in the training or inference environment.

Training budgets are nested prefixes of one fixed rule order:

- pilot: 64 rules;
- small: 2,000 rules;
- medium: 10,000 rules.

No rotation, color permutation, or duplicate sampling is used in the first
comparison. The independent variable is the target, not augmentation volume.

## Arms

Both arms receive the same prompt containing three complete demonstration
pairs and one query input.

1. `direct`: generate the query output grid as compact JSON.
2. `program`: generate a Python `transform(grid)` program. At inference the
   program must pass a restrictive AST validator, replay every demonstration
   exactly in a resource-limited subprocess, and then produce the query grid.

The model, revision, modality, LoRA modules/rank, row order, learning-rate
schedule, epoch count, and generated-token limit are identical. The initial
profile uses JSON text prompts and trains language projections only. This
isolates structured-output supervision from visual-path adaptation, which is a
separate later experiment. The 16,384-token limit is required by complete
three-demonstration episodes; targets are never truncated.

## Metrics and decision

Primary metric: paired query exact match on held-out synthetic rules.

Secondary metrics:

- valid-grid rate and cell accuracy;
- program parse/validation rate;
- complete demonstration replay rate;
- query exact conditional on complete replay;
- runtime, peak memory, and generated tokens.

The 64/2,000/5,000 budgets are compared on the fixed validation set. One
budget is selected before opening test labels, and the sealed test set is used
once. Proceed beyond 5,000 rules only if `program` has a positive paired
exact difference, paired exact p-value at most 0.05, and a task-bootstrap 95%
interval whose lower bound is above zero. Training loss or demonstration
replay alone cannot pass the gate.

## Commands

```bash
python experiments/042_structured_rule_sft/build_corpus.py \
  --pairs-root /data/nvarc-synthetic-puzzles \
  --program-parquet-root /data/nvarc-artifacts-puzzles/outputs \
  --output-dir /work/exp042/corpus \
  --labels-dir /sealed/exp042 \
  --config experiments/042_structured_rule_sft/config.json

python experiments/042_structured_rule_sft/train.py \
  --config experiments/042_structured_rule_sft/config.json \
  --train-file /work/exp042/corpus/train.jsonl \
  --arm direct --budget 2000 --output-dir /work/exp042/direct-2000
```

Run `train.py` again for `program`, then use `infer.py` on
`validation_challenges.jsonl`. Grade each budget with `grade.py --phase
validation`, then freeze one budget with `select_budget.py`. Run only that
budget on `test_challenges.jsonl`; `grade.py --phase test
--selection-manifest ...` is the only command that should receive
`test_labels.jsonl`.

