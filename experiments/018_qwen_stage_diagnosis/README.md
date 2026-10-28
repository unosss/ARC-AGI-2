# Experiment 018: Qwen stage diagnosis on `0934a4d8[0]`

## Goal

This is a causal, single-task diagnosis of the real evaluation item
`0934a4d8[0]`. It asks where the pinned Qwen + task-local RSLoRA pipeline fails,
without changing the Experiment 015 training recipe or allowing the test label
into ordinary candidate generation.

The completed v1 and v2 L4 runs emitted the canonical
`stage_diagnosis_report.json` and `stage_diagnosis_report_v2.json`. Their
downloaded artifacts are preserved separately under `output_v001/` and
`output_v002/`; v1 was not overwritten.

## V2 design and hypotheses

V1 established four facts that motivate v2: the unhinted oracle crop failed;
the same crop plus two grid-only transpose demonstrations succeeded; the full
baseline failed; and forcing eight truth rows left only `(8,0)` wrong. The
RSLoRA truth probability of that cell was `0.010953`, below `min_prob=0.2`.

V2 adds:

- `transpose_hint_full`: the two transpose demonstrations plus the original
  30×30 test input. Success would show that operation inference was sufficient
  while source localization remained latent.
- `transpose_hint_mask`: the same demonstrations plus the oracle source mask
  on the 30×30 canvas. Success here but not for `transpose_hint_full` would
  implicate localization after operation uncertainty is removed.
- `transpose_hint_crop_0..4`: the same demonstrations applied separately to
  all five existing 3×9 correspondence crops, with identical 300-second
  budgets. Crop 2 is the asserted true source. This tests whether hinted
  execution is source-specific.
- `prefix_8_rows_plus_1_cell`: exactly eight full truth rows and truth cell
  `(8,0)` are forced, then DFS generates the remaining suffix. Its generated
  suffix scores are not comparable to conditions with other prefix lengths.
- `baseline_exp015_comparable`: a clearly labeled v2 baseline using Exp015's
  two inference color permutations across all eight D4 geometries (16 views),
  batch size 4, `dfs_window=540`, and a 1,200-second overall cap. The original
  single-view 300-second baseline is retained separately.

All operation hints are inference-only grid pairs. Neither they nor any oracle
test information enter RSLoRA training. The two-worker label boundary remains
unchanged.

## Results

### V2 results

The v2 Kaggle kernel completed successfully. The canonical and nested report
copies are byte-identical, the v2 schema and lifecycle checks pass, and the
ordinary worker still had no labels while the oracle worker opened its payload
only after all 128 optimizer steps. There was no traceback, worker failure,
timeout, or OOM. Logs contain only the known non-fatal Unsloth import-order and
duplicate CUDA factory warnings, plus notebook-export deprecation warnings.

Candidate accuracy is exact coordinate-wise accuracy against the 9×3 truth (27
cells); absent or wrong-shaped cells count as incorrect. “Candidates” is the
number of distinct emitted grids, while “exact” is the report's exact-match
count.

- `transpose_hint_full`: 2 candidates, 0 exact; truth absent (rank none); best
  accuracy 6/27. Runtime limit: 300 s (`dfs_window=300`).
- `transpose_hint_mask`: 0 candidates, 0 exact; truth absent (rank none); best
  accuracy 0/27. Runtime limit: 300 s (`dfs_window=300`).
- Hinted correspondence crops, each with a 300 s limit and 300 s DFS window:
  `transpose_hint_crop_0` emitted 1 candidate, 0 exact, truth absent, best 2/27;
  crop 1 emitted 1, 0 exact, truth absent, best 4/27; true-source crop 2 emitted
  1, 1 exact, truth rank 1, best 27/27; crop 3 emitted 0, 0 exact, truth absent,
  best 0/27; and crop 4 emitted 1, 0 exact, truth absent, best 3/27.
- `prefix_8_rows_plus_1_cell`: 1 candidate, 1 exact; truth rank 1; best accuracy
  27/27. Runtime limit: 300 s (`dfs_window=300`).
- `baseline_exp015_comparable`: all 16 views decoded without timeout and yielded
  11 distinct candidates, 0 exact; truth absent (rank none); best accuracy
  19/27. Runtime limit: 1,200 s with `dfs_window=540`.

Measured training time was 417.608 s for the ordinary worker and 411.237 s for
the oracle replay. Ordinary post-training generation took 44.228 s; oracle
post-training diagnostics took 81.088 s. The report was printed at about
1,309.763 s of kernel elapsed time, and notebook export finished at about
1,316.900 s. The artifact does not retain separate wall-clock durations for
individual conditions, so the condition figures above are exact configured
limits, not claimed elapsed times.

#### V2 causal conclusions

The operation hint is not sufficient on the full 30×30 input, and combining it
with the zero-masked 30×30 canvas is also not sufficient. In contrast, with the
same hint and budget, only the asserted true 3×9 correspondence crop yields the
truth. Within this run, that is positive evidence that source correspondence
matters and that the exact cropped representation avoids a failure retained by
both the full and masked canvases. It does not prove that “localization alone”
solves the task: the successful condition supplies both an oracle crop and an
oracle operation hint, and the crop prompt remains OOD.

Forcing eight rows plus `(8,0)` closes exactly, as hypothesized, but v2 also
closes from three, four, five, and eight forced rows. The smallest exact-closing
prefix is therefore three rows, not eight rows plus one cell. The fairer
Exp015-compatible 16-view baseline still misses the truth, despite reaching
19/27, so none of the oracle successes is a fair-solver success.

#### V1/v2 reconciliation

V1 and v2 agree that the unhinted/full baseline fails and that a transpose hint
on the exact source crop succeeds. They differ materially in decoding and
likelihood: v1 emitted two transpose-probe candidates and needed more than eight
forced rows without closing; v2 emitted one probe candidate and closes from
three rows. Most sharply, RSLoRA's truth probability for `(8,0)` changed from
`0.010953` (rank 2) in v1 to `0.980392` (rank 1) in v2. The RSLoRA per-token
truth NLL improved from `0.800223` to `0.645409`, while it still remained worse
than the unchanged Base value `0.414531`.

Those differences mean the v1 final-cell bottleneck is not replay-stable and
must not be treated as the unique causal failure. V2 set
`CUBLAS_WORKSPACE_CONFIG`, unlike v1, and its ordinary/oracle training losses
also differ from v1; these are separate stochastic training replays rather than
a controlled intervention on only one decoder input. The strongest
run-internal v2 findings are the source-specific hinted-crop success, exact
closure after a three-row prefix, and continued failure of the Exp015-compatible
baseline. Generalization beyond this one item, checkpoint, and run is not
established.

### V1 results

The v1 run completed successfully and passed the report schema and lifecycle
checks. Candidate accuracy below is exact coordinate-wise cell accuracy against
the 9×3 truth (27 cells); absent or wrong-shaped cells count as incorrect.
Truth rank is the one-based position in the emitted candidate list.

### Candidate outcomes

- Execution, `transpose_probe`: 2 candidates; truth present at rank 1; best
  accuracy 27/27. With the exact crop and two synthetic transpose examples, the
  model can execute the operation. This is an oracle, OOD capability probe, not
  a fair solver result.
- Localization, `baseline`: 1 candidate; truth absent (rank none); best accuracy
  5/27.
- Localization, `oracle_crop`: 0 candidates; truth absent (rank none); best
  accuracy 0/27.
- Localization, `oracle_mask`: 2 candidates; truth absent (rank none); best
  accuracy 0/27.
  The crop and mask interventions therefore do not establish localization or
  distractor removal as sufficient. Crop failure may still reflect its OOD
  prompt shape.
- Correspondence controls: `crop_0` produced 1 candidate (best 1/27), `crop_1`
  produced 1 (0/27), true-source `crop_2` produced 0 (0/27), `crop_3` produced
  1 (1/27), and `crop_4` produced 2 (0/27). Truth was absent in every crop, so
  every truth rank is none, `only_true_source_yields_truth` is false, and
  correspondence is inconclusive.

### Base versus RSLoRA likelihood

Teacher forcing used the full 30×30 baseline prompt and the same 36 truth
tokens (27 cells plus separators/EOS). Base per-token NLL was 0.414531 and
RSLoRA was 0.800223, a +0.385692 regression. Base per-cell NLL was 0.552658 and
RSLoRA was 1.066962, a +0.514304 regression.

RSLoRA-minus-Base mean cell NLL by row was: row 0 `+3.363043`, row 1
`+0.068249`, row 2 `-0.132632`, row 3 `+0.006095`, row 4 `+0.403754`, row 5
`-0.005099`, row 6 `-0.471097`, row 7 `-0.000650`, and row 8 `+1.397071`.
Thus adaptation sharply hurt the first row and the start of the final row while
improving rows 2, 5, 6, and 7.

Using the configured 0.2 probability threshold, the exact low-confidence Base
cells were `(row 0, col 0)=7` (probability 0.006587, rank 2), `(0,1)=7`
(0.098097, rank 2), and `(4,0)=4` (0.118200, rank 2). RSLoRA's were
`(0,0)=7` (0.00000614, rank 3), `(0,1)=7` (0.005257, rank 4), `(4,0)=4`
(0.029182, rank 2), and `(8,0)=1` (0.010953, rank 2).

### Forced-prefix decoding

No tested prefix produced an exact completion, so the smallest exact-closing
prefix is null. Prefixes of 0, 1, 2, 3, 4, 5, and 8 rows produced respectively
2, 0, 2, 2, 2, 1, and 1 candidates; truth was absent throughout. Their best
cell accuracies were respectively 6/27, 0/27, 24/27, 24/27, 24/27, 26/27, and
26/27, and every truth rank was none. With five or eight truth rows forced, the
sole remaining error was
`(8,0)`: the decoder emitted 5 instead of 1. The RSLoRA truth probability at
that cell was 0.010953, below the 0.2 DFS expansion threshold, explaining why
even the eight-row prefix could not close under this search configuration.

### Runtime, limits, and diagnostics

- Ordinary and oracle-replay training each completed all 128 steps in 425.658 s
  and 408.811 s, with final losses 0.002995 and 0.002934.
- The ordinary generation-complete event followed training by 5.466 s. The
  oracle payload was opened 0.001 s after oracle training completed; diagnostics
  then took 37.248 s. The report was printed at about 1,568 s kernel elapsed,
  and notebook export finished at about 1,578 s.
- Each condition and DFS window had a 300 s budget; each worker had a 10,800 s
  timeout. Search used `min_prob=0.2`. Suffix scores cover generated tokens only
  and are not comparable across prefix lengths.
- There was no traceback, worker failure, timeout, or OOM. Non-fatal logs
  included Unsloth import-order and duplicate CUDA factory warnings. More
  importantly, PyTorch warned that deterministic mode did not guarantee
  deterministic cuBLAS operations because `CUBLAS_WORKSPACE_CONFIG` was unset.

### Interpretation and caveats

The positive hinted-transpose probe shows that execution is available when both
source and operation are supplied. Unhinted crop, mask, and all correspondence
conditions failed, so this run cannot isolate localization from operation
inference or crop-prompt OOD effects. RSLoRA worsened overall truth likelihood,
especially at the first two cells and final-row first cell. The forced-prefix
results identify the final-row first token as an additional concrete decoding
bottleneck under the 0.2 pruning threshold, but no tested prefix closed exactly.
These findings apply to one task, checkpoint, adaptation replay, and bounded
search; the cuBLAS warning also weakens claims of bitwise replay determinism.

## Pinned model and adaptation

- checkpoint: `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`
- task-local data: the task's train demonstrations only
- train augmentation: 16 color permutations × 8 D4 transforms
- RSLoRA: rank 256, alpha 32, dropout 0
- one epoch, exactly 128 optimizer steps
- learning rate `5e-5`
- LoRA seed 42; train augmentation seed 1
- bfloat16, no 4-bit loading
- Kaggle NVIDIA L4

All model turns contain only the checkpoint's grid chat format. There are no
natural-language task instructions.

## Label boundary and lifecycle

The runner creates a solver bundle from competition evaluation challenges. It
contains complete train pairs and the test input, but no test output.

1. A fresh ordinary worker trains RSLoRA for 128 steps.
2. That worker generates the baseline from the full 30×30 input and exits.
3. Only then does the parent open the separately held evaluation solution.
4. The parent verifies that input rows `0..2`, columns `14..22` form a 3×9 crop
   whose transpose exactly equals the truth. The run aborts if this fails.
5. The parent writes a tagged oracle payload outside the solver directory.
6. A fresh oracle worker repeats the same label-free training recipe.
7. Only after its 128th optimizer step does that worker open the oracle payload.
8. It performs no optimizer step after opening the payload.

The report records and validates every boundary event. Oracle information never
enters ordinary generation or either RSLoRA training run.

## Conditions and hypotheses

### Execution

`transpose_demonstration_probe` appends two synthetic grid→transpose-grid
examples to the inference context and queries the oracle source crop. These are
grid-only operation hints and are not included in LoRA training.

If this succeeds, the adapted model and decoder can execute transpose when both
the object and operation are supplied. This is a causal capability probe, not a
fair ARC solver.

### Localization

- `baseline`: original train demonstrations and full 30×30 test input.
- `oracle_source_crop`: same demonstrations, but only the exact 3×9 source.
- `oracle_source_mask`: same 30×30 shape, with every cell outside the exact
  source bounding box set to zero.

Crop success with baseline failure supports a localization/context bottleneck.
Mask versus crop helps distinguish scale/shape effects from distractor removal.
The crop query is explicitly out-of-distribution: ARC demonstrations describe
full puzzle inputs, while the query has been replaced by a parent-selected
sub-grid.

### Correspondence

Five separately decoded 3×9 crops use identical search budgets. Exactly one is
the asserted true source and four are spatially plausible controls. The report
states whether the truth appears only for the true source. This tests whether
success is source-specific rather than a generic consequence of reducing input
size.

### Model likelihood

The canonical truth is teacher-forced under the full baseline prompt for:

- Base checkpoint (adapter disabled)
- trained RSLoRA

The report retains every output token's NLL, correct-token probability, and
full-vocabulary rank, plus per-cell and per-row summaries. The digit/newline/EOS
mapping is asserted against the checkpoint tokenizer before scoring.

### Decoding

The full-input RSLoRA query is seeded with exact truth prefixes of 0, 1, 2, 3,
4, 5, and 8 complete rows, then DFS searches only the suffix. The endpoint is
the smallest prefix that yields any exact completion.

DFS scores are accumulated only over generated suffix tokens. Scores are
reported with their suffix token counts and may be compared within a prefix
condition, but neither raw nor normalized scores are compared across different
prefix lengths.

## Interpretation matrix

- Hint succeeds, crop fails: transpose is executable but the oracle crop prompt
  remains OOD or task context interferes.
- Crop succeeds, mask succeeds, baseline fails: localization/distractors are the
  leading bottleneck.
- Crop succeeds, mask fails: preserving the 30×30 canvas harms execution, so
  context length or zero background remains causal.
- Only the true correspondence crop succeeds: source correspondence matters.
- Distractors also produce truth: crop success is not source-specific evidence.
- Base/RSLoRA truth NLL is poor at particular rows/cells: model likelihood is a
  bottleneck there.
- Truth NLL is strong but ordinary decoding misses, while a short forced prefix
  closes exactly: autoregressive search/early-token error is implicated.
- Even an 8-row prefix fails: the final-row continuation remains outside the
  searched high-probability path or the search budget is insufficient.

These outcomes are diagnostic for this one item and checkpoint. They do not
estimate hidden-test accuracy or establish a generally fair solver.

## Artifacts

- reusable CPU helpers: `arc_solver/nvarc_stage_diagnosis.py`
- v2 report contract: `report_schema.json`
- Kaggle package: `kaggle_runs/nvarc_stage_diagnosis_l4/`
- generated monolithic kernel: `nvarc_stage_diagnosis_l4.py`

Build locally:

```bash
python kaggle_runs/nvarc_stage_diagnosis_l4/build_kernel.py
```

Push and run:

```bash
kaggle kernels push -p kaggle_runs/nvarc_stage_diagnosis_l4
kaggle kernels status unosss/nvarc-stage-diagnosis-l4
```
