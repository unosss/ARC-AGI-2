# Experiment 025: Qwen 2D speed optimization

Status: baseline, reversible-merge, and batched-rescore benchmarks complete.

## Goal

Measure and reduce the wall time of Experiment 024 without silently reducing
its task count, training steps, decode views, DFS budgets, or rescore views.
Only optimizations with explicit numerical/candidate parity checks may replace
the baseline path.

## Fixed production settings

- task-local RSLoRA: rank 256, 128 optimizer steps;
- decoding: 2 color permutations x 8 geometries = 16 views;
- DFS per view: 4096 nodes, 12288 model-forward tokens, 128 candidates;
- rescoring: 8 views per unique candidate;
- DFS physical batch size: 4.

## Profiling scope

`profile_one_task.py` measures one hard-set task using the production task-local
budgets. The RunPod instance does not yet have the expensive arm-level global
checkpoints, so this first profile loads the SFT base checkpoint and a frozen,
zero-initialized structural sidecar. Tensor shapes and compute budgets match
the task-local production phase, but generated candidates and accuracy do not.

The profile is therefore valid for locating runtime bottlenecks and estimating
capacity, not for comparing solver accuracy.

## Planned comparisons

1. Current PEFT task adapter during DFS and rescoring. Complete.
2. Reversibly merged task adapter, with candidate and NLL parity checks.
   Complete, but rejected for production because bf16 merging changed DFS
   candidates, NLL values, counters, and lower-ranked ordering.
3. Sequential 8-view NLL versus metadata-aware batched NLL, with score/ranking
   parity checks. Complete, but rejected because it was slower and changed
   bf16 NLLs and scores.
4. End-to-end timing before and after accepted optimizations. Complete: neither
   tested optimization met the parity gate, so the accepted production path is
   unchanged from the 801.002 s baseline.

Experiment 024's difficult 16-task paired accuracy run can now resume. A
task-local phase took 13.35 minutes on the profiled hard task, so 32 sequential
arm-task evaluations project to about 7.12 GPU-hours before arm-level global
training and task-to-task variation. This is a capacity estimate, not a
runtime guarantee.

## Merge benchmark result

The benchmark trained the adapter once for all 128 steps, then ran the same
adapter unmerged and reversibly merged over all 16 decode views and all 8-view
candidate rescoring calls. No production budget was changed.

- Unmerged inference: 615.1520 s.
- Merged inference: 443.7266 s (1.3863x, 27.87% faster).
- Candidate payload hashes differed.
- Both test items retained identical top-2 grids, but test item 1 exchanged one
  candidate and reordered four common lower-ranked grids.
- Two of 32 decode views differed in candidate token sets.
- For matching decode tokens, maximum absolute NLL difference was 0.33132777.
- Across matching ranked grids, maximum absolute NLL difference was 2.10879517.
- Nodes were 32,206 unmerged versus 32,140 merged; logical model-forward tokens
  were 172,787 versus 172,721.

The speedup is real, but bf16 merge is not semantically equivalent and
`merge_for_inference` must remain disabled for production. Exact payloads,
tokens, grids, hashes, timings, and counters are retained in
`merge_parity_4090.json`; the concise decision record is
`merge_parity_analysis.json`.

## Batched rescore benchmark result

Task `195c6913` was trained once for all 128 steps with an unmerged rank-256
RSLoRA. Its 16 production decode views ran once with DFS physical batch size 4
and the full per-view budgets (4096 nodes, 12288 model-forward tokens, up to
128 candidates, and 931 new tokens). DFS produced one frozen pool of 6 and 10
unique candidates for the two test items. The same adapter and pool were then
rescored first with the default sequential path and then with
`rescore_batch_size=8`; each candidate retained all 8 rescore views.

- Sequential rescoring: 117.1768 s.
- Batched rescoring: 119.1101 s.
- Batched speedup: 0.9838x, or 1.65% slower.
- All 128 per-view records retained the same schema, token counts, row IDs,
  view order, and candidate pool.
- NLLs were not numerically identical. Maximum absolute differences were
  1.73892593 for per-view `sum_nll`, 0.00186981 for per-view `mean_nll`,
  0.00001502 for `first_token_nll`, and 0.03366768 across 3,840 per-row means.
- Candidate aggregate `mean_sum_nll` and score differed by as much as
  0.47229069.
- Candidate ordering and top-2 grids happened to remain identical for both
  test items, but score parity failed.

The batched path is therefore rejected for production. The configuration
continues to default to sequential rescoring when `rescore_batch_size` is
absent or 1. Exact candidate/view payloads are in
`rescore_batching_4090.json`; `rescore_batching_analysis.json` is the concise
decision record and `rescore_batching_4090.log` is the captured run log.
