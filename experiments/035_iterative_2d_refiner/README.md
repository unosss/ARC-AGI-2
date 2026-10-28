# Experiment 035: iterative 2-D refiner

Experiment 035 tests whether a non-autoregressive, three-stage 2-D correction
head improves the top-two exact-match rate of the existing staged-Qwen DFS.

Arms:

- `baseline`: canonical top-two DFS drafts, scored with existing candidate views.
- `global_refiner`: merge G0 drafts with G1-G3 from a deterministic globally
  trained `Qwen2DRefiner`, then rescore the unique pool.
- `task_refiner`: copy the global head, adapt it with leave-one-pair-out drafts
  while Qwen stays frozen, then refine and rescore as above.

The runner package is
`kaggle_runs/qwen_iterative_refiner_runpod`. Runtime manifests, calibration
results, checkpoints, candidates, and the final report are written to the
operator-selected work/output directories rather than this source directory.
`schema.json` documents candidate payloads and `config.json` identifies the
canonical runner config.
