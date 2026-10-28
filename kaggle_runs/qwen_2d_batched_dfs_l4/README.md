# Experiment 022 Kaggle Phase A

This package measures only DFS engine parity and throughput. It does not load
ARC solution files and its report explicitly forbids accuracy interpretation.

- `smoke-config.json`: 16 nodes/view, one timing repetition; operational only.
- `full-config.json`: preregistered 4,096 nodes/view and three timing
  repetitions; median speedup is the Phase A throughput statistic.
- Views use the exact four public-NVARC groups:
  `[0,2,8,10]`, `[1,3,9,11]`, `[4,6,12,14]`, `[5,7,13,15]`.
- A real-Qwen batch-size-one parity run is included in addition to the
  16-view independent-versus-batched comparison.

Build and submit from this directory:

```bash
python build_kernel.py --config smoke-config.json
kaggle kernels push -p .
```

After the smoke succeeds, rebuild with `full-config.json` and submit again.

## Recorded runs

- `smoke_v001`: exact parity, 1.23x speedup; non-inferential.
- `full_v002`: 1.887x median speedup, but view 15 diverged in candidates and
  logical counters. Phase A therefore failed and no paired accuracy run follows.
