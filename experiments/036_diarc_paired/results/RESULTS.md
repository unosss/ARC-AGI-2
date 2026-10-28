# Experiment 036: base vs DiARC

## Protocol

- Base: `sorokin/qwen3_4b_grids15_sft139`
- Treatment: released `yyxdnmd/DiARC-adapters/qwen3-4b/arc-agi-2`
- Both arms: identical 128-step problem LoRA, 8 geometric decode views,
  4,096 DFS nodes per view, and 8-view rescoring
- Exploratory task: previously exposed hard task `195c6913`
- Labels were opened only after both candidate files were hashed.

## Result

Neither arm solved either test item and the exact truth was absent from both
candidate pools.

| Item | Base top-2 best cell | DiARC top-2 best cell | Delta |
|---|---:|---:|---:|
| 0 | 92.33% | 95.22% | +2.89 pp |
| 1 | 91.56% | 91.56% | 0.00 pp |
| Mean | 91.94% | 93.39% | +1.44 pp |

For item 0, DiARC placed its best generated grid at rank 2, while the base
arm's best pool grid was only 93.00% accurate at rank 7. For item 1, both arms
selected a 91.56% candidate in the top two; a 96.00% candidate existed deeper
in each pool (base rank 8, DiARC rank 4).

LOO NLL after problem adaptation was 0.11137 for base and 0.10740 for DiARC.
Peak allocated memory was 18.36 GB and 20.56 GB respectively.

## Interpretation

DiARC provides a real but incomplete signal: it improved one difficult output
and candidate ranking, but did not recover the exact transformation. The second
item still exposes a selection problem because the strongest available
candidate remained outside the top two. This one-task exposed result justifies
testing a small preregistered hard panel; it is not evidence for the published
33.06% score on our pipeline.
