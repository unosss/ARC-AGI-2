# Experiment 016: RSLoRA Leave-One-Out検証

## 目的

Experiment 015では、task-local RSLoRA後のDFS candidate poolに正解がないことが主要な失敗原因だった。
本実験では、RSLoRAがtaskの規則を学習してheld-out例へ汎化しているかを、Base Qwenとのpaired
Leave-One-Out（LOO）で確認する。

## LOOの作り方

Experiment 015の4 tasksについて、trainペアを1件ずつheld-out testへ移した。

- `0934a4d8`: 4 folds
- `36a08778`: 6 folds
- `981571dc`: 4 folds
- `aa4ec2a5`: 3 folds
- 合計: 17 folds

各foldでは、held-out inputだけをsolverへ渡し、held-out outputは候補生成完了後まで別ディレクトリに
保持した。残りのtrainペアだけがLoRA学習に使われる。

## 比較条件

共通条件:

- checkpoint: `sorokin/qwen3_4b_grids15_sft139`
- inference augmentation: 2色置換 × 8幾何変換
- `min_prob`: 0.2
- DFS window: 540秒
- decode batch: 4
- decode-only cap: 1,200秒/fold/arm
- accelerator: Kaggle NVIDIA L4 ×4
- selection: `score_kgmon` top 2

Base:

- task-local学習なし
- LoRA adapterを無効化して推論
- optimizer steps: 0

RSLoRA:

- 残りtrainペアを16色置換 × 8幾何変換
- rank 256、alpha 32
- 1 epoch、128 optimizer steps/fold

学習時間がDFS予算を減らさないよう、両armへ同じdecode-only capを独立に与えた。

## 実行順序とlabel分離

1. Base候補生成（held-out labelsなし）
2. RSLoRA候補生成（held-out labelsなし）
3. 親finalizerがlabelsを開いてcandidate coverageとrankを採点
4. 独立したNLL phaseで再度LoRA学習
5. 各foldの学習完了後にだけlabelを開く
6. labelを開いた後はoptimizer stepもDFSも行わず、Base/RSLoRAのteacher-forced NLLだけを測る

NLLはcanonical viewと16 augmented viewsの両方で測定し、pre/postで同じtoken長を使用した。

## 結果

Kaggle kernel:
[NVARC RSLoRA LOO L4 v1](https://www.kaggle.com/code/unosss/nvarc-rslora-loo-l4)

Phase runtime:

- Base candidates: 21分13秒
- RSLoRA candidates: 51分02秒
- NLL: 29分37秒
- 合計: 約1時間42分

### Candidate generation / top-2

| 指標 | Base | RSLoRA |
|---|---:|---:|
| truthがcandidate poolに存在 | 16/17 | 17/17 |
| truthがrank 1 | 15/17 | 15/17 |
| truthがtop 2 | 16/17 | 15/17 |
| unique candidates | 57 | 62 |
| candidate samples | 268 | 300 |

Paired top-2 outcome:

- RSLoRAのみ成功: 0 folds
- Baseのみ成功: 1 fold
- 両者同じ成否: 16 folds

`36a08778~loo~004`ではBase rank 1からRSLoRA rank 3へ悪化した。

`aa4ec2a5~loo~001`ではBaseは正解候補を生成できなかったが、RSLoRAは正解候補を生成した。ただし
rank 5だったため、top-2 accuracyは改善しなかった。

つまりRSLoRAはcandidate coverageを1件改善した一方、candidate rankingを2件悪化させた。
`score_kgmon`と`score_full_probmul_3`のexact結果は同じだった。

### Teacher-forced NLL

| 評価 | 改善 | 同値 | 悪化 |
|---|---:|---:|---:|
| canonical | 14 | 0 | 3 |
| 16-view mean | 10 | 0 | 7 |

RSLoRAは大部分のfoldでcanonical outputの尤度を上げた。しかしaugmentationを含めると改善は
10/17へ減り、DFS top-2は改善しなかった。

### 学習と実行の検証

- Base optimizer steps: 全foldで0
- RSLoRA optimizer steps: 全foldで128
- candidate phaseのRSLoRA loss: 0.000105〜0.009057
- NLL phaseのRSLoRA loss: 0.000088〜0.009739
- decode cap到達: 0 folds
- error / traceback: なし
- 全17 foldsでlabel lifecycleを検証
- label open後のoptimizer step / DFS call: 0

生の完全な結果は [`paired_loo_report.json`](paired_loo_report.json) に保存した。

## 解釈

### LoRAは学習している

canonical NLLが14/17 foldsで改善しているため、LoRA更新が無効だったわけではない。残りtrainペアから
held-out outputへ一般化する方向へ、teacher-forced likelihoodは多くの場合改善した。

### ただし目的関数と最終評価が一致していない

NLL改善はDFS top-2改善へ変換されなかった。むしろtop-2ではBaseが1 fold上回った。

考えられる理由:

- LoRAがcanonical orientationへ強く適応し、色・回転augmentation全体の安定性を損なう
- 正解系列の尤度は上がっても、他候補の尤度がそれ以上に上がる
- DFSの閾値・分岐順序とteacher-forced NLLは同じ指標ではない
- `score_kgmon`の票とmulti-view NLLがLoRA後の候補分布を適切に順位付けできていない

### このLOOは実際のtestよりかなり簡単

Baseだけで16/17 foldsの正解候補、15/17 foldsのrank 1を得ている。そのため、このLOOセットには
LoRAの改善余地がほとんどない。

また、checkpointはARC-AGI-2 evaluation由来の合成データを学習している可能性がある。本結果は
「LoRAが完全な未見規則へ汎化した」ことを証明しない。確認できたのは、既知に近いtrainペア上で
LoRAがNLLを改善する一方、現在のDFS + selectionでは最終top-2を改善しないということだけである。

## 結論

現在の設定では、RSLoRAは必要性を示せなかった。

- candidate coverage: わずかに改善（+1 fold）
- candidate top-2: 悪化（-1 fold）
- canonical NLL: 多くのfoldで改善
- multi-view NLL: 7/17で悪化

したがってExperiment 015のgeneration missを解決する次の優先事項は、LoRA step数やtraining lossを
増やすことではない。まず、LoRA後にaugmentation間の安定性と候補rankが悪化する理由、および
異なるseed/passによるcandidate diversityを調べる必要がある。
