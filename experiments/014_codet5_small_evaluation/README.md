# Experiment 014: CodeT5-small Evaluation

## 目的

公開済みの `mindware/arc-codet5-small`（76.9M parameters）が、学習に使われていないと作者が説明する ARC-AGI-2 evaluation 120 tasks に対して、どの程度汎化するかを固定条件で測る。

## モデル

- checkpoint: `mindware/arc-codet5-small`
- Hugging Face commit: `5eb6a06b85278d8bfebbc2986eceb24967be52fc`
- initialization: `Salesforce/codet5-small`
- license: MIT
- 公開モデルの説明によれば ARC-AGI-2 training と合成データは学習に使用され、ARC-AGI-2 evaluation は未使用
- ただし、合成データとの概念的な重複まで排除された保証はない

## 初回評価条件

- data: ARC Prize 2026 ARC-AGI-2 `evaluation_challenges` / `evaluation_solutions`
- tasks: ID 昇順の全120問
- conditions:
  - `baseline`: TTTなし、AIRVなし
  - `ttt`: TTTのみ
  - `ttt_airv`: TTT + AIRV
- TTT budget: 4,800 augmented examples
- inference budget: 2,400 examples
- self/model ensemble: disabled
- accelerator: Kaggle NVIDIA L4
- test output は推論入力から除去し、採点時だけ使用する

この一度目の固定実行を初期ベンチマークとする。以後この120問の結果を見て設計・ハイパーパラメータを変更した場合、このデータは validation として扱い、最終的な未見テストとは呼ばない。

## 結果（初回固定評価）

Kaggle kernel v4 が完走した。結果の機械可読な完全版は
[`evaluation_report.json`](evaluation_report.json) に保存した。

- `baseline`: task exact 2/120、item exact 2/172、top-1 cell accuracy 0.61197、valid grid rate 0.96221
- `ttt`: task exact 1/120、item exact 1/172、top-1 cell accuracy 0.66850、valid grid rate 0.99709
- `ttt_airv`: task exact 0/120、item exact 1/172、top-1 cell accuracy 0.61971、valid grid rate 0.98256

実行時間と peak GPU memory:

- `baseline`: 155.160秒、4,766 MiB
- `ttt`: 275.082秒、21,246 MiB
- `ttt_airv`: 756.290秒、22,384 MiB

TTTはいずれも4,800 examples、30 optimizer stepsを完了した。aggregate lossは
`ttt` が0.05010、`ttt_airv` が0.04974だった。

### 解釈

- 未調整の公開checkpointでも baseline は2 task exactを得たが、正答率は1.67%に留まる。
- この固定設定ではTTTはcell accuracyと出力形式の安定性を改善した一方、完全一致は2問から1問へ低下した。
- AIRV追加は計算時間を約2.75倍（TTT比）に増やしたが、task exactを改善しなかった。
- したがって、低いtraining lossだけでは未見タスクの規則推論や完全一致を保証しない。今後この結果を用いて変更する場合、evaluation 120問はvalidationとして扱う。

Kernel: <https://www.kaggle.com/code/unosss/arc-codet5-small-evaluation>
