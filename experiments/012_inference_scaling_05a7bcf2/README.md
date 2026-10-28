# Experiment 012: Inference Scaling on 05a7bcf2

## 目的

難しい1タスクに計算資源を集中し、Qwen3.8-27Bが正答へ到達するために必要な
thinking量またはagentic turn数を測る。

対象は`05a7bcf2`。Experiment 011では、区切り線と投影という一部の構造を
認識したが、最良でもtrain 0/3・セルAccuracy 0.30だった。

## 比較A: one-shot thinking

同じtrain/test入力、同じ乱数seedを使い、以下を独立に1回ずつ実行する。

| 条件 | thinking | reasoning_effort | 最大出力 |
|---|---:|---|---:|
| off | 無効 | - | 4,096 |
| low | 有効 | low | 12,288 |
| medium | 有効 | medium | 12,288 |
| xhigh | 有効 | xhigh | 12,288 |

thinking条件では公式推奨に合わせて`temperature=1.0`、`top_p=0.95`、
`top_k=20`を使う。offでは`temperature=0.7`、`top_p=0.8`を使う。

各条件で次を記録する。

- 実際の出力トークン数
- 生成時間
- JSON/programを返せたか
- Sandbox通過数
- train完全一致数とセルAccuracy
- train完全一致時のtest完全一致

## 比較B: agentic turn scaling

thinkingを無効に固定し、Verifierの反例を返すループを最大10 turnまで実行する。
各turnは最大4,096トークン。

1. 規則と`solve(grid)`を生成
2. 全trainで実行
3. Sandboxエラー、欠損、余分、色違い、最大40件の不一致座標を返す
4. 改善しない場合は別のscene decompositionを要求
5. train完全一致で早期終了

次を記録する。

- 各turnの最良`train_matches + cell_accuracy`
- 初めてtrain完全一致したturn
- 初めてtest完全一致したturn
- 累積出力トークン数と生成時間

## 解釈

- thinkingだけ成功: 一発の候補生成に十分な内部推論量が必要
- turnだけ成功: 外部反例と修正ループが重要
- 両方成功: 正答までのtoken・時間効率を比較
- 両方失敗: モデルの抽象化語彙または提案能力が不足

## Kaggle

`kaggle_runs/qwen38_scaling_05a7bcf2_l4/`

出力:

- `scaling_results.json`: 生応答、全候補、全差分
- `scaling_summary.json`: 条件別・turn別の集計
