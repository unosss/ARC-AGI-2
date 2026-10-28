# Experiment 011: Agentic Program Repair

## 目的

Experiment 009のone-shotプログラム合成を、trainで得た厳密な反例を使って
修正するagenticループへ変更する。小型モデルの最初の規則候補が不完全でも、
外部実行器による検証と再推論で完全一致へ到達できるかを測る。

## 対象

Experiment 009と同じレビュー済み10タスクを使用する。

- `03560426`
- `045e512c`
- `05269061`
- `05a7bcf2`
- `05f2a901`
- `0607ce86`
- `06df4c85`
- `09629e4f`
- `0962bcdd`
- `0a1d4ef5`

モデルはQwen3.8-27BをNF4でL4へロードする。

## 処理

各タスクについて最大3ラウンド実行する。

1. train input/outputから最大2個の`solve(grid)`を生成
2. Sandboxで静的検証
3. 全train inputへ実行
4. 次の診断を生成
   - 構文・Sandbox・runtimeエラー
   - 完全一致したtrain数
   - セルAccuracy
   - 出力サイズ
   - 欠損、余分、色違いの件数
   - 最大40個の不一致座標と期待値
5. 診断と直前のプログラムをQwenへ返して修正
6. 全train完全一致で終了

ラウンド1で最良スコアが改善しなければ、ラウンド2では既存規則の局所修正ではなく、
中心的な解釈を捨てて別のscene decompositionを作るよう指示する。

## Experiment 009からの変更

- one-shot 4サンプルから、最大3回の逐次修正へ変更
- `enable_thinking=False`
- 最大出力を4096から3072トークンへ削減
- train不一致を分類してプロンプトへ返す
- 過去候補を保持し、最良候補を基準にフィードバック
- nested helper関数をSandboxで許可
- 未使用変数名`_`を許可
- `float`と`round`を安全な組み込み関数へ追加
- 直接再帰は明示的に拒否

旧Sandboxで誤棄却された6候補が、新Sandboxではすべて静的検証を通ることを確認した。

## 成功条件

主指標は全train完全一致へ到達したタスク数とする。補助的に以下を比較する。

- 何ラウンド目で完全一致したか
- ラウンドごとの最良`train_matches + cell_accuracy`
- 有効プログラム率
- test完全一致
- one-shotだったExperiment 009の`0/10`からの改善

## 実行

Kaggle構成:

`kaggle_runs/qwen38_agentic_l4/`

出力:

- `program_results.jsonl`
- `program_summary.json`
- `program_submission.json`

## 実行結果

Kaggle L4、Qwen3.8-27B、最大3ラウンドで実行した。

- 応答: 30
- 構造化応答: 19
- 候補プログラム: 19
- Sandbox通過: 8
- train完全一致タスク: 1/10
- test完全一致: 0/1
- 生成時間: 9,936秒（約2時間46分）

`05269061`はラウンド2の別解釈でtrain 3/3へ到達した。しかし、入力中で最初に
現れる色の順序から周期列を定める偽規則だったため、testでは16/49セルしか一致
しなかった。

`05f2a901`はラウンド0のセルAccuracy 0.8528から、ラウンド2でtrain 2/3・
セルAccuracy 0.9448まで改善した。反例フィードバックは候補修正に寄与したが、
3ラウンドでは完全一致に届かなかった。

結論として、agentic修正はone-shotのtrain完全一致0/10を1/10へ改善したが、
構造化応答19/30と低く、長い差分フィードバックによる出力打ち切りも新しい
ボトルネックになった。
