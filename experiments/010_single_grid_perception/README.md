# Experiment 010: 単一グリッド知覚ベンチマーク

## 目的

`input -> output` の規則推論から切り離して、モデルが input と output の
各グリッドを単独で正しく読めているかを測る。

Experiment 009では、生成形式をプログラムにしても単純な位置関係を誤認する
ケースがあった。本実験では、失敗箇所を次の3段階に分解する。

1. **reading**: サイズ、色数、指定セル、指定行
2. **objects**: 4/8近傍の同色連結成分、bbox、正規化形状
3. **relations**: 非ゼロ領域、空行・空列、同形状ペア、対称性

正解はPythonでグリッドから決定的に生成する。自由記述の品質評価は行わない。

## 比較条件

同じ設問を次の3形式で与える。

- `text`: JSON配列のみ
- `image`: ARC標準色の画像のみ
- `both`: JSON配列と画像

対象はレビュー済み10タスクの最初のtrainペアで、inputとoutputを別々に扱う。
合計20グリッド、60応答になる。input/output間の対応や変換規則は質問しない。

## 評価項目

各応答は固定JSONスキーマで返し、以下を独立に採点する。

- `structured_rate`: JSONとして解析できた割合
- `reading`: 基本的なセル情報の平均スコア
- `objects`: 成分数と上位6成分の記述スコア
- `relations`: bbox、空行・空列、同形状、対称性の平均スコア
- `overall`: 上記3段階の単純平均
- `exact_rate`: 全フィールド完全一致率

この分解により、次のように診断できる。

- readingから低い: 入力表現または基本的な読み取りの問題
- readingは高くobjectsが低い: 物体化・セグメンテーションの問題
- objectsは高くrelationsが低い: 抽象関係の認識問題
- 単一グリッドは高いが変換で失敗: 帰納推論の問題

## 実装

- `arc_solver/perception.py`: 正解生成、画像化、JSON解析、段階別採点
- `scripts/grid_perception_baseline.py`: ローカル/汎用CLI
- `tests/test_perception.py`: 決定的な単体テスト
- `kaggle_runs/qwen38_perception_l4/`: Qwen3.8-27B・L4実行構成

ローカル実行例:

```bash
python scripts/grid_perception_baseline.py \
  --model /path/to/model \
  --output-dir outputs/grid_perception_010 \
  --load-in-4bit
```

Kaggle用スクリプトの再生成:

```bash
python kaggle_runs/qwen38_perception_l4/build_kernel.py
```

## 解釈上の注意

- 成分は「同色かつ非ゼロ」で定義している。人間の意味的な部品分割そのものではない。
- 画像には座標ラベルを描かない。プロンプトで左上原点・0-indexを指定する。
- `largest_components_4` は出力量を抑えるため上位6成分だけを採点する。
- 本実験は知覚の必要条件を測るもので、変換規則を推論できる十分条件ではない。
