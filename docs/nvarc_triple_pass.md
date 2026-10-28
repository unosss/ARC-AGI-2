# Triple-Pass NVARC+ は何をしているのか

対象Notebook:

<https://www.kaggle.com/code/mayankraaz/arc-agi-2-triple-pass-nvarc-sota-35-9-lb>

## 一言で説明すると

この手法は、Qwenにグリッドを一度だけ渡して答えさせるものではない。

問題ごとに次の処理を行う。

1. train例から、その問題専用の追加学習データを作る
2. Qwen 4Bをその問題専用にLoRA学習する
3. 学習後のモデルから大量の答え候補を探索する
4. 複数の見方で候補を再採点する
5. seedを変えて同じ処理を最大3回行う
6. 集まった候補から異なる2回答を提出する

つまり、モデルの一発推論ではなく、**問題ごとの小規模学習と候補探索を組み合わせたsolver**である。

---

## 入力

一つのARCタスクには、次のデータがある。

- train input → train output の組が数件
- 正解を求めるtest input

test outputはsolverには与えられない。

例えばtrain例が3件ある場合、その3件から問題の規則を学習し、test inputへ適用する。

---

## 全体の処理フロー

```text
ARC task
  │
  ├─ train例から64系列の追加学習データを作る
  │
  ├─ Qwen 4Bをtask内LoRA学習
  │
  ├─ 制約付きDFSで多数の出力候補を生成
  │
  ├─ 候補を8種類の変換上で再採点
  │
  ├─ seedを変更して最大3 pass繰り返す
  │
  ├─ 全passの候補をまとめる
  │
  └─ 上位2候補をsubmissionへ出力
```

以下で各段階を説明する。

---

## 1. train例から、その問題専用の学習データを作る

### Task permutation

train例の一つを一時的に正解対象として隠し、残りを例としてモデルへ渡す。

train例がA、B、Cなら、次の疑似問題を作れる。

```text
BとCを例として見せ、Aのoutputを予測させる
AとCを例として見せ、Bのoutputを予測させる
AとBを例として見せ、Cのoutputを予測させる
```

これなら本物のtest outputを使わずに、教師あり学習用のinputと正解outputを作れる。

### 64系列の変換

Notebookでは、それぞれの疑似問題へ次の変換を適用する。

- 回転・反転を含むD4変換: 8通り
- 色の置換: 8通り

組み合わせると `8 × 8 = 64` 系列になる。

例えば「赤い物体を右へ移動する」という問題を回転すれば、「別の色の物体を下へ移動する」という見た目になる。しかし変換前後の対応を保存しているため、予測後に元の向きと色へ戻せる。

目的は、モデルが特定の向きや色番号を記憶するのではなく、その問題で共通する変換を学習することにある。

---

## 2. Qwen 4Bを、その問題専用に追加学習する

使用モデルは `sorokin/qwen3_4b_grids15_sft139` である。

基盤はQwen3系4Bモデルで、事前にARC形式のデータでSFTされている。この共通checkpointから、問題ごとに学習をやり直す。

追加学習にはRSLoRAを使う。

- rank: 256
- attention、MLP、embedding、出力headなどを対象にする
- 一つのtaskが終わったら、次のtaskでは共通checkpointへ戻る

つまり、120問を順番に学習して一つのモデルを育てるわけではない。

```text
共通checkpoint
  ├─ task A専用LoRA → task Aを推論
  ├─ task B専用LoRA → task Bを推論
  └─ task C専用LoRA → task Cを推論
```

各task専用モデルは、train例に含まれる規則へ重みを適応させることが期待される。

---

## 3. 出力を一度生成するのではなく、DFSで候補を探索する

通常のLLM推論は、次tokenを順番に選び、最終的な回答を一つまたは数個生成する。

このNotebookは、ARCの出力で使用できるtokenを次に限定する。

- 色を表す数字 `0`〜`9`
- 改行
- 出力終了

そして、各tokenの確率を使って深さ優先探索（DFS）する。

```text
候補prefix
  ├─ 0を追加
  ├─ 1を追加
  ├─ 2を追加
  └─ ...
```

確率が低すぎる枝は途中で捨てる。これにより、自由な文章を生成させず、ARCのグリッドとして成立する候補を多数探索する。

この処理が `Turbo DFS` と呼ばれている。

重要なのは、Qwenが最終回答を直接決定するのではなく、**探索する枝の確率を与えるモデル**として使われている点である。

---

## 4. 候補を元に戻し、別の見方でも再採点する

回転・反転・色置換したinputから得た予測は、元の座標系と色へ逆変換する。

その後、同じ候補を8種類の変換上でもう一度モデルへ評価させる。

候補の評価には主に次が使われる。

- その候補が何回生成されたか
- 生成時のNLL（モデルがどの程度自然と判断したか）
- 別のaugmentation上でも高確率になるか

同じ規則を正しく表す候補なら、向きや色を変更しても比較的安定して高く評価される、という仮定である。

---

## 5. Triple Pass

一回のLoRA学習と探索では、seedによって結果が変わる。

そこでseedやaugmentation順序を変更し、同じtaskを最大3回処理する。

### Pass A

- 主seed: 42
- 標準設定で処理する
- 全実行時間の45%付近まで使用する

### Pass B

- 主seed: 137
- Pass Aとは異なるaugmentation系列を使う
- Pass Aで未処理だったtaskを先に補う
- 全実行時間の78%付近まで使用する

### Pass C

- seed: 911など
- 残り時間で難しいtaskを再処理する
- 候補の状態に応じて処理を変える

注意点として、全taskを必ず3回実行するわけではない。時間内に処理できるtaskだけが複数passを受ける。

---

## 6. Adaptive Controller

Pass Cでは、それまでの候補を見て次の処理を選ぶ。

### reseed

候補が一つも得られなかった場合、seedを変更して再実行する。

### deepen

探索がtimeoutした場合、DFS時間と候補上限を増やす。

### verify_more

上位候補の点数差が小さく、どちらが正しいか不明な場合、追加の見方で再採点する。

### light

少数の候補が安定している場合、軽い設定で処理する。

これは「難しい問題へ多くの計算時間を使う」ための制御である。ただし、このcontroller単体がLeaderboard scoreを何点改善したかは公開成果物から確認できない。

---

## 7. 複数passの候補をまとめる

Pass A、B、Cは、それぞれ候補とscoreを保存する。

最後に全passの候補を一つにまとめ、次の二つのscoreで順位付けする。

### KG scorer

大まかには次の組み合わせである。

```text
候補が生成された回数 - augmentation上の平均NLL
```

何度も生成され、異なる見方でも高確率になる候補を上位にする。

### probability-multiplication scorer

生成時のNLLと、別augmentationでのNLLを組み合わせる。

KG scorerとは順位が変わるため、二つのscorerから異なる有力候補を取得できる。

---

## 8. 二つの回答を選ぶP4

ARCの提出では、各test inputに対して二つまで回答できる。

同じような候補を二つ出すより、異なる規則を表す候補を出した方が、どちらかが正解する可能性がある。

Notebookでは次のように選ぶ。

1. Attempt 1にはKG scorerの1位を使う
2. Attempt 2には別scorer上位の異なる候補を優先する
3. 必要なら、shapeが異なる候補を探す
4. または、25%以上のcellが異なる候補を探す
5. 見つからなければ、単に異なる候補を使う

これが `P4 structural distinctness` と呼ばれている。

ただし、コードにはfallbackがあるため、Attempt 2が常に構造的に大きく異なるとは限らない。

---

## 9. LOO-DSL

LLMとは別に、固定された小さなプログラムの組み合わせでもtrain例を説明できるか探索する。

候補primitiveには例えば次がある。

- 回転・反転
- crop
- 色変換
- tiling
- object操作
- symmetry repair
- cellwise logic

### Leave-One-Out検証

train例の一つを隠し、残りのtrain例から見つけたプログラムが、隠した例も完全一致で解けるか確認する。

```text
A、Bから規則を探索 → Cを正解できるか
A、Cから規則を探索 → Bを正解できるか
B、Cから規則を探索 → Aを正解できるか
```

検証に通ったDSL候補があれば、神経モデルの二つの回答枠に空きがある場合などに追加する。

ただし、公開されている実行成果物ではLOO認証は `0/120` だった。そのrunではDSL部分はscoreへ寄与していない。

---

## Experiment 014との違い

| 項目 | Experiment 014 | Triple-Pass NVARC+ |
|---|---|---|
| モデル | CodeT5 77M | Qwen系4B |
| task内学習 | 全parameter TTT | RSLoRA |
| 学習系列 | TTT合計4,800例 | 各taskで64系列 |
| 候補生成 | 通常のseq2seq生成 | 制約付きDFS |
| 再採点 | 投票・モデル確率 | 8-view NLL |
| 再実行 | 一条件につき1回 | 最大3 pass |
| 計算配分 | 全taskで固定 | controllerで可変 |
| model ensemble | なし | pass間で候補pooling |
| 補助solver | なし | LOO-DSL |
| GPU | L4 ×1 | L4 ×4 |

Experiment 014は「TTTすると通常生成の精度が上がるか」を確認した。

NVARC+は「TTTしたモデルを確率評価器として使い、大量の候補を探索・検証する」構成である。したがって、単純なモデルサイズ差だけではない。

---

## 公開Notebookのscoreについて

Notebook本文は次のablationを主張している。

- single-pass: 27.64%
- fixed timeout: 30.32%
- dual-pass: 31.39%
- triple-pass + controller: 33.89%
- P4 + LOO: 35.97%

一方、公開物から確認できる範囲は次の通り。

- Kaggleページに表示されているscore: 31.39
- 現在公開されている実行成果物: 4 taskだけを処理するdebug profile
- 35.97%に対応する完全なsubmission、task別結果、ablation logは含まれない

したがって、35.97%は作者の主張として扱い、現時点ではコードから再現確認できた値とは区別する必要がある。

---

## データ汚染について

上流NVARCの公開recipeでは、次が確認できる。

- ARC-AGI-2 evaluation puzzle由来の合成データをSFTへ使用
- augmented evaluationをvalidationにも使用

そのため、public evaluation 120問に対するlocal scoreは完全な未見汎化評価ではない。

ただし、Kaggleのhidden leaderboard taskそのものを学習している証拠は、Notebookのコードからは確認できない。Leaderboard scoreとlocal evaluation scoreは分けて扱う必要がある。

---

## このNotebookから採用を検討すべき部分

重要度が高い順に、次が中核と考えられる。

1. ARC用にSFTされた4Bモデル
2. taskごとのLoRA学習
3. ARC文法に制約したDFS候補探索
4. 複数augmentation上での候補再採点
5. 複数seedの候補pooling

次の機能は、効果を個別に検証する必要がある。

- Adaptive Controller
- P4 selection
- LOO-DSL
- 3 pass目

再現実験をする場合は、最初から全機能を組み込むのではなく、まず

```text
Qwen 4B
  + task-local LoRA
  + constrained DFS
  + multi-view rescoring
```

をsingle-passで再現し、その後にpass追加やselection方法を一つずつ比較するのがよい。
