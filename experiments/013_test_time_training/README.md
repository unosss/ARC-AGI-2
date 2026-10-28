# Experiment 013: Test-Time Training

## 目的

ARC-AGI-2の各タスクについて、モデルに規則の説明やPythonプログラムを直接生成させず、
問題内のtrain入出力でモデルをタスク固有に適応させ、test出力グリッドを直接生成する。

Experiment 012では、Qwen3.8-27Bに知覚・規則推論・プログラム化を一度に要求した。
約4時間で生成された7候補はすべてSandbox検証前に無効となり、thinking量やagentic
turn数を増やすだけでは実用的な探索にならなかった。

本実験では次の仮説を検証する。

1. ARC向けの事前学習によって、グリッド知覚をモデル内部に獲得できる。
2. 問題内の少数のtrain例でも、拡張とTest-Time Training（TTT）を使えば、
   タスク固有の変換へ適応できる。
3. Augment, Inference, Reverse-Augmentation and Vote（AIRV）によって、
   単発生成より安定した完全一致出力を得られる。

## 用語

- **事前学習**: 多数の公開・合成ARCタスクから一般的なグリッド知覚と変換を学ぶ。
- **TTT / TTFT**: 評価対象タスクに含まれるtrain入出力だけを教師として、
  推論直前にモデルの重みを追加更新する。
- **Test-Time Augmentation**: 入力を回転・反転などで変換して複数回推論する。
- **AIRV**: 拡張したタスクを推論し、予測を元の座標・色へ戻して投票する。

testの正解はTTTに使用しない。タスクごとに共通の初期checkpointへ戻し、
別タスクで更新した重みも引き継がない。

## 全体構成

```text
公開・合成ARCデータ
    ↓
グリッド知覚・変換の事前学習
    ↓
共通checkpoint
    ↓ タスクごとに初期化
train例の拡張
    ↓
タスク内TTT
    ↓
複数checkpoint・複数変換でtest推論
    ↓
逆変換・妥当性検査・投票
    ↓
最大2候補をsubmissionへ出力
```

## モデル

### 第一候補

短い入出力を正確に変換しやすいencoder-decoder Transformerを使用する。
最初の検証は0.5B〜1B級で行い、パイプライン成立後に3B〜8B級を検討する。

大規模な汎用チャットモデルに長いreasoningを生成させる構成は使用しない。
モデルの役割は、タスク記述から出力グリッドのtoken列を直接生成することに限定する。

### グリッド表現

各セルを通常の数字文字列として扱うのではなく、色と構造の境界を明示する。

```text
<H=3><W=4>
<ROW><C0><C2><C2><C0>
<ROW><C0><C0><C2><C0>
<ROW><C0><C0><C0><C0>
```

専用tokenを用いる理由は次の通り。

- 色番号が他の自然言語tokenと混ざらない。
- 行境界とグリッド境界を明示できる。
- 出力サイズとセル数を機械的に検証できる。
- tokenizationの違いによる座標の崩れを減らせる。

入力には全trainペアと予測対象のquery inputを含める。trainペアの順序は固定せず、
学習時に入れ替える。

## 事前学習

TTTだけで知覚能力を一から獲得することは期待しない。共通checkpointに、
少なくとも次の能力を事前に学習させる。

### 段階A: 正確なグリッド読み取り

- グリッド再構成
- 指定座標・行・列の抽出
- 色別セル数
- bounding boxと空行・空列
- 4近傍・8近傍の連結成分

### 段階B: 空間構造

- 回転・反転・転置
- 平行移動と切り出し
- 同形状判定
- 対称性、周期、反復
- 枠、領域、穴、包含関係

### 段階C: ARC変換

- 公開ARCタスク
- 手続き的に生成したARC風タスク
- 複数操作を合成したタスク

段階A・Bは自動的に正解を生成できるため、知覚と座標操作をARC規則推論から分離して
評価できる。Experiment 010の知覚指標を事前学習checkpointの診断にも使用する。

## タスク内データ生成

対象タスクを `D = {(x_i, y_i)}` とする。各trainペアを順番にqueryとし、
残りのペアをdemonstrationとして入力する。

```text
context: (x_2 → y_2), (x_3 → y_3)
query:   x_1
target:  y_1
```

queryを交代させ、trainペア数と同数の基本学習例を作る。

### 拡張

入力・出力・全demonstrationへ同じ可逆変換を適用する。

- D4幾何変換: 4回転と反転を合わせた8通り
- 色の全単射置換
- demonstration順序の置換
- 必要に応じて余白の追加・削除

幾何変換は必須とする。色置換は背景色を固定する条件と、全色を置換する条件を
分けてablationする。余白変換は規則を変えてしまう場合があるため、初期実験では無効にする。

## TTT

タスクごとに共通checkpointから開始し、拡張したタスク内データで追加学習する。

初期候補:

- optimizer: AdamW
- 更新対象: まずLoRA、必要なら全parameter更新と比較
- 精度: bf16
- checkpoint: 複数stepで保存
- early stopping: train lossだけでは決めない

train出力の暗記でもlossは下がるため、学習lossはモデル選択の主指標にしない。

### Leave-One-Out検証

厳密な検証では、1つのtrainペアをgradient更新から完全に除外し、残りだけで適応して
除外ペアを予測する。

```text
適応: 例2・例3 / 評価: 例1
適応: 例1・例3 / 評価: 例2
適応: 例1・例2 / 評価: 例3
```

各foldの完全一致率と拡張間の安定性から、学習率・step数・checkpointを選ぶ。
最終的なtest推論では、選択した設定を使って全trainペアで改めて適応する。

Leave-One-Outは計算量が大きいため、最初は少数タスクで設定を決める。
全タスク実行時は、事前に選んだ共通設定と軽量な安定性評価を使用する。

## AIRV推論

1. trainペアとtest inputを同じ可逆変換で拡張する。
2. TTT済みモデルで出力グリッドを生成する。
3. 生成結果を逆回転・逆反転・逆色置換する。
4. サイズ、セル数、色範囲を検査する。
5. 元の座標系で完全一致する候補をまとめる。
6. 得票、モデル確率、拡張間安定性、Leave-One-Out性能で順位付けする。
7. 上位2候補を提出形式へ変換する。

セル単位の多数決は、存在しない中間的な図形を作る可能性があるため原則使用しない。
出力グリッド全体の完全一致票を基本とする。

## 評価指標

### 知覚

- structured / valid grid rate
- Experiment 010のreading、objects、relations
- グリッド再構成の完全一致率

### タスク内適応

- Leave-One-Out完全一致率
- 拡張間の予測一致率
- TTT前後の改善タスク数

### 最終性能

- test grid完全一致率
- 1候補目・2候補目のいずれかが正解した率
- 1タスクあたりのTTT時間と推論時間
- peak GPU memory

セルaccuracyは診断用に記録するが、モデル選択の最終基準は完全一致とする。

## 実験段階

### Phase 1: パイプライン検証

- 既知の正解を持つ20タスク
- 0.5B〜1B級モデル
- D4拡張のみ
- TTTなし / LoRA TTTを比較
- valid grid rateとLeave-One-Outを確認

### Phase 2: Ablation

- 色置換の有無
- TTT step数
- LoRA / 全parameter更新
- AIRVの有無
- 専用grid token / 通常の数字表現
- 知覚事前学習の有無

### Phase 3: スケール

- 3B〜8B級モデル
- 合成事前学習データの拡大
- ARC-AGI-2 local evaluation全体
- Kaggle L4での総実行時間を測定

## 成功条件

Phase 1では次を最低条件とする。

1. valid grid rateが99%以上。
2. TTTによってLeave-One-Out完全一致タスク数が非TTT条件より増える。
3. AIRVによって単発推論より完全一致タスク数が増える。
4. 1タスクあたりの時間から、Kaggle制限内に全タスクを処理できる見込みが立つ。

知覚指標が低いままTTTだけが改善した場合は、規則の暗記やデータ漏洩を疑う。

## 主なリスク

- 合成事前学習の語彙にない抽象化は学習できない。
- 少数例への過学習で、train再現とtest汎化が乖離する。
- 色置換や余白追加が元のタスクの意味を変える場合がある。
- タスクごとのgradient更新がKaggleの時間制限を超える。
- 自己回帰生成では、1セルの誤りで完全一致を失う。
- ARC-AGI-1で有効だった設定が、より複合的なARC-AGI-2でも有効とは限らない。

## 実装前に決める事項

1. 初期checkpointとparameter数
2. 利用可能な公開・合成事前学習データ
3. 専用grid tokenを追加学習できるか
4. LoRA対象層と1タスクあたりのstep上限
5. Phase 1で使う20タスク
6. Kaggle提出時に許可される外部datasetとmodelの登録方法

## Phase 1 実行結果

### 実装

- upstream: `jcole75/arc_2025_mindsai`
- commit: `d4ec950b11e31a9114efe82c86c389808224edc9`
- model: `mindware/arc-codet5-660m`
- Kaggle: [ARC MindsAI TTT Phase 1](https://www.kaggle.com/code/unosss/arc-mindsai-ttt-phase-1)
- 固定seed: `13013`
- 対象: ARC-AGI-2 training splitから固定した20タスク、22 test grids
- TTT: 800拡張例、13 optimizer steps、全parameter更新
- AIRV: 400推論例

Transformers 5との互換性のため、任意のFlax import、公開checkpointのBPE
tokenizer読込、TrainingArgumentsの差分を吸収した。中間checkpointは保存せず、
TTT完了時の最終モデルだけを保存する。

### Smoke test

3タスク、24拡張例で1 optimizer stepを実行した。

- 完全一致: 2/3
- valid grid rate: 100%
- top-1 cell accuracy: 99.48%
- 実行時間: 360.1秒
- peak GPU memory: 16,128 MiB

詳細は `smoke_report.json` に保存した。

### 20タスク比較

**非TTT**

- task exact: 18/20
- item exact: 19/22
- valid grid rate: 100%
- top-1 cell accuracy: 95.52%
- 実行時間: 107.8秒
- peak GPU memory: 5,996 MiB

**TTT**

- task exact: 17/20
- item exact: 18/22
- valid grid rate: 100%
- top-1 cell accuracy: 94.98%
- 実行時間: 199.9秒
- peak GPU memory: 17,438 MiB
- 13 optimizer steps、最終loss 0.006329

**TTT + AIRV**

- task exact: 18/20
- item exact: 19/22
- valid grid rate: 100%
- top-1 cell accuracy: 93.46%
- 実行時間: 332.2秒
- peak GPU memory: 17,160 MiB
- 13 optimizer steps、最終loss 0.007522

詳細は `phase1_report.json` に保存した。

### 判定

実行パイプラインとvalid grid rate 99%以上という条件は達成した。一方、この20タスクでは
TTT単独は非TTTより1タスク悪化し、TTT + AIRVも非TTTと同数だった。したがって、
「TTTによって完全一致タスク数が増える」というPhase 1の成功条件は達成していない。

ただし、この結果を未知タスクへの汎化性能とは解釈できない。公開CodeT5 checkpointは
ARC系公開・合成データで長期事前学習されており、今回使用したARC-AGI-2 training splitは
学習データまたはその派生例に含まれる可能性が高い。非TTTの18/20という高い値も、
この汚染・記憶の可能性と整合する。

今回確認できたのは次の範囲である。

1. 660MモデルのTTT + AIRVをL4上で実行できる。
2. 20タスクの3条件比較は約10分で完了し、試行錯誤可能な速度である。
3. 既知に近いタスクでは、短いTTTが性能を改善せず破壊する場合がある。
4. 次の評価には、checkpointの事前学習集合と重ならないclean holdoutが必要である。

