# Experiment 015: NVARC Single-pass再現

## 目的

公開Notebook
[ARC-AGI-2: Triple-Pass NVARC+ SOTA](https://www.kaggle.com/code/mayankraaz/arc-agi-2-triple-pass-nvarc-sota-35-9-lb)
から、Triple-pass以前のsingle-pass中核を再現する。

今回の目的はLeaderboard scoreの再現ではなく、次の処理が固定条件で一巡することの確認である。

1. ARC向けSFT済みQwen 4Bを読み込む
2. taskごとにRSLoRAを学習する
3. 制約付きTurbo DFSでグリッド候補を生成する
4. 候補を複数viewのNLLで再採点する
5. `score_kgmon`上位2候補を提出形式へ変換する

## 再現範囲

含めるもの:

- task-local RSLoRA
- 幾何変換8通り
- train color augmentation 16通り（合計128系列）
- inference color augmentation 2通り（合計16 decode views）
- constrained Turbo DFS
- multi-view NLL rescoring
- `score_kgmon` top-2 selection

含めないもの:

- Pass B / Pass C
- Adaptive Controller
- P4 structural-distinctness selection
- LOO-DSL
- 複数passのcandidate pooling

上流コードと出典は
[`third_party/nvarc_single_pass/UPSTREAM.md`](../../third_party/nvarc_single_pass/UPSTREAM.md)
に固定した。Notebook cells 5–8の本体は変更せず、Kaggle向けpackaging時だけ、このリポジトリの
`arc_solver` packageとの名前衝突を避けるためmodule名とimportを変更する。

## 固定条件

- model: `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`
- model size: 約4B parameters
- accelerator: Kaggle NVIDIA L4 ×4
- tasks:
  - `0934a4d8`
  - `36a08778`
  - `981571dc`
  - `aa4ec2a5`
- workers: 4（各GPUが1 taskを担当）
- LoRA: RSLoRA、rank 256、alpha 32
- epochs: 1
- optimizer steps: 128/task
- train augmentation: 16 color permutations × 8 geometric transforms
- inference augmentation: 2 color permutations × 8 geometric transforms
- task cap: 1,200秒
- total deadline: 3,300秒
- seed: LoRA 42、train augmentation 1、inference augmentation 2
- task order: sorted
- candidate selection: `score_kgmon` top 2

test outputはsolver入力から除去し、全worker終了後に親finalizerだけがsolution fileを開いて採点した。

## 実行結果

Kaggle kernel:
[NVARC Single Pass L4 v5](https://www.kaggle.com/code/unosss/nvarc-single-pass-l4/versions/5)

- status: COMPLETE
- wall time: 1,523.381秒（約25分23秒）
- peak GPU memory: 18,204 MiB
- task exact: 2/4
- item exact: 3/5
- valid-grid rate: 100%
- top-1 cell accuracy: 93.3852%
- candidate shards: 67
- candidate samples: 73
- unique candidates: 23
- identity fallback: 0
- zero-valued attempt-2 fallback: 1

### task別

| task | task exact | 時間 | TTT loss | unique / samples | 備考 |
|---|---:|---:|---:|---:|---|
| `0934a4d8` | 0 | 528.9秒 | 0.004686 | 9 / 14 | top-1 cell accuracy 70.37% |
| `36a08778` | 0 | 1,354.4秒 | 0.005074 | 11 / 31 | 2 test items中1件exact |
| `981571dc` | 1 | 1,215.6秒 | 0.000178 | 1 / 12 | task cap到達、attempt-2 fallback |
| `aa4ec2a5` | 1 | 660.6秒 | 0.005895 | 2 / 16 | exact |

生の結果は [`smoke_report.json`](smoke_report.json) に保存する。

## 確認できたこと

- SFT済みQwen 4Bを4 GPUで独立taskへロードできた
- 全taskでRSLoRAが128 optimizer stepsを完了した
- constrained DFSが全taskで候補を生成した
- 73候補すべてが有効なARC gridで、multi-view rescoringを通過した
- 5 test itemsすべてについてsubmission形式の2 attemptsを作成した
- labelsはsolver subprocessへ渡されていない

つまり、NVARC single-passの実行経路は再現できた。

## 解釈と限界

`2/4`を汎化性能や期待Leaderboard scoreとして扱うことはできない。

- 4問は上流Notebook既定のdebug tasksで、無作為標本ではない
- checkpointの上流SFTにはARC-AGI-2 evaluation由来の合成データが含まれる
- sample数が小さい
- claimed `single-pass: 27.64%`に対応する120問またはhidden test実行ではない

また、TTT lossが小さいこと自体はtest outputの完全一致を保証しない。`0934a4d8`ではlossが低くても
不正解であり、候補生成・候補選択を含む全経路が重要である。

このsmoke結果を確認するまで、計画どおり120問評価やLeaderboard提出、設定調整は行っていない。

## 追加16問

最初の4問と重複しないevaluation ID昇順の次の16問を、同じscience設定で追加実行した。
変更したのは対象taskと全体deadline（10,800秒）のみである。

Kaggle kernel:
[NVARC Single Pass L4 v6](https://www.kaggle.com/code/unosss/nvarc-single-pass-l4/versions/6)

- status: COMPLETE
- wall time: 3,695.584秒（約61分36秒）
- peak GPU memory: 15,470 MiB
- task exact: 2/16
- item exact: 3/28
- valid-grid rate: 100%
- top-1 cell accuracy: 72.3882%
- candidate shards: 270
- candidate samples: 322
- unique candidates: 180
- candidate-backed items: 27/28
- identity fallback: 1
- zero-valued attempt-2 fallback: 2
- 全taskで128 optimizer stepsを完了
- task cap到達: `195c6913`、`221dfab4`
- 候補ゼロ: `271d71e2`

task exactだったのは `135a2760` と `1818057f`。`1ae2feb7` は3 test items中1件だけexactだった。

生の結果は [`additional16_report.json`](additional16_report.json) に保存した。

### 20問累計

初回4問と追加16問を合算すると次の通り。

- task exact: 4/20（20.0%）
- item exact: 6/33（18.2%）
- top-1 cell accuracy: 75.57%（item数による加重平均）
- valid-grid rate: 100%
- candidate samples: 395
- unique candidates: 203

ただし、これはID昇順の20問であり無作為標本ではない。またcheckpointのevaluation由来SFT汚染があるため、
未見汎化性能やLeaderboard期待値とは解釈しない。

## Candidate diagnostics（20問再実行）

v5/v6ではcandidate shardをdownload後に保持していなかったため、同じ20 taskを同じscience設定で再実行した。
変更したのは、全体deadlineを14,400秒（4時間）にしたことと、全worker終了後にparent finalizerがlabelsを
開いて行うpost-hoc診断だけである。task capは1,200秒のまま、提出候補も従来どおり
`score_kgmon` top-2であり、oracle情報はsolverにも提出選択にも渡していない。

Kaggle kernel:
[NVARC Single Pass L4 v7](https://www.kaggle.com/code/unosss/nvarc-single-pass-l4/versions/7)

- status: COMPLETE
- wall time: 4,477.296秒（約74分37秒）
- peak GPU memory: 18,224 MiB
- task exact: 4/20
- item exact: 6/33
- top-1 cell accuracy: 73.4825%
- valid-grid rate: 100%
- candidate samples / unique: 394 / 203
- candidate-backed items: 31/33
- identity fallback: 2
- zero-valued attempt-2 fallback: 6

### Oracle集計

- truthがcandidate pool内に存在: 8/33 items
- generation miss（truth不在）: 25/33 items
- `score_kgmon` top-1 / top-2: 6/33 / 6/33 items
- `score_kgmon` selection miss（truthは存在するがrank > 2）: 2 items
- `score_full_probmul_3` top-1 / top-2: 6/33 / 6/33 items
- `score_full_probmul_3` selection miss: 2 items
- task oracle exact: 4/20
- KG top-2 task exact: 4/20
- probmul top-2 task exact: 4/20

2件のselection missは `142ca369[0]` と `20270e3b[0]` で、両scorerともtruth rank 5だった。
残る27件のKG top-2失敗itemのうち25件はgeneration missである。失敗16 taskはすべて少なくとも1 itemが
generation missであり、selectionだけを直せばtask exactになるtaskはなかった。

### task / item別

`n/shape` はunique candidate数 / truthと同shapeのcandidate数、`K/P` はKG/probmulのtruth rank、
`best` はpool内best candidateのcell accuracyである。`-` はtruth不在。

| task | oracle / KG / P task exact | item診断 |
|---|---|---|
| `0934a4d8` | 0 / 0 / 0 | `0: 9/9, K-/P-, best .740741` |
| `135a2760` | 1 / 1 / 1 | `0: 4/4, K1/P1, best 1.000000` |
| `136b0064` | 0 / 0 / 0 | `0: 1/1, K-/P-, best .721805` |
| `13e47133` | 0 / 0 / 0 | `0: 0/0, K-/P-, best .000000`; `1: 1/1, K-/P-, best .958889` |
| `142ca369` | 0 / 0 / 0 | `0: 8/8, K5/P5, best 1.000000`; `1: 7/7, K-/P-, best .968421` |
| `16b78196` | 0 / 0 / 0 | `0: 4/4, K-/P-, best .941111` |
| `16de56c4` | 0 / 0 / 0 | `0: 1/1, K-/P-, best .703704`; `1: 8/8, K-/P-, best .978836` |
| `1818057f` | 1 / 1 / 1 | `0: 2/2, K1/P1, best 1.000000` |
| `195c6913` | 0 / 0 / 0 | `0: 4/4, K-/P-, best .952222`; `1: 8/8, K-/P-, best .957778` |
| `1ae2feb7` | 0 / 0 / 0 | `0: 17/17, K-/P-, best .980000`; `1: 5/5, K-/P-, best .991667`; `2: 8/8, K1/P1, best 1.000000` |
| `20270e3b` | 0 / 0 / 0 | `0: 17/6, K5/P5, best 1.000000`; `1: 10/0, K-/P-, best .000000` |
| `20a9e565` | 0 / 0 / 0 | `0: 15/1, K-/P-, best .071429`; `1: 6/1, K-/P-, best .647059` |
| `21897d95` | 0 / 0 / 0 | `0: 4/4, K-/P-, best .253333`; `1: 6/6, K-/P-, best .354167` |
| `221dfab4` | 0 / 0 / 0 | `0: 3/3, K-/P-, best .825600`; `1: 7/7, K-/P-, best .923333` |
| `247ef758` | 0 / 0 / 0 | `0: 6/6, K-/P-, best .970000`; `1: 9/9, K-/P-, best .935185` |
| `269e22fb` | 0 / 0 / 0 | `0: 8/8, K-/P-, best .577500`; `1: 6/6, K-/P-, best .585000` |
| `271d71e2` | 0 / 0 / 0 | `0: 0/0, K-/P-, best .000000` |
| `36a08778` | 0 / 0 / 0 | `0: 5/5, K1/P1, best 1.000000`; `1: 11/11, K-/P-, best .986667` |
| `981571dc` | 1 / 1 / 1 | `0: 1/1, K1/P1, best 1.000000` |
| `aa4ec2a5` | 1 / 1 / 1 | `0: 2/2, K1/P1, best 1.000000` |

generation miss 25件のbest cell accuracyは、min / Q1 / median / Q3 / maxが
`0.0000 / 0.5775 / 0.8256 / 0.9589 / 0.9917`（mean `0.6810`）だった。
高cell-accuracyのnear missは多いが、exact grid自体がpoolにないためscorer変更では救えない。
一方、`20270e3b[1]` は10候補すべてshape違い、`13e47133[0]` と `271d71e2[0]` は候補ゼロであり、
明確なgeneration/fallback failureである。

### cap・candidate数・TTT lossの解釈

- cap到達は `195c6913`、`21897d95`、`221dfab4` の3 taskで、その6 itemsはすべてgeneration missだった。
  ただしuncapped taskにもgeneration missが多数あるため、capだけが原因とは断定できない。
- truth-present itemのunique candidate数は平均5.875、median 4.5、truth-absent itemは平均6.24、
  median 6であり、「候補数を増やすだけ」で解決する関係はこの20問では見えない。
- exact 4 taskのTTT loss medianは0.003416、failed 16 taskは0.012871だったが、failed側にも
  loss 0.0044程度のtaskがあり、標本も小さい。TTT loss単独を成功判定や調整根拠にはできない。
- fallback 6 items中、exactは1 item（`981571dc[0]`）だけだった。候補ゼロの2 itemsはいずれも失敗した。

### v5 + v6との再現差

task exact `4/20`、item exact `6/33`、全itemのexact/non-exact、unique candidate合計`203`は一致した。
一方、candidate samplesは`395 → 394`、top-1 cell accuracyは`75.5696% → 73.4825%`で、
task別candidate数も多くが一致しなかった。cap判定も `981571dc` は到達から未到達へ、
`21897d95` は未到達から到達へ変化した。したがって固定seedでもGPU並列実行と時刻capを含むrunは
bitwise deterministicではない。ただし主結論（exact成否とgeneration優位のbottleneck）は安定している。

完全なpost-hoc reportは
[`candidate_diagnostics20_report.json`](candidate_diagnostics20_report.json) に保存した。
Kaggle download原本と同じSHA-256
`f4389d22dee2588aa96ce3bea92679877138630c9971145f5c85e868c0009ac0`
であり、candidate shardを保持しなくても各itemのrank、coverage、best gridを再確認できる。
この診断に基づくscorer変更やlabel-driven tuning、変更scorerでのsubmission runは行っていない。
