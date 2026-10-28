import React from "react";
import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Select,
  Stack,
  Stat,
  Text,
  useCanvasAction,
  useCanvasState,
} from "@cursor/canvas";

const root = "experiments/031_representation_causality";

const tasks = {
  "72207abc": {
    family: "indexed sequence",
    stability: "強く安定",
    tone: "success" as const,
    transfer: "2 / 2 exact",
    entities: "active row、ordered seed word、background",
    relations: "seed[n mod k] ↔ triangular column T(n)",
    invariance: "色番号の意味とgap長を捨て、順序とgrid境界を保持",
    operator: "三角数列の列へseed色を周期配置",
    ambiguity: "有限範囲lookupは形式上可能だが、意味的な分岐なし",
    concrete: "cell列を直接操作するのではなく、疎な出力を「数列の添字集合」として扱った。",
  },
  "7039b2d7": {
    family: "interval quotient",
    stability: "強く安定",
    tone: "success" as const,
    transfer: "2 / 2 exact",
    entities: "divider rows/columns、最大non-divider intervals",
    relations: "row interval × column interval ↔ output cell",
    invariance: "区間幅・絶対位置を捨て、区間数・順序・色を保持",
    operator: "各非空compartmentを1 cellへquotient化",
    ambiguity: "代表cell copyとuniform aggregationは観測上同値",
    concrete: "27×27等のpixel gridを、dividerが定める5×5等の区画構造へ置き換えた。",
  },
  "ad38a9d0": {
    family: "shape classes",
    stability: "局所的に不安定",
    tone: "warning" as const,
    transfer: "2 / 2 exact",
    entities: "最大4-connected components、normalized shapes",
    relations: "shape class ↔ target color",
    invariance: "translationは共通して捨てるが、rotation/reflectionの扱いが不一致",
    operator: "component geometryを保ち、class色へ置換",
    ambiguity: "translation-only、rotation invariant、D4 invariant",
    concrete: "今回のtestでは同じ答えだが、未知向きのcomponentではrunごとに予測が分岐し得る。",
  },
  "bf32578f": {
    family: "boundary geometry",
    stability: "意味的に不安定",
    tone: "danger" as const,
    transfer: "1 / 2 exact",
    entities: "half-boundary＋mirror＋interior / rowごとのsource interval",
    relations: "canvas中央 / endpoint列をreflection referenceにする",
    invariance: "closed-wall topologyを保持するrunと、row intervalだけを保持するrun",
    operator: "strict interior fill / endpoint基準の区間fill",
    ambiguity: "幅1のtrain境界では両operatorが同値",
    concrete: "誤runは軸3、正runは中央線3.5。testで列5を2セル過剰着色した。",
  },
  "0d87d2a6": {
    family: "incidence graph",
    stability: "安定した未識別性",
    tone: "warning" as const,
    transfer: "0 / 2 exact",
    entities: "border marker pairs、guide axes、red components",
    relations: "guide-component overlap / overlap-or-contact",
    invariance: "面積・縦横比・順序を捨て、incidenceとcomponent extentを保持",
    operator: "guideを描き、selected component全体を再着色",
    ambiguity: "trainにcontact-only componentがなく、2関係は識別不能",
    concrete: "3回ともoverlapを選び同じ12セルで失敗。contactへ介入すると完全一致した。",
  },
} as const;

type TaskId = keyof typeof tasks;

function TupleCard({
  label,
  symbol,
  value,
}: {
  label: string;
  symbol: string;
  value: string;
}) {
  return (
    <Card>
      <CardBody>
        <Stack gap={5}>
          <Row gap={7} align="center">
            <Pill tone="info">{symbol}</Pill>
            <H3>{label}</H3>
          </Row>
          <Text tone="secondary">{value}</Text>
        </Stack>
      </CardBody>
    </Card>
  );
}

export default function RepresentationCausality() {
  const [taskId, setTaskId] = useCanvasState<TaskId>("selected-task", "bf32578f");
  const dispatch = useCanvasAction();
  const task = tasks[taskId];
  const open = (path: string) => dispatch({ type: "openFile", path });

  return (
    <div style={{ padding: 28, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={20}>
        <Stack gap={6}>
          <Text tone="tertiary">EXPERIMENT 031 · NO TRAINING</Text>
          <H1>「異なる表現を組み立てる」とは何か</H1>
          <Text tone="secondary">
            raw gridから独立に表現を抽出し、別の実行者へ表現だけを渡して検証した。
          </Text>
        </Stack>

        <Row gap={12} wrap>
          <Stat label="表現のみでexact" value="7 / 10" tone="positive" />
          <Stat label="両run exact" value="3 / 5 tasks" />
          <Stat label="Micro cell" value="98.419%" tone="positive" />
          <Stat label="学習" value="なし" />
        </Row>

        <Callout tone="info" title="操作的な定義">
          task表現 Z = (E, R, I, O, A)：entities、relations、invariances、
          output operator、未解決 alternatives。名称が違うだけでは別表現ではない。
          同値関係・参照枠・保持情報・演算・曖昧性が変われば別表現である。
        </Callout>

        <Card>
          <CardHeader>
            <Row justify="space-between" align="center" wrap>
              <Stack gap={3}>
                <H2>task別の5要素</H2>
                <Text tone="secondary">3つの独立記録を意味レベルで比較</Text>
              </Stack>
              <Select
                value={taskId}
                onChange={(value) => setTaskId(value as TaskId)}
                options={Object.keys(tasks).map((id) => ({
                  value: id,
                  label: `${id} · ${tasks[id as TaskId].family}`,
                }))}
                style={{ minWidth: 290 }}
              />
            </Row>
          </CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Row gap={8} align="center" wrap>
                <Pill tone={task.tone}>{task.stability}</Pill>
                <Pill>{task.transfer}</Pill>
                <Text tone="tertiary">{task.family}</Text>
              </Row>
              <Text>{task.concrete}</Text>
              <Divider />
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                  gap: 10,
                }}
              >
                <TupleCard label="Entities" symbol="E" value={task.entities} />
                <TupleCard label="Relations" symbol="R" value={task.relations} />
                <TupleCard label="Invariances" symbol="I" value={task.invariance} />
                <TupleCard label="Operator" symbol="O" value={task.operator} />
                <TupleCard label="Alternatives" symbol="A" value={task.ambiguity} />
              </div>
              <Button
                variant="secondary"
                onClick={() => open(`${root}/representations/${taskId}_r1.json`)}
              >
                run 1の完全な表現を開く
              </Button>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader><H2>最も明確な因果介入</H2></CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text>
                `0d87d2a6`で寸法・guide・component・色・描画順を固定し、
                選択relationだけを変更した。
              </Text>
              <Row gap={12} wrap>
                <Stat label="overlap only" value="12 cells wrong" tone="warning" />
                <Stat label="overlap + contact" value="Exact" tone="positive" />
                <Stat label="条件間の差" value="12 cells" />
              </Row>
              <Text tone="secondary">
                差はcontact-onlyの4×3 componentだけに局在した。したがって外在表現は
                実行出力を因果的に制御した。この時点ではselected component一覧を
                与えていたため、Experiment 032でそのshortcutを除去した。
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Callout tone="warning" title="精度より重要な発見">
          失敗は表現生成の全面崩壊ではなかった。`bf32578f`はtrain同値な反射参照枠、
          `0d87d2a6`はtrain同値な接触述語の選択で失敗した。安定した表現でも、
          少数train例が区別しない一般化を全runが同じように誤選択できる。
        </Callout>

        <Card>
          <CardHeader><H2>Experiment 032 · relation自体を実行できるか</H2></CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Row gap={12} wrap>
                <Stat label="Exact conditions" value="6 / 6" tone="positive" />
                <Stat label="Exact cases" value="14 / 14" tone="positive" />
                <Stat label="Cell accuracy" value="100%" tone="positive" />
              </Row>
              <Text>
                selected set、正規化shape、出力intervalを与えず、entity座標と抽象relation
                だけから新規配置を実行した。overlap/contactは8・8・10セル、反射規則は
                3・6セル、translation/D4は10・11セルだけ条件間で変化し、すべて期待support
                と完全一致した。
              </Text>
              <Text tone="secondary">
                これにより一覧コピーではなく、外在化されたrelationを再利用して計算できることを確認。
                残る中心課題は、raw gridから複数候補を作り、train同値候補から意図されたものを選ぶ段階。
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Row gap={8} wrap>
          <Button onClick={() => open(`${root}/ANALYSIS.md`)}>詳細分析を開く</Button>
          <Button variant="secondary" onClick={() => open(`${root}/score.json`)}>
            採点データを開く
          </Button>
          <Button variant="secondary" onClick={() => open(`${root}/README.md`)}>
            protocolを開く
          </Button>
          <Button
            variant="secondary"
            onClick={() => open("experiments/032_relation_counterfactuals/ANALYSIS.md")}
          >
            Experiment 032分析
          </Button>
        </Row>
      </Stack>
    </div>
  );
}

