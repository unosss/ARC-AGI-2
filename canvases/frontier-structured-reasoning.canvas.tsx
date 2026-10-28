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

const root = "experiments/030_frontier_structured_reasoning";

const tasks = {
  "42f14c03": {
    ok: true,
    rule: "テンプレート内の穴と、外にある単色パーツを向き込みで照合し、穴を対応色で埋める。",
    alternatives: "色だけの対応／最大テンプレートの無条件選択",
    evidence: "全trainをセル単位で再構成。distractorは少なくとも1つの穴に対応パーツがない。",
  },
  "72207abc": {
    ok: true,
    rule: "活動行の三角数列 0,1,3,6,10,… の列に、入力seed色を循環配置する。",
    alternatives: "一定間隔／単純な色反復",
    evidence: "1,2,3,… と増える間隔と、trainごとに異なる色周期を同時に説明。",
  },
  "0d87d2a6": {
    ok: false,
    rule: "端のmarker対からguide行・列を引き、guideと交差する赤component全体を緑にする。",
    alternatives: "guide上だけを緑化／guideに接触したcomponentも緑化",
    evidence: "選択規則は全trainを完全再構成したが、testのguideに接する4×3矩形を見逃した。",
  },
  "bf32578f": {
    ok: true,
    rule: "色付きhalf-boundaryを内側seamで反射し、両境界に囲まれた背景だけを塗る。",
    alternatives: "固定幅の矩形fill／左端を反射軸にする",
    evidence: "矩形fillはV字の肩で過剰着色。内側境界なら太いtest rasterも一貫して処理。",
  },
  "7ee1c6ea": {
    ok: true,
    rule: "色5のframeの厳密な内部だけで、2つの非0・非5色を交換する。",
    alternatives: "grid全体で色交換",
    evidence: "frame外にある同じ色が変化しないtrainセルが、全体交換を反証。",
  },
  "ad38a9d0": {
    ok: true,
    rule: "componentの幾何形状を辞書として色へ写像する。",
    alternatives: "面積だけで色を決める／面積とbounding boxで決める",
    evidence: "面積3の直線とL字が別色になるため、面積だけの仮説を具体的に棄却。",
  },
  "7039b2d7": {
    ok: true,
    rule: "単色dividerで分かれた最大row/column runの直積を区画とし、各区画を1 pixelへ縮約する。",
    alternatives: "出力寸法=divider数+1",
    evidence: "境界上のdividerは空区画を作らない。run数ならそのtrain例を正確に説明。",
  },
  "b27ca6d3": {
    ok: true,
    rule: "2-cellの赤dominoだけを対象に、Chebyshev距離1の背景を緑にする。",
    alternatives: "孤立赤cellも同様に囲む",
    evidence: "全trainでdomino周辺と境界clipを再構成し、孤立赤cellは不変。",
  },
  "f8a8fe49": {
    ok: true,
    rule: "赤境界は保存、灰sourceを消去し、向かい合う最寄りの赤境界に対して灰を反射する。",
    alternatives: "灰を残した複製／遠い境界で反射",
    evidence: "水平・垂直の双方を同一の座標反射式で再構成。",
  },
  "fcc82909": {
    ok: true,
    rule: "各2×2 objectの下へ、そのobject内の異色数だけ、幅2の色3を伸ばす。",
    alternatives: "面積／非zero cell数に比例",
    evidence: "trainの高さ3,2／2,3／4,2,2を異色数がすべて説明。",
  },
} as const;

type TaskId = keyof typeof tasks;

function Process() {
  const steps = [
    "構造entityを抽出",
    "input/output対応を列挙",
    "複数仮説を生成",
    "train出力を完全生成",
    "不一致セルで反証",
    "生存仮説をtestへ適用",
  ];
  return (
    <Stack gap={8}>
      <H2>観測された推論フロー</H2>
      <Row gap={6} wrap>
        {steps.map((step, i) => (
          <React.Fragment key={step}>
            <Pill tone={i === 4 ? "warning" : "info"}>{i + 1}. {step}</Pill>
            {i < steps.length - 1 && <Text tone="tertiary">→</Text>}
          </React.Fragment>
        ))}
      </Row>
    </Stack>
  );
}

export default function FrontierStructuredReasoning() {
  const [taskId, setTaskId] = useCanvasState<TaskId>("selected-task", "0d87d2a6");
  const dispatch = useCanvasAction();
  const task = tasks[taskId];

  const openFile = (path: string) => dispatch({ type: "openFile", path });

  return (
    <div style={{ padding: 28, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={20}>
        <Stack gap={6}>
          <Text tone="tertiary">EXPERIMENT 030 · BLIND 10 TASKS</Text>
          <H1>検証可能な中間推論の分析</H1>
          <Text tone="secondary">
            正解を開く前に、観察・競合仮説・train完全予測・反証・最終test予測を保存した。
          </Text>
        </Stack>

        <Row gap={12} wrap>
          <Stat label="Task exact" value="9 / 10" tone="positive" />
          <Stat label="Micro cell" value="99.224%" tone="positive" />
          <Stat label="誤り" value="12 cells" tone="warning" />
          <Stat label="前回との差" value="±0 task" />
        </Row>

        <Card>
          <CardBody><Process /></CardBody>
        </Card>

        <Card>
          <CardHeader>
            <Row justify="space-between" align="center" wrap>
              <H2>task別の推論記録</H2>
              <Select
                value={taskId}
                onChange={(value) => setTaskId(value as TaskId)}
                options={Object.keys(tasks).map((id) => ({
                  value: id,
                  label: `${tasks[id as TaskId].ok ? "✓" : "✗"} ${id}`,
                }))}
                style={{ minWidth: 220 }}
              />
            </Row>
          </CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Row gap={8} align="center">
                <Pill tone={task.ok ? "success" : "danger"}>
                  {task.ok ? "Exact match" : "12-cell mismatch"}
                </Pill>
                <Text tone="tertiary">{taskId}</Text>
              </Row>
              <Stack gap={5}>
                <H3>選択した規則</H3>
                <Text>{task.rule}</Text>
              </Stack>
              <Divider />
              <Stack gap={5}>
                <H3>比較した仮説</H3>
                <Text tone="secondary">{task.alternatives}</Text>
              </Stack>
              <Stack gap={5}>
                <H3>反証・選択根拠</H3>
                <Text tone="secondary">{task.evidence}</Text>
              </Stack>
              <Row gap={8} wrap>
                <Button
                  variant="secondary"
                  onClick={() => openFile(`${root}/traces/${taskId}.json`)}
                >
                  完全なtraceを開く
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => openFile(`${root}/score.json`)}
                >
                  採点結果を開く
                </Button>
              </Row>
            </Stack>
          </CardBody>
        </Card>

        <Callout tone="warning" title="唯一の失敗は知覚崩壊ではない">
          0d87d2a6では、予測規則が3つのtrainを全セル完全一致させた。しかしtestでは、
          guideと重なるcomponentだけでなく、guideに辺で接するcomponentも緑化するのが正解だった。
          つまり失敗点は、trainで区別できない「交差」と「到達・接触」の意味選択にある。
        </Callout>

        <Card>
          <CardHeader><H2>この実験で分かったこと</H2></CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text>
                大きなモデルは固定の物体分割だけを使っていない。taskごとにcomponent、frame、
                divider run、反射境界、数列など異なる表現を組み立てていた。
              </Text>
              <Text>
                中間推論を記録しても精度は前回の9/10から上がらなかった。一方で、
                成功が偶然の出力一致ではなく、具体的な反証を伴う仮説選択だったことを確認できた。
              </Text>
              <Text>
                次の中心課題は「train完全一致」を超えて、複数の同値仮説からtestに通用する
                意図された因果・幾何 semanticsを選ぶ学習である。
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Row gap={8} wrap>
          <Button onClick={() => openFile(`${root}/ANALYSIS.md`)}>分析文書を開く</Button>
          <Button variant="secondary" onClick={() => openFile(`${root}/README.md`)}>
            実験protocolを開く
          </Button>
        </Row>
      </Stack>
    </div>
  );
}

