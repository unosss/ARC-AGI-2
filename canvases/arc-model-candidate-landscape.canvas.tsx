import React from "react";
import {
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
  useCanvasState,
  useHostTheme,
} from "@cursor/canvas";

type Family = "all" | "grid" | "vlm" | "latent";
type Candidate = {
  name: string;
  family: Exclude<Family, "all">;
  params: string;
  representation: string;
  reasoning: string;
  deploy: string;
  arc2: string;
  readiness: "今すぐ" | "要実装" | "公開待ち";
  verdict: string;
  priority: number;
  source: string;
};

const candidates: Candidate[] = [
  {
    name: "Qwen3-4B grids15 + DiARC",
    family: "grid",
    params: "3.6B effective",
    representation: "16-token grid。現在のbaseと完全に同系統",
    reasoning: "正解と「もっともらしい誤答」をDPOで識別",
    deploy: "既存pipelineへLoRA adapter追加。T4×2 notebook実績あり",
    arc2: "33.06%（論文）",
    readiness: "今すぐ",
    verdict: "最優先。新モデルではないが、現在の手札へ最短で追加できる明確な上位版。",
    priority: 1,
    source: "https://github.com/szu-tera/DiARC",
  },
  {
    name: "VARC ViT",
    family: "grid",
    params: "18M",
    representation: "64×64 canvasのimage-to-image。2×2 patch",
    reasoning: "10-block ViT＋問題別TTT。規則をpixel変換として学習",
    deploy: "非常に軽量。code・checkpoint・予測公開済み",
    arc2: "8.3% single / 11.1% ensemble",
    readiness: "今すぐ",
    verdict: "2D知覚仮説を最も直接検証できる。Qwenとは誤答相関も異なる可能性が高い。",
    priority: 2,
    source: "https://github.com/lillian039/VARC",
  },
  {
    name: "GRAM",
    family: "latent",
    params: "10.9M",
    representation: "30×30 gridを再帰的なlatent trajectoryで更新",
    reasoning: "確率的な複数trajectory。完成案を反復修正",
    deploy: "サイズは理想的。ただし調査時点で実装・checkpointを確認できず",
    arc2: "11.1%（論文）",
    readiness: "公開待ち",
    verdict: "今回考えた反復修正に最も近い本命。ただし公開物が揃うまで再現コストが高い。",
    priority: 3,
    source: "https://ahn-ml.github.io/gram-website/",
  },
  {
    name: "TRM",
    family: "latent",
    params: "7M",
    representation: "30×30 padded grid＋2-layer recursive network",
    reasoning: "共有networkを反復し、answerとlatentを更新",
    deploy: "極小。codeとARC Prize検証checkpointあり",
    arc2: "7.8% paper / 6.2% verified",
    readiness: "今すぐ",
    verdict: "再帰推論の再現可能な基準。単独性能よりQwen/VARCとのensemble候補。",
    priority: 4,
    source: "https://github.com/SamsungSAILMontreal/TinyRecursiveModels",
  },
  {
    name: "Qwen3-VL-4B-Thinking",
    family: "vlm",
    params: "4B",
    representation: "画像＋native tokenizer。空間理解を強化したVLM",
    reasoning: "Thinking版。AIME-25 74.5、長い推論出力に対応",
    deploy: "BF16 checkpointでも提出圏内。LoRAも48GBなら可能",
    arc2: "未報告",
    readiness: "要実装",
    verdict: "一般VLMでは第一候補。ただしARCでは画像化がtoken-gridより有利という証拠はまだない。",
    priority: 5,
    source: "https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking",
  },
  {
    name: "Qwen3-VL-8B-Thinking",
    family: "vlm",
    params: "8B",
    representation: "Qwen3-VLの視覚encoder＋DeepStack",
    reasoning: "4Bより強いThinking版。AIME-25 80.3",
    deploy: "推論は可能。問題別学習はQLoRA推奨",
    arc2: "未報告",
    readiness: "要実装",
    verdict: "4Bで信号が出た場合のscale-up先。最初から試すと原因分離が難しい。",
    priority: 6,
    source: "https://github.com/QwenLM/Qwen3-VL",
  },
  {
    name: "MiniCPM-V 4.5",
    family: "vlm",
    params: "8.7B",
    representation: "SigLIP2＋高密度visual resampler",
    reasoning: "fast/deep thinking切替。OCR・図表知覚が強い",
    deploy: "AWQ/int4あり。BF16学習は重くQLoRA向き",
    arc2: "未報告",
    readiness: "要実装",
    verdict: "知覚は強いがARCの規則帰納との接続は未知。VLM比較の二番手。",
    priority: 7,
    source: "https://huggingface.co/openbmb/MiniCPM-V-4_5",
  },
  {
    name: "Kimi-VL-A3B-Thinking",
    family: "vlm",
    params: "16B total / 3B active",
    representation: "native-resolution MoonViT＋MoE language decoder",
    reasoning: "long-thinking SFT/RL。MathVistaなどで強い",
    deploy: "計算は軽いが全16B weightsを保持。量子化前提",
    arc2: "未報告",
    readiness: "要実装",
    verdict: "推論/知覚の均衡は良い。custom MoE運用がQwen3-VLより複雑。",
    priority: 8,
    source: "https://github.com/moonshotai/Kimi-VL",
  },
  {
    name: "LPN",
    family: "latent",
    params: "178M",
    representation: "入出力例をlatent programへencode",
    reasoning: "test-timeに連続latent空間をgradient search",
    deploy: "weightsは軽いが高精度設定はtest-time FLOPsが大きい",
    arc2: "未報告（ARC-AGI-1 15.5%）",
    readiness: "今すぐ",
    verdict: "固定DSLを避けたprogram inductionとして重要。ただしARC-2根拠と時間予算が弱い。",
    priority: 9,
    source: "https://github.com/clement-bonnet/lpn",
  },
  {
    name: "GLM-4.1V-9B-Thinking",
    family: "vlm",
    params: "10.3B",
    representation: "高解像度VLM、任意aspect ratio",
    reasoning: "視覚reasoning RL。10B級benchmarkで強い",
    deploy: "BF16は24GBで窮屈。量子化推論/QLoRA向け",
    arc2: "未報告",
    readiness: "要実装",
    verdict: "能力は高いが今回の制約ではQwen3-VL-4Bより優先度が低い。",
    priority: 10,
    source: "https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking",
  },
];

const familyOptions = [
  { value: "all", label: "全候補" },
  { value: "grid", label: "2D grid特化" },
  { value: "latent", label: "再帰・latent推論" },
  { value: "vlm", label: "汎用VLM" },
];

function CandidateRow({ candidate }: { candidate: Candidate }) {
  const theme = useHostTheme();
  const tone =
    candidate.readiness === "今すぐ"
      ? "success"
      : candidate.readiness === "公開待ち"
        ? "warning"
        : "info";
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "42px minmax(180px, 1.1fr) 100px minmax(210px, 1.5fr) minmax(180px, 1fr)",
        gap: 12,
        alignItems: "start",
        padding: "14px 0",
        borderBottom: `1px solid ${theme.stroke.secondary}`,
      }}
    >
      <Text tone="tertiary">#{candidate.priority}</Text>
      <Stack gap={5}>
        <H3>{candidate.name}</H3>
        <Row gap={6} wrap>
          <Pill tone={tone}>{candidate.readiness}</Pill>
          <Pill>{candidate.params}</Pill>
          <Pill>{candidate.arc2}</Pill>
        </Row>
        <a
          href={candidate.source}
          target="_blank"
          rel="noreferrer"
          style={{ color: theme.text.link, fontSize: 12 }}
        >
          一次情報を開く
        </a>
      </Stack>
      <Text>{candidate.params}</Text>
      <Stack gap={6}>
        <Text>{candidate.representation}</Text>
        <Text tone="secondary">{candidate.reasoning}</Text>
        <Text tone="tertiary">{candidate.deploy}</Text>
      </Stack>
      <Text>{candidate.verdict}</Text>
    </div>
  );
}

export default function ArcModelCandidateLandscape() {
  const [family, setFamily] = useCanvasState<Family>("model-family", "all");
  const visible = candidates.filter(
    (candidate) => family === "all" || candidate.family === family,
  );

  return (
    <div style={{ padding: 28, maxWidth: 1240, margin: "0 auto" }}>
      <Stack gap={22}>
        <Stack gap={6}>
          <Text tone="tertiary">ARC-AGI-2 · MODEL LANDSCAPE · 2026-10-26</Text>
          <H1>2D知覚・推論・提出可能性で選ぶモデル候補</H1>
          <Text tone="secondary">
            公開論文・公式repository・model cardを基準に比較。ARC-2値は評価条件が異なるため、
            同一条件の順位ではなく候補選定の根拠として扱う。
          </Text>
        </Stack>

        <Row gap={12} wrap>
          <Stat label="最優先" value="DiARC" tone="positive" />
          <Stat label="純2D本命" value="VARC 18M" tone="positive" />
          <Stat label="反復推論本命" value="GRAM 10.9M" tone="warning" />
          <Stat label="汎用VLM本命" value="Qwen3-VL 4B" />
        </Row>

        <Callout tone="info" title="最も重要な発見">
          現在使っている checkpoint は Qwen3-4B grids15 そのものです。DiARCは同じbaseへ、
          plausibleだが規則として誤った出力を負例にするDPO adapterを追加し、ARC-AGI-2で
          33.06%を報告しています。別モデル探索の前に比較できる、最短かつ最も強い候補です。
        </Callout>

        <Card>
          <CardHeader>
            <Row justify="space-between" align="center" wrap>
              <H2>候補一覧</H2>
              <Select
                value={family}
                onChange={(value) => setFamily(value as Family)}
                options={familyOptions}
                style={{ minWidth: 180 }}
              />
            </Row>
          </CardHeader>
          <CardBody>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "42px minmax(180px, 1.1fr) 100px minmax(210px, 1.5fr) minmax(180px, 1fr)",
                gap: 12,
                paddingBottom: 8,
              }}
            >
              <Text tone="tertiary">順位</Text>
              <Text tone="tertiary">モデル・根拠</Text>
              <Text tone="tertiary">規模</Text>
              <Text tone="tertiary">表現・推論・実行</Text>
              <Text tone="tertiary">今回の判断</Text>
            </div>
            {visible.map((candidate) => (
              <CandidateRow key={candidate.name} candidate={candidate} />
            ))}
          </CardBody>
        </Card>

        <Divider />

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
          <Stack gap={8}>
            <H2>実験順序</H2>
            <Text>1. 公開DiARC adapterを現在のQwen3-4Bへ適用し、同じ難問panelで比較</Text>
            <Text>2. VARC 18Mを独立系2D solverとして再現し、Qwenとの誤答相関を測定</Text>
            <Text>3. TRM checkpointを軽量ensemble候補として評価</Text>
            <Text>4. Qwen3-VL-4Bは画像入力とtext-grid入力の両方で小規模比較</Text>
          </Stack>
          <Stack gap={8}>
            <H2>今は後回し</H2>
            <Text>
              8〜10Bの汎用VLMは提出可能ですが、ARC-2での直接根拠がありません。
              4B VLMで視覚入力の信号が確認できてからscaleする方が原因を分離できます。
            </Text>
            <Text>
              GRAMは構造的には最も魅力的ですが、公開実装を確認できていません。
              論文から再実装するより、公開を待ちながらTRM/VARCを先に比較する方が現実的です。
            </Text>
          </Stack>
        </div>

        <Callout tone="warning" title="benchmarkの読み方">
          TRMはpaper 7.8%に対してARC Prize検証6.2%、HRMはpaper 5%に対して半非公開検証2%です。
          小型ARC専用モデルの公称値には再現差があります。一方DiARC/Qwen値も学習データと
          test-time computeが異なるため、最終判断はこのrepositoryの同一panelで行う必要があります。
        </Callout>
      </Stack>
    </div>
  );
}
