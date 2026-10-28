# Qwen3.8-27B ARC agent: acquisition and feasibility

Acquired and piloted 2026-10-23. The initial public-source inspection is
retained below, followed by the one-task quantized feasibility result.

## One-task RTX 4090 quantized pilot

The fixed task was `135a2760`, loaded from Experiment 015 historical challenge
data. Its solution stayed in `sealed_labels/solution.json`; `pilot_runner.py`
opened that file only after writing and SHA-256 sealing the complete trace and
submission. This was one attempt only, not a multi-task run.

### Exact runtime

- GPU: one RTX 4090, 24,564 MiB.
- Model: `6block/Qwen3.8-27B-GGUF`,
  revision `5312b0f1c3705b45a48a18f051b4dc482627c17f`,
  `Qwen3.8-27B-Q4_K_M.gguf`, 17,743,780,512 bytes, file SHA-256
  `038b8d86da2e388e4c3f5bafdf5a8aa4dcb630861a18d430b3a79c538c1a3beb`.
  Q4_K_M fit, so the Q4_K_S fallback was not used.
- Server: llama.cpp `b11669`, CUDA 12.8 binary SHA-256
  `3ac6495070ae9627825dbe706de3b580fdbedd2fbe079197b2c6c81982dbd0f9`.
  Arguments: 32,768 context, all 99 layers on GPU, one slot, flash attention,
  Q8_0 K/V cache, Jinja tool templates, metrics enabled.
- Observed server VRAM: 17,906 MiB after the run. Root disk: 19/50 GiB used.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, thinking enabled; seed 0.
- Harness budgets: one attempt, 40 turns, 120,000 generated tokens, 28,000
  estimated-token context budget, 16,384 tokens per turn, early-submit nudge
  every six Python calls, and a documented 10,800-second operational wall
  deadline. No optional unpublished harness enhancements were enabled.

Before the ARC attempt, the llama.cpp OpenAI endpoint was required to call the
`python` tool with exact code `print(2+3)`. It returned a native `tool_calls`
response (321 prompt tokens, 54 completion tokens), establishing compatibility
with the acquired harness.

### Outcome

The attempt stopped on its generated-token budget after 2,726.6 seconds:

- status `failed: token budget`; 13 model turns;
- 120,923 completion tokens and 74,728 cumulative prompt tokens;
- 44.35 generated tokens/second end-to-end;
- seven Python calls, zero `submit` calls, one early-submit nudge;
- six 16,384-token turns ended at the per-turn length limit;
- no submitted program, therefore no train feedback or candidate test grid;
- final fallback grid was the test input; held-out exact score **0/1**.

This is a concrete negative harness-feasibility result, not evidence about the
model's general ARC accuracy. Q4_K_M serving and tool calls work on one 4090,
but the public-default REPL policy is operationally ineffective for this
quantization/configuration because the model repeatedly consumes full reasoning
turns and ignores both cutoff instructions and the submit nudge. The trace
SHA-256 is
`d43b7c9496a9f04be9832c7cd9c218a6bebb1f3e20c6db024fc6ba1bb12748ca`.
All turns, Python source/output, token counts, programs (empty), fallback final
grid, score, and full server timing log are under
`kaggle_runs/qwen38_repl_4090/`.

## Controlled force-submit A/B

One treatment attempt was run on the same live server, task `135a2760`, seed,
model, sampling, 40-turn limit, 120,000 generated-token budget, 28,000-token
context budget, six-call nudge frequency, 16,384-token per-request server
maximum, and 10,800-second operational deadline. The treatment config is
identical to the baseline config after removing its sole added field,
`force_submit_after: 6`. No other `run_attempt` treatment was enabled.

The named forced-tool protocol was **not accepted by llama.cpp b11669**. The
HTTP request returned success, and the isolated one-tool probe happened to emit
`submit`, but the server logged this exact warning for that probe and both
treatment requests:

`Wrong type supplied for parameter 'tool_choice', using default value:
[json.exception.type_error.302] type must be string, but is object`

Thus the OpenAI/vLLM-style named-tool object emitted by the acquired client was
silently downgraded to llama.cpp's default tool selection. Of the two turns on
which the harness requested forced submission, one consumed 16,384 tokens and
ended at the length limit without a tool call; the next selected `python`.
The protocol was not changed to llama.cpp's string form and the attempt was not
retried. Consequently, this run verifies the backend blocker but is **not a
valid efficacy test** of enforced submission.

Observed baseline versus treatment:

- Turns: 13 versus 13.
- Python calls: 7 versus 7.
- Submit calls: 0 versus 0. The treatment recorded two force attempts, both
  downgraded by the backend.
- Best train fit: no program evaluated in either arm (`best_ok = -1` of 2).
- Verified: false versus false; both failed on the generated-token budget.
- Held-out output exactness: 0/1 versus 0/1; both emitted the input-grid
  fallback.
- Completion tokens: 120,923 versus 120,923; cumulative prompt tokens: 74,728
  versus 74,728.
- Wall time: 2,726.617 s versus 2,722.703 s.
- End-to-end completion throughput: 44.349 versus 44.413 tokens/s.

After removing elapsed timestamps and the treatment-only force counter, the
attempt records are identical. The treatment trace was written and SHA-256
sealed before the held-out solution was opened; its digest is
`4097d1b14db31f8eb7693980fde1f529e84dbeb6d9b7a32bf2031387d342ec5f`.
Separate treatment artifacts, including the raw compatibility responses and
post-run backend diagnosis, are under
`kaggle_runs/qwen38_repl_4090/output_force_submit_6/`; its run log is under
`kaggle_runs/qwen38_repl_4090/run_force_submit_6/`.

### Corrected llama.cpp force-submit treatment

The treatment was rerun with an explicit runner-side llama.cpp compatibility
mode; the acquired public `client.py` was not modified. Ordinary turns still
sent both `python` and `submit` with `tool_choice: "auto"`. A forced turn sent
only the `submit` schema with llama.cpp's string
`tool_choice: "required"`. Unit tests assert both payload forms. The corrected
config preserved the baseline model, task, seed, sampling, 40-turn limit,
120,000-token budget, 28,000-token context budget, six-call nudge, 16,384
per-turn maximum, and 10,800-second deadline; all unrelated treatments
remained disabled.

Before the controlled attempt, a 256-token live preflight recorded its complete
request and response. It exposed only `submit`, used `"required"`, and returned
a native `submit` tool call containing an identity transform. The runner's
repeat preflight did the same. The backend's wrong-type warning count remained
three before and after the corrected run, confirming that no corrected request
was downgraded.

The corrected run made two forced requests after the sixth Python call. The
first still spent 16,384 tokens reasoning and reached its length limit before a
call; `required` cannot force a call before the model exhausts its generation
budget. The next forced request did call `submit` on turn 13. It submitted an
explicit identity placeholder, reproduced **0/2** demonstrations, and received
the intended cell-level failure diff. The model then made one more Python call
and stopped on the generated-token budget.

Baseline versus invalid treatment versus corrected treatment:

- Turns: 13 / 13 / 14.
- Python calls: 7 / 7 / 7.
- Submit calls: 0 / 0 / **1**.
- Forced requests: 0 / 2 downgraded / 2 accepted.
- Best train fit: -1 / -1 / **0 of 2**.
- Completion tokens: 120,923 / 120,923 / 127,396. The runner checks the
  cumulative budget before each turn, so the final allowed turn can overshoot.
- Cumulative prompt tokens: 74,728 / 74,728 / 85,607.
- Wall time: 2,726.617 / 2,722.703 / 2,891.458 seconds.
- Held-out exact score: 0/1 / 0/1 / 0/1.

Thus the compatibility correction changed the mechanism as intended: unlike
the invalid treatment, it elicited a real submit and exposed training feedback.
It did not improve this one-task, one-seed outcome because the forced program
was only a placeholder and the model failed to turn the returned diff into a
correct transform before exhausting its budget. The complete trace was written
and SHA-256 sealed before label access; its digest is
`6101bfc733e86fd75624c4c0e8eb177b4d4c69616b9ce9eb6e20a0045b0e7466`.
The preflight request/response, attempt, sealed trace, score, backend
verification, and three-arm comparison are under
`kaggle_runs/qwen38_repl_4090/output_force_submit_6_llamacpp_corrected/`;
the run log is under
`kaggle_runs/qwen38_repl_4090/run_force_submit_6_llamacpp_corrected/`.

### 4,096-token sequential comparison

A controlled sequential comparison was run on the same live server on
2026-10-23. Arm A completed first; its trace was written and SHA-256 sealed
before the held-out label was opened and before Arm B started. Arm B was then
run, sealed, and scored in the same way. Both arms retained task `135a2760`,
seed 0, sampling, 40 turns, 120,000 generated tokens, the 28,000-token context
budget, nudge frequency six, and the 10,800-second operational deadline. All
optional treatments remained off. Arm A changed only `turn_max_tokens` from
16,384 to 4,096 (plus the transport-only llama.cpp compatibility client). Arm B
additionally set `force_submit_after: 2`.

On Arm B's forced turns, the request exposed only `submit` and sent
`tool_choice: "required"`. The live preflight returned a native submit call.
The server's inherited wrong-type warning count remained three after Arm B, so
the corrected requests were not silently downgraded.

| Arm | Exact | Verified | Best train fit | Turns | Python | Submit | Cut | Generated | Prompt | Wall (s) | Tok/s |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior 16k baseline | 0/1 | no | none (-1/2) | 13 | 7 | 0 | 6 | 120,923 | 74,728 | 2,726.617 | 44.349 |
| Prior 16k corrected force=6 | 0/1 | no | 0/2 | 14 | 7 | 1 | 6 | 127,396 | 85,607 | 2,891.458 | 44.060 |
| Arm A: 4k, force=0 | 0/1 | no | none (-1/2) | 33 | 5 | 0 | 28 | 122,648 | 221,715 | 2,735.983 | 44.828 |
| Arm B: 4k, force=2 | 0/1 | no | 0/2 | 32 | 4 | 1 | 27 | 121,649 | 184,332 | 2,727.262 | 44.605 |

The 4k cap did deliver `THINKCAP_HINT` after every
`finish_reason="length"` turn that had no tool call: 28 times in Arm A and 27
times in Arm B. A length-finished response that already contained a parsed tool
call was executed directly and did not need the hint. The model rarely acted on
the hint immediately: the next turn transitioned to a tool on 2 of 27
observable opportunities in Arm A and 3 of 26 in Arm B. Most cut turns resumed
long analysis instead.

Arm A correctly recognized that the task repairs corrupted periodic patterns,
but repeatedly re-derived the rule and never submitted a program. Arm B's first
forced request, on turn 7, did produce a real submit. The submitted program was
a stale, unrelated heuristic that changed the top-left cell of every uniform
2x2 block to zero; it reproduced 0/2 demonstrations. The resulting cell-level
diff helped the model reason toward periodic-pattern repair, but it never
submitted a refinement. Eleven subsequent required-submit requests each spent
the full 4,096-token allowance without calling the only exposed tool. This
confirms that llama.cpp's `required` setting constrains any emitted call but
does not compel the model to call before exhausting its generation budget.

Neither treatment should be adopted as configured. The 4k cap multiplied cut
turns and cumulative prompt tokens without improving exactness, verification,
or candidate quality. Force-after-two obtained earlier feedback than
force-after-six, but the candidate was premature and the repeated forced turns
then became length-limited analysis loops. A future experiment should change
the action policy rather than only lowering the cap—for example, a separate
short non-thinking action turn after a cut—while retaining the verified
llama.cpp one-tool plus `"required"` compatibility mechanism.

Arm A's sealed trace digest is
`2233142bee377ad5a3a34020eb363aa5225a8b7a09840297e6d9e538471f3ef6`;
Arm B's is
`80423c59ebc494ec1c2d15999d37d6ebda6db990b29a9c0c006ebace39bb1b38`.
The complete artifacts and run logs are under
`kaggle_runs/qwen38_repl_4090/output_thinkcap_4096_a/`,
`output_thinkcap_4096_force2_b/`, `run_thinkcap_4096_a/`, and
`run_thinkcap_4096_force2_b/`. Their machine-readable comparison is
`kaggle_runs/qwen38_repl_4090/thinkcap_4096_sequential_comparison.json`.

Important caveats: llama.cpp tokenization/chat-template behavior is not the
published four-L4 vLLM stack; Q4_K_M changes weights; only one selected task and
one seed were tested; the public notebook itself launched direct mode; and this
pilot neither reproduces nor tests the public 15.69 leaderboard score or the
unshared 45–79% claims.

## Public artifacts acquired

- Kaggle notebook:
  <https://www.kaggle.com/code/madarshbb/arc-qwen-3-8-27b-agent>
- Discussion:
  <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546>
- Exact notebook source and metadata are under `public_notebook/`.
- The notebook's published output files and log are under
  `public_notebook/output_files/`.

Notebook attachments:

- Model
  `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`,
  version 1: **30,890,069,847 bytes** uncompressed. It contains sixteen
  approximately 1.524 GB layer shards, `outside.safetensors` (6,007,102,112
  bytes), and `mtp.safetensors` (477,202,224 bytes).
- Dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`: 180 files,
  **5,187,963,152 bytes** in the first/complete 200-file API page. Important
  files include vLLM 0.19.0 (432,281,473-byte wheel), Transformers 4.57.6, and
  `requirements.lock`.
- Dataset `keithtyser/taaf-duck-qwen38-serving-v1`: 78 files,
  **12,901,192 bytes**. It supplies a Transformers 5.8.0 overlay plus TAAF/ARC3
  reference source and serving metadata.
- Competition data: `arc-prize-2026-arc-agi-2`.

## What the public notebook actually runs

The source contains both a Python-REPL program-synthesis harness and a direct
answer path. However, the published launch configuration explicitly sets
`"mode": "direct"`. Therefore the public **15.69** notebook does **not** run its
Python/submit tool loop in the submitted configuration.

Processing flow:

1. Immediately write a valid fallback submission by copying each test input
   into both attempts.
2. On an ordinary notebook save, stop after the fallback. The downloaded output
   log confirms this path: 240 tasks, zero outputs answered by the agent.
3. Only on a Kaggle competition rerun, install the pinned offline wheelhouse and
   TAAF Transformers overlay.
4. Start a local vLLM OpenAI-compatible server using tensor parallelism 4,
   tool parser `qwen3_coder`, reasoning parser `qwen3`, prefix caching, 90% GPU
   memory utilization, maximum model length 81,920, and maximum 16 sequences.
   A startup ladder tries FP8 KV cache, Qwen3.5 MTP speculative decoding, generic
   MTP, and progressively plainer fallbacks.
5. In the active direct path, process shorter prompts first and request two
   samples per task at reasoning effort `xhigh`, temperature 1.0, top-p 0.95,
   top-k 20, thinking budget 40,960, and maximum 49,152 output tokens. Ten
   requests may run concurrently. A short non-thinking rescue request is used
   when no grid is parsed.
6. Atomically update the submission after completed answers and stop with a
   35-minute reserve inside Kaggle's 12-hour limit.

The included but inactive REPL path exposes `python(code)` and `submit(code)`.
Python calls receive numpy arrays and ARC helpers in isolated subprocesses.
Submitted `transform(grid)` programs are checked against all demonstrations;
cell-level failure diffs are returned, and demo-verified programs are used to
produce/vote on test outputs.

## RunPod environment

SSH target inspected: `root@213.192.2.77:40020`.

- GPU: **1 x NVIDIA GeForce RTX 4090**, UUID
  `GPU-3e006944-6bb9-bac0-3f88-75655a275dc2`, compute capability 8.9.
  `nvidia-smi` reports 24,564 MiB total and 24,082 MiB free; PyTorch reports
  25,252,724,736 bytes addressable. Driver 595.91.07; driver-supported CUDA
  13.2.
- RAM: **1,081,934,258,176 bytes total**, 951,355,011,072 bytes available at
  inspection; no swap.
- Container root disk: **53,687,091,200 bytes total**, 53,670,469,632 bytes
  available. No separate `/workspace` mount was present. Host storage devices
  are visible, but their large host mounts are not general writable model
  storage inside this container.
- Shared memory: 71,000,002,560 bytes.
- CPU: 256 logical CPUs, dual-socket AMD EPYC 7763.
- Python: 3.12.3 at `/usr/local/bin/python`.
- PyTorch: 2.8.0+cu128; runtime CUDA 12.8; CUDA available.
- CUDA 12.8 runtime directory exists, but `nvcc` is not installed.
- Installed inference packages: PyTorch and Triton are present; vLLM, SGLang,
  Transformers, and FlashAttention are not installed.

## Feasibility

This pod cannot host the referenced checkpoint fully on its single GPU for the
published serving setup. The checkpoint is 30,890,069,847 bytes (about 28.77
GiB), already larger than the GPU before KV cache and runtime allocations.
Even a language-model-only load of the layer shards is too tight for vLLM at
90% utilization once non-layer weights and runtime memory are included.
CPU offload or a smaller quantization would materially change performance and
would not reproduce the published L4x4 configuration.

The intended backend is vLLM 0.19.0 with the supplied Transformers 5.8.0
overlay, exposed as a local OpenAI-compatible server. The public notebook uses
four-way tensor parallelism on 4 x L4. The RTX 4090 is an Ada GPU and can
execute FP8, but GPU count/VRAM—not basic FP8 capability—is the immediate
blocker. The 50 GiB container disk can hold the 30.89 GB checkpoint, but leaves
limited room for the 5.19 GB wheelhouse, installations, caches, and artifacts.

## Public evidence versus unshared claims

- Public, attributable result: the notebook page labels version 2 as **15.69**
  on L4 x4. Its saved quick-run output is only a fallback and is not evidence of
  that score's per-task behavior.
- Discussion statements such as 45% from an NVARC ensemble, 87% under a
  time-constrained run, roughly 79% ± 5% CV, and roughly 45%/52 of 120 at
  pass@2 are participant claims with different or unspecified evaluation
  settings. They are not reproduced by the public notebook and do not provide a
  complete recipe.

Missing for a faithful reproduction of the stronger REPL claims:

- exact active REPL configuration and prompt/harness revision;
- evaluation split, task IDs, scoring protocol, and leakage controls;
- complete traces, seeds, attempt policy, and candidate aggregation;
- measured throughput and scheduling under the 12-hour L4x4 constraint;
- any fine-tuning/SFT data or weights, if used;
- exact ensemble components and selection logic for ensemble scores.

## Reproduction boundary

The original FP8 checkpoint still requires the multi-GPU setup described
above. This pilot instead established the narrower result that Q4_K_M fits and
the acquired REPL harness executes on one RTX 4090. It does not make the
quantized llama.cpp setup a reproduction of the notebook's vLLM direct path or
of any reported leaderboard score.
