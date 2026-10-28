# Experiment 027: Qwen3-VL Grid Perception

## Question

Does a Qwen model with a native vision encoder and two-dimensional visual
position encoding read ARC grids more accurately than the same model receiving
only a serialized numeric grid?

This experiment measures perception before attempting `input -> output` rule
induction. It does not fine-tune the model and it does not report ARC solve
accuracy.

## Fixed comparison

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Decoding: greedy, thinking disabled, at most 3,072 new tokens
- Precision: NF4 4-bit weights, bf16 compute
- Data: the first train input and output from the ten previously reviewed tasks
- Scale: 20 isolated grids and three modalities, for 60 responses

The three arms receive the same question:

1. `text`: compact JSON grid only.
2. `image`: a lossless ARC-palette rendering only.
3. `both`: the same image and JSON grid.

The fixed task IDs are:

`03560426`, `045e512c`, `05269061`, `05a7bcf2`, `05f2a901`,
`0607ce86`, `06df4c85`, `09629e4f`, `0962bcdd`, `0a1d4ef5`.

## Objective scoring

`arc_solver/perception.py` computes every expected answer directly from the
grid. Model responses are scored without an LLM judge:

- `reading`: dimensions, color counts, selected cells and complete rows;
- `objects`: exact 4/8-connected-component counts and component descriptions;
- `relations`: foreground bounding box, empty rows/columns, equal shapes and
  symmetry;
- `exact`: every requested field matches.

Results are reported separately for input and output grids. Raw generations,
parsed JSON, expected JSON, timing, token counts, model revision and input
hashes are retained. The runner resumes by a unique
`(task_id, role, modality)` key and never silently skips a failed response.

## Decision rule

Vision support is useful at this stage only if `both` improves the mean
`overall` score over `text`, without reducing `reading`, and the gain occurs on
more individual grids than the corresponding loss count. `image` diagnoses
what the visual path can recover by itself.

Even a positive result establishes only better isolated-grid perception.
Transformation inference and exact test-grid generation require a separate
follow-up experiment.

## Result

The completed RunPod run produced all 60 responses for 20 grids. Every
response parsed as a JSON object, but no response was exactly correct.

- Across all grids, `text` scored reading `0.127500`, objects `0.133333`,
  relations `0.170000`, and overall `0.143611`.
- `image` scored reading `0.002500`, objects `0.100000`, relations `0.110000`,
  and overall `0.070833`.
- `both` scored reading `0.152500`, objects `0.133333`, relations `0.200000`,
  and overall `0.161944`.
- The paired `both - text` overall comparison had 6 wins, 4 losses, and 10
  ties. The mean overall gain was `0.018333`; the mean reading gain was
  `0.025000`.
- By role, `text` versus `both` overall was `0.132778` versus `0.132778` on
  inputs and `0.154444` versus `0.191111` on outputs.
- Structured-response rate was `1.0` and exact rate was `0.0` for every
  modality/role aggregate.

The predefined decision rule passed: `both` improved mean overall score,
did not reduce reading, and had more paired wins than losses. The measured
effect supports using the visual input alongside JSON for this narrow
perception probe, especially on output grids, but the low absolute scores and
zero exact matches do not support a claim of reliable grid perception or ARC
solving.

The result-bearing invocation took `1121.112011` seconds (18m 41.112s);
summed generation time was `1118.849815` seconds. It ran on an NVIDIA GeForce
RTX 4090 with Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Transformers
4.57.6, Accelerate 1.15.0, BitsAndBytes 0.50.2, and qwen-vl-utils 0.0.14.
The model revision resolved from `main` to
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. The official training-challenges
SHA-256 was
`779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`.

Official task `0a1d4ef5` has a 2x3 first training output, so the fixed `(2,2)`
probe is represented as JSON `null` for that grid. This preserves the five
fixed probe keys without substituting a different cell. The output directory
also retains the failed-attempt logs and partial raw generations that exposed
this compatibility issue.

## Files

- `runner.py`: resumable Qwen3-VL benchmark.
- `run_config.json`: frozen result-bearing configuration.
- `analyze.py`: deterministic paired report generation.
- `runpod_setup.sh`: reproducible RunPod environment setup.
- `tests/test_qwen3vl_perception.py`: config, pairing and analysis tests.

