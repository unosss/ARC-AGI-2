#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/root/qwen38_repl_4090}"
MODEL="${MODEL:-Qwen3.8-27B-Q4_K_M.gguf}"
SERVER="$(find "$ROOT/bin" -type f -name llama-server | head -n1)"
export LD_LIBRARY_PATH="$(dirname "$SERVER"):${LD_LIBRARY_PATH:-}"
exec "$SERVER" \
  --model "$ROOT/model/$MODEL" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 --n-gpu-layers 99 --parallel 1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --metrics --no-webui
