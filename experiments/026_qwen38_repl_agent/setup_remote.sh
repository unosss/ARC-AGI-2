#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/qwen38_repl_4090}"
TAG=b11669
MODEL_REV=5312b0f1c3705b45a48a18f051b4dc482627c17f
MODEL=Qwen3.8-27B-Q4_K_M.gguf
mkdir -p "$ROOT"/{bin,model,run}

base="https://github.com/ggml-org/llama.cpp/releases/download/$TAG"
for archive in \
  "llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" \
  "cudart-llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"; do
  test -f "$ROOT/$archive" || curl -fL --retry 8 --retry-all-errors -o "$ROOT/$archive.part" "$base/$archive"
  test -f "$ROOT/$archive" || mv "$ROOT/$archive.part" "$ROOT/$archive"
done
tar -xzf "$ROOT/llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" -C "$ROOT/bin"
tar -xzf "$ROOT/cudart-llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" -C "$ROOT/bin"

url="https://huggingface.co/6block/Qwen3.8-27B-GGUF/resolve/$MODEL_REV/$MODEL"
if test ! -f "$ROOT/model/$MODEL"; then
  curl -fL -C - --retry 12 --retry-all-errors -o "$ROOT/model/$MODEL.part" "$url"
  mv "$ROOT/model/$MODEL.part" "$ROOT/model/$MODEL"
fi
echo "038b8d86da2e388e4c3f5bafdf5a8aa4dcb630861a18d430b3a79c538c1a3beb  $ROOT/model/$MODEL" | sha256sum -c -
df -h /
