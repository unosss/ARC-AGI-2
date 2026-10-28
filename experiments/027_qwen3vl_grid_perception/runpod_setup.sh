#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade \
  "transformers>=4.57.0,<5" \
  "accelerate>=1.2" \
  "bitsandbytes>=0.49.2" \
  "qwen-vl-utils==0.0.14" \
  "pillow>=10" \
  "pytest>=8"

python - <<'PY'
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
