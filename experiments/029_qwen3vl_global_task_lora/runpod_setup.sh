#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade --break-system-packages \
  "transformers>=4.57.0,<5" \
  "accelerate>=1.2" \
  "bitsandbytes>=0.49.2" \
  "peft==0.21.2" \
  "qwen-vl-utils==0.0.14" \
  "pillow>=10" \
  "pytest>=8"

python - <<'PY'
import accelerate
import bitsandbytes
import peft
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("peft", peft.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
