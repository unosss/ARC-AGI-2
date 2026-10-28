#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade --break-system-packages \
  'transformers>=4.57,<5' \
  'peft==0.21.2' \
  'accelerate>=1.10,<2' \
  'safetensors>=0.5,<1'

python - <<'PY'
from importlib.metadata import version

assert version("peft") == "0.21.2", version("peft")
major = int(version("transformers").split(".", 1)[0])
minor = int(version("transformers").split(".", 2)[1])
assert major == 4 and minor >= 57, version("transformers")
print("Experiment 034 runtime dependencies are ready")
PY
