#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade \
  "transformers==4.57.6" \
  "accelerate==1.15.0" \
  "peft==0.17.1" \
  "bitsandbytes==0.48.2" \
  "datasets==4.4.1" \
  "pyarrow>=18,<23" \
  "pillow>=10" \
  "numpy>=1.26,<3"

