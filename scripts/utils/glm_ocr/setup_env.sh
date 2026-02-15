#!/usr/bin/env bash
# Setup isolated venv for GLM-OCR (vLLM backend).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up GLM-OCR (vLLM) environment ==="

if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it first: https://docs.astral.sh/uv/"
    exit 1
fi

uv venv .venv --python 3.12
# GLM-OCR model support requires vLLM nightly (>= 0.16.0dev).
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --python .venv/bin/python
# Transformers from git source needed for glm_ocr model type recognition.
uv pip install git+https://github.com/huggingface/transformers.git --python .venv/bin/python
uv pip install -r requirements.txt --python .venv/bin/python

echo "=== Setup complete ==="
echo "Run OCR with: PYTHONPATH=. .venv/bin/python scripts/test_ocr_real.py glm-ocr --gpu 3"
