#!/usr/bin/env bash
# Setup isolated venv for MarkItDown OCR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up MarkItDown OCR environment ==="

if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it first: https://docs.astral.sh/uv/"
    exit 1
fi

uv venv .venv --python 3.11
uv pip install -r requirements.txt --python .venv/bin/python

echo "=== Setup complete ==="
echo "Run OCR with: .venv/bin/python run_ocr.py --dataset arxivqa"
