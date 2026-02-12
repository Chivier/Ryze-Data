#!/usr/bin/env bash
# Setup isolated venv for DeepSeek-OCR v2.
# Aligns with the official DeepSeek-OCR install flow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up DeepSeek-OCR v2 environment ==="
echo "Official baseline: CUDA 11.8 + PyTorch 2.6.0"

if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it first: https://docs.astral.sh/uv/"
    exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.12.9}"
PYTHON_BIN=".venv/bin/python"
VLLM_WHEEL="${VLLM_WHEEL:-vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"

uv venv .venv --python "${PYTHON_VERSION}"

# Official torch/cu118 baseline.
uv pip install --python "${PYTHON_BIN}" \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install model runtime dependencies used by the official HF path.
uv pip install --python "${PYTHON_BIN}" \
    transformers==4.46.3 tokenizers==0.20.3

# vLLM is optional in this repo (transformers backend is default), but we
# install it by default to stay close to official instructions.
if [[ "${INSTALL_VLLM}" == "1" ]]; then
    if [[ -f "${VLLM_WHEEL}" ]]; then
        echo "Installing vLLM from official wheel: ${VLLM_WHEEL}"
        uv pip install --python "${PYTHON_BIN}" "${VLLM_WHEEL}"
    else
        echo "Official wheel not found (${VLLM_WHEEL}); using upstream nightly vLLM."
        uv pip install --python "${PYTHON_BIN}" \
            -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
    fi
else
    echo "Skipping vLLM installation (INSTALL_VLLM=${INSTALL_VLLM})."
fi

# Project dependencies (dataset loading and IO helpers).
uv pip install --python "${PYTHON_BIN}" -r requirements.txt

# Official acceleration dependency.
uv pip install --python "${PYTHON_BIN}" flash-attn==2.7.3 --no-build-isolation

echo "=== Setup complete ==="
echo "Run OCR with: .venv/bin/python run_ocr.py --dataset arxivqa --gpu 0"
echo "Optional backend flag: --backend transformers|auto|vllm"
echo "Note: dependency conflict warnings between vLLM and transformers may appear; upstream README indicates this is expected."
