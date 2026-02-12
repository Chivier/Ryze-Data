#!/usr/bin/env bash
# ==============================================================================
# run_ocr_benchmark.sh - Shell wrapper for OCR-precompute benchmark runner
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage: scripts/benchmark/run_ocr_benchmark.sh [runner options]

Wrapper around:
  uv run python scripts/benchmark/run_ocr_benchmark.py

Examples:
  scripts/benchmark/run_ocr_benchmark.sh --dataset arxivqa
  scripts/benchmark/run_ocr_benchmark.sh --dataset slidevqa --max-samples 50
EOF
    exit 0
fi

uv run python scripts/benchmark/run_ocr_benchmark.py "$@"
