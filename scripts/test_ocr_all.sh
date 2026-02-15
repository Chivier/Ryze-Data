#!/usr/bin/env bash
# Master orchestration script: run real-file OCR tests for all models.
#
# Usage:
#   bash scripts/test_ocr_all.sh [GPU_ID]
#
# Default GPU is 3 (assumes GPU 0 is occupied).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GPU="${1:-3}"
PASS=0
FAIL=0
SKIP=0

run_test() {
    local model="$1"
    local venv_dir="$2"
    local gpu_arg="$3"  # empty string or "--gpu N"

    echo ""
    echo "################################################################"
    echo "  Testing: $model"
    echo "  Venv:    $venv_dir"
    echo "################################################################"

    local python_bin="$venv_dir/.venv/bin/python"

    if [[ ! -x "$python_bin" ]]; then
        echo "SKIP: venv not found at $python_bin"
        SKIP=$((SKIP + 1))
        return
    fi

    # shellcheck disable=SC2086
    if PYTHONPATH=. "$python_bin" scripts/test_ocr_real.py "$model" $gpu_arg; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================================"
echo "  OCR Real-File Test Suite"
echo "  GPU: $GPU"
echo "  Started: $(date)"
echo "============================================================"

# 1. markitdown (CPU only, fastest)
run_test "markitdown" "scripts/utils/markitdown" ""

# 2. marker (GPU)
run_test "marker" "scripts/utils/marker" "--gpu $GPU"

# 3. paddleocr (GPU)
run_test "paddleocr" "scripts/utils/paddleocr" "--gpu $GPU"

# 4. deepseek-ocr (GPU, model load ~60s)
run_test "deepseek-ocr" "scripts/utils/deepseek_ocr_v1" "--gpu $GPU"

# 5. glm-ocr (GPU, model load ~60s)
run_test "glm-ocr" "scripts/utils/glm_ocr" "--gpu $GPU"

echo ""
echo "============================================================"
echo "  Final Summary: $PASS PASS, $FAIL FAIL, $SKIP SKIP"
echo "  Finished: $(date)"
echo "============================================================"

# Exit non-zero if any model failed.
[[ "$FAIL" -eq 0 ]]
