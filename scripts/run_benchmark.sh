#!/usr/bin/env bash
# ==============================================================================
# Parallel OCR Benchmark Launch Script
#
# Runs 5 benchmark paths in parallel across 4x A6000 GPUs:
#   GPU 0: DeepSeek-OCR v1
#   GPU 1: DeepSeek-OCR v2
#   GPU 2: vLLM Server A (port 8000) — QA engine
#   GPU 3: vLLM Server B (port 8001) — QA engine
#   CPU:   Baseline + Marker + MarkItDown
#
# Usage:
#   QA_MODEL_PATH=/path/to/Qwen3-VL-8B bash scripts/run_benchmark.sh
#
# Environment variables:
#   QA_MODEL_PATH  (required)  Path to the Qwen3-VL-8B model weights
#   QA_MODEL_NAME  (default: Qwen3-VL-8B)
#   MAX_SAMPLES    (default: 0 = all)
#   DATASETS       (default: "arxivqa slidevqa")
# ==============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
QA_MODEL_PATH="${QA_MODEL_PATH:?QA_MODEL_PATH must be set to the Qwen3-VL-8B model path}"
QA_MODEL_NAME="${QA_MODEL_NAME:-Qwen3-VL-8B}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DATASETS="${DATASETS:-arxivqa slidevqa}"

VLLM_PORT_A=8000
VLLM_PORT_B=8001
VLLM_GPU_A=2
VLLM_GPU_B=3
HEALTH_TIMEOUT=300  # seconds

RESULTS_BASE="data/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/benchmark/${TIMESTAMP}"

# ── Derived paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Parallel OCR Benchmark"
echo " $(date)"
echo "============================================================"
echo " QA_MODEL_PATH : $QA_MODEL_PATH"
echo " QA_MODEL_NAME : $QA_MODEL_NAME"
echo " MAX_SAMPLES   : $MAX_SAMPLES"
echo " DATASETS      : $DATASETS"
echo " LOG_DIR       : $LOG_DIR"
echo "============================================================"

# ── Cleanup handler ───────────────────────────────────────────────────────────
CHILD_PIDS=()
VLLM_PIDS=()

cleanup() {
    echo ""
    echo "[cleanup] Terminating child processes..."

    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    echo "[cleanup] Shutting down vLLM servers..."
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            # Give it a moment, then force kill
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done

    echo "[cleanup] Done."
}

trap cleanup EXIT INT TERM

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Start vLLM Servers
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Phase 1: Starting vLLM servers..."

# Server A — GPU 2, port 8000
echo "[vLLM-A] Starting on GPU $VLLM_GPU_A, port $VLLM_PORT_A ..."
CUDA_VISIBLE_DEVICES=$VLLM_GPU_A python -m vllm.entrypoints.openai.api_server \
    --model "$QA_MODEL_PATH" \
    --served-model-name "$QA_MODEL_NAME" \
    --port "$VLLM_PORT_A" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    > "$LOG_DIR/vllm_a.log" 2>&1 &
VLLM_PID_A=$!
VLLM_PIDS+=("$VLLM_PID_A")
echo "[vLLM-A] PID=$VLLM_PID_A"

# Server B — GPU 3, port 8001
echo "[vLLM-B] Starting on GPU $VLLM_GPU_B, port $VLLM_PORT_B ..."
CUDA_VISIBLE_DEVICES=$VLLM_GPU_B python -m vllm.entrypoints.openai.api_server \
    --model "$QA_MODEL_PATH" \
    --served-model-name "$QA_MODEL_NAME" \
    --port "$VLLM_PORT_B" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    > "$LOG_DIR/vllm_b.log" 2>&1 &
VLLM_PID_B=$!
VLLM_PIDS+=("$VLLM_PID_B")
echo "[vLLM-B] PID=$VLLM_PID_B"

# ── Health check ──────────────────────────────────────────────────────────────
wait_for_vllm() {
    local port=$1
    local name=$2
    local elapsed=0

    echo "[health] Waiting for $name (port $port) ..."
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "[health] $name is ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo "[health] ERROR: $name did not become ready within ${HEALTH_TIMEOUT}s"
    echo "[health] Check logs: $LOG_DIR/vllm_*.log"
    exit 1
}

wait_for_vllm $VLLM_PORT_A "Server-A"
wait_for_vllm $VLLM_PORT_B "Server-B"

echo ">>> Phase 1 complete: both vLLM servers are ready."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Run parallel benchmarks (per dataset)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Phase 2: Running benchmarks..."

# Define the 5 benchmark processes
# Format: name|ocr_model|gpu_env|vllm_port
BENCH_SPECS=(
    "baseline||cpu|$VLLM_PORT_A"
    "deepseek-ocr|deepseek-ocr|0|$VLLM_PORT_A"
    "deepseek-ocr-v2|deepseek-ocr-v2|1|$VLLM_PORT_B"
    "marker|marker|cpu|$VLLM_PORT_B"
    "markitdown|markitdown|cpu|$VLLM_PORT_A"
)

run_single_benchmark() {
    local name=$1
    local ocr_model=$2
    local gpu_env=$3
    local vllm_port=$4
    local dataset=$5
    local run_dir="${RESULTS_BASE}/run_${name}"
    local log_file="${LOG_DIR}/${dataset}_${name}.log"

    local cmd=(
        python -m src.cli.main benchmark run
        --dataset "$dataset"
        --results-dir "$run_dir"
        --qa-model "$QA_MODEL_NAME"
        --qa-api-base "http://localhost:${vllm_port}/v1"
        --qa-api-key EMPTY
    )

    if [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
        cmd+=(--max-samples "$MAX_SAMPLES")
    fi

    if [ "$name" = "baseline" ]; then
        cmd+=(--include-baseline --ocr-models "")
    else
        cmd+=(--no-baseline --ocr-models "$ocr_model")
    fi

    # Set GPU visibility
    if [ "$gpu_env" = "cpu" ]; then
        export CUDA_VISIBLE_DEVICES=""
    else
        export CUDA_VISIBLE_DEVICES="$gpu_env"
    fi

    echo "[bench] Starting $name on dataset=$dataset gpu=$gpu_env port=$vllm_port"
    "${cmd[@]}" > "$log_file" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "[bench] $name/$dataset completed successfully"
    else
        echo "[bench] WARNING: $name/$dataset exited with code $rc (see $log_file)"
    fi

    return $rc
}

for dataset in $DATASETS; do
    echo ""
    echo "──── Dataset: $dataset ────"

    CHILD_PIDS=()
    FAIL_COUNT=0

    for spec in "${BENCH_SPECS[@]}"; do
        IFS='|' read -r name ocr_model gpu_env vllm_port <<< "$spec"
        run_single_benchmark "$name" "$ocr_model" "$gpu_env" "$vllm_port" "$dataset" &
        CHILD_PIDS+=($!)
    done

    # Wait for all benchmark processes for this dataset
    for pid in "${CHILD_PIDS[@]}"; do
        if ! wait "$pid"; then
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done

    if [ $FAIL_COUNT -gt 0 ]; then
        echo "[bench] WARNING: $FAIL_COUNT process(es) failed for dataset=$dataset"
    fi

    echo "──── Dataset $dataset complete ────"
done

echo ""
echo ">>> Phase 2 complete."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Merge reports
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Phase 3: Merging reports..."

for dataset in $DATASETS; do
    echo "[merge] Merging results for dataset=$dataset"
    python scripts/merge_benchmark_reports.py \
        --results-base "$RESULTS_BASE" \
        --dataset "$dataset" \
        --output-dir "$RESULTS_BASE" \
        2>&1 | tee "$LOG_DIR/merge_${dataset}.log"
done

echo ""
echo ">>> Phase 3 complete."
echo ""
echo "============================================================"
echo " Benchmark finished at $(date)"
echo " Logs:    $LOG_DIR/"
echo " Results: $RESULTS_BASE/"
echo "============================================================"
