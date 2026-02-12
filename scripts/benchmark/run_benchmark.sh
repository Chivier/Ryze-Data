#!/usr/bin/env bash
# ==============================================================================
# run_benchmark.sh - Run parallel OCR benchmark evaluation
#
# Runs 5 benchmark paths in parallel for each dataset:
#   baseline        (CPU  + Server A)
#   deepseek-ocr    (GPU0 + Server A)
#   deepseek-ocr-v2 (GPU1 + Server B)
#   marker          (CPU  + Server B)
#   markitdown      (CPU  + Server A)
#
# Prerequisite: vLLM servers must be running (start_services.sh).
#
# Usage:
#   scripts/benchmark/run_benchmark.sh [options]
#
# Options:
#   --datasets <list>    Space-separated datasets (default: from config)
#   --max-samples <n>    Max samples per dataset (default: 0 = all)
#   --models <list>      Comma-separated models to run (default: all 5)
#   -h, --help           Show help
#
# Examples:
#   scripts/benchmark/run_benchmark.sh
#   scripts/benchmark/run_benchmark.sh --max-samples 5
#   scripts/benchmark/run_benchmark.sh --datasets arxivqa --max-samples 20
#   scripts/benchmark/run_benchmark.sh --models deepseek-ocr,baseline
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# ── Defaults ─────────────────────────────────────────────────────────────────
datasets="${DATASETS}"
max_samples="${MAX_SAMPLES}"
selected_models=""

# ── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: scripts/benchmark/run_benchmark.sh [options]

Run parallel OCR benchmark evaluation across 5 model paths.
vLLM servers must be running (see start_services.sh).

Options:
  --datasets <list>    Space-separated dataset names (default: "$DATASETS")
  --max-samples <n>    Max samples per dataset, 0 = all (default: $MAX_SAMPLES)
  --models <list>      Comma-separated subset of models to run
                       Available: baseline, deepseek-ocr, deepseek-ocr-v2, marker, markitdown
  -h, --help           Show this help message

Examples:
  scripts/benchmark/run_benchmark.sh                                # full run
  scripts/benchmark/run_benchmark.sh --max-samples 5                # quick 5-sample test
  scripts/benchmark/run_benchmark.sh --datasets arxivqa             # one dataset only
  scripts/benchmark/run_benchmark.sh --models deepseek-ocr,marker   # two models only
EOF
}

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets|-d)    datasets="$2"; shift 2 ;;
        --max-samples)    max_samples="$2"; shift 2 ;;
        --models)         selected_models="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)                err "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── Pre-flight checks ───────────────────────────────────────────────────────
if ! check_servers_healthy; then
    err "vLLM servers are not running."
    echo "  Start them first:  scripts/benchmark/start_services.sh"
    exit 1
fi

ensure_log_dir
mkdir -p "$RESULTS_BASE" "$SHARED_DATA_DIR"

# ── Define benchmark specs ───────────────────────────────────────────────────
# Format: name|ocr_model|gpu_env|vllm_port
ALL_SPECS=(
    "baseline||cpu|$VLLM_PORT_A"
    "deepseek-ocr|deepseek-ocr|0|$VLLM_PORT_A"
    "deepseek-ocr-v2|deepseek-ocr-v2|1|$VLLM_PORT_B"
    "marker|marker|cpu|$VLLM_PORT_B"
    "markitdown|markitdown|cpu|$VLLM_PORT_A"
)

# Filter specs if --models was given
BENCH_SPECS=()
if [[ -n "$selected_models" ]]; then
    IFS=',' read -ra model_filter <<< "$selected_models"
    for spec in "${ALL_SPECS[@]}"; do
        spec_name="${spec%%|*}"
        for m in "${model_filter[@]}"; do
            if [[ "$spec_name" == "$m" ]]; then
                BENCH_SPECS+=("$spec")
                break
            fi
        done
    done
    if [[ ${#BENCH_SPECS[@]} -eq 0 ]]; then
        err "No matching models found for: $selected_models"
        echo "  Available: baseline, deepseek-ocr, deepseek-ocr-v2, marker, markitdown"
        exit 1
    fi
else
    BENCH_SPECS=("${ALL_SPECS[@]}")
fi

# ── Display config ───────────────────────────────────────────────────────────
print_banner
echo -e "  ${DIM}Datasets    :${NC} $datasets"
echo -e "  ${DIM}Max Samples :${NC} $max_samples ${DIM}(0=all)${NC}"
echo -e "  ${DIM}Models      :${NC} ${#BENCH_SPECS[@]} paths"
echo -e "  ${DIM}Results     :${NC} $RESULTS_BASE"
echo -e "  ${DIM}Logs        :${NC} $LOG_DIR"
echo ""

# ── Trap for clean shutdown ──────────────────────────────────────────────────
CHILD_PIDS=()
cleanup() {
    echo ""
    warn "Interrupted! Terminating benchmark processes..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 130
}
trap cleanup INT TERM

# ── Run benchmarks ───────────────────────────────────────────────────────────
overall_fail=0

for dataset in $datasets; do
    step "Dataset: $dataset"

    CHILD_PIDS=()
    fail_count=0

    for spec in "${BENCH_SPECS[@]}"; do
        IFS='|' read -r name ocr_model gpu_env vllm_port <<< "$spec"
        run_dir="${RESULTS_BASE}/run_${name}"
        log_file="${LOG_DIR}/${dataset}_${name}.log"

        cmd=(
            python -m src.cli.main benchmark run
            --dataset "$dataset"
            --results-dir "$run_dir"
            --data-dir "$SHARED_DATA_DIR"
            --qa-model "$QA_MODEL_NAME"
            --qa-api-base "http://localhost:${vllm_port}/v1"
            --qa-api-key EMPTY
        )

        if [[ "$max_samples" -gt 0 ]] 2>/dev/null; then
            cmd+=(--max-samples "$max_samples")
        fi

        if [[ "$name" == "baseline" ]]; then
            cmd+=(--include-baseline --ocr-models "")
        else
            cmd+=(--no-baseline --ocr-models "$ocr_model")
        fi

        cuda_env=""
        if [[ "$gpu_env" != "cpu" ]]; then
            cuda_env="$gpu_env"
        fi

        info "Starting ${BOLD}${name}${NC} (GPU=${gpu_env}, port=${vllm_port})"

        (
            export CUDA_VISIBLE_DEVICES="$cuda_env"
            "${cmd[@]}" > "$log_file" 2>&1
            rc=$?
            if [[ $rc -eq 0 ]]; then
                echo -e "${GREEN}[done]${NC} ${name}/${dataset} completed"
            else
                echo -e "${RED}[fail]${NC} ${name}/${dataset} exited with code $rc (see $log_file)" >&2
            fi
            exit $rc
        ) &
        CHILD_PIDS+=($!)
    done

    # Wait for all processes in this dataset
    for pid in "${CHILD_PIDS[@]}"; do
        if ! wait "$pid" 2>/dev/null; then
            fail_count=$((fail_count + 1))
        fi
    done

    if [[ $fail_count -gt 0 ]]; then
        warn "$fail_count process(es) failed for $dataset"
        overall_fail=$((overall_fail + fail_count))
    else
        ok "All benchmarks completed for $dataset"
    fi
done

echo ""
echo "  =============================================="
echo "  Results: $RESULTS_BASE/"
echo "  Logs:    $LOG_DIR/"
echo "  =============================================="

if [[ $overall_fail -gt 0 ]]; then
    warn "Some benchmarks failed. Check logs for details."
    echo ""
    echo "  Next: review logs, fix issues, re-run."
    echo "        Already-completed samples are cached and will be skipped."
    exit 1
else
    ok "All benchmarks completed successfully!"
    echo ""
    echo "  Next: scripts/benchmark/generate_report.sh"
    echo ""
fi
