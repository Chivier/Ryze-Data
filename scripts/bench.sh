#!/usr/bin/env bash
# ==============================================================================
# bench.sh - Orchestrator for OCR benchmark workflow
#
# Thin wrapper that delegates to individual scripts in scripts/benchmark/:
#   start_services.sh   — start/stop/status for vLLM QA servers
#   stop_services.sh    — stop vLLM QA servers
#   run_benchmark.sh    — run parallel OCR benchmark evaluation
#   generate_report.sh  — merge and display results
#
# Usage:
#   scripts/bench.sh <command> [options]
#
# Commands:
#   setup              Check prerequisites and install dependencies
#   start              Start vLLM QA servers
#   stop               Stop vLLM QA servers
#   status             Show server status and GPU usage
#   run [options]      Run benchmark (servers must be running)
#   report [dataset]   Merge and view benchmark reports
#   quick [options]    Start -> 5 samples -> report -> stop (all-in-one)
#   full [options]     Start -> all samples -> report -> stop (all-in-one)
#   single <model>     Debug a single OCR model
#   clean              Remove all benchmark results and logs
#   help               Show this help
#
# Configuration:
#   Copy .bench.env.example to .bench.env and set QA_MODEL_PATH.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/benchmark"

source "$BENCH_DIR/_env.sh"

# ==============================================================================
# Command: setup
# ==============================================================================
cmd_setup() {
    print_banner
    step "Checking prerequisites..."

    local issues=0

    # Python
    if command -v python3 &>/dev/null; then
        local pyver
        pyver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        local major minor
        major=$(echo "$pyver" | cut -d. -f1)
        minor=$(echo "$pyver" | cut -d. -f2)
        if (( major >= 3 && minor >= 10 )); then
            ok "Python $pyver"
        else
            err "Python >= 3.10 required (found $pyver)"
            issues=$((issues + 1))
        fi
    else
        err "python3 not found"
        issues=$((issues + 1))
    fi

    # uv
    if command -v uv &>/dev/null; then
        ok "uv $(uv --version 2>/dev/null | head -1)"
    else
        err "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        issues=$((issues + 1))
    fi

    # GPU
    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        ok "NVIDIA GPUs: $gpu_count"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
            | while read -r line; do dim "    $line"; done
    else
        warn "nvidia-smi not found (GPU support unavailable)"
    fi

    # curl
    if command -v curl &>/dev/null; then
        ok "curl available"
    else
        err "curl not found (required for health checks)"
        issues=$((issues + 1))
    fi

    # vLLM
    if python3 -c "import vllm" 2>/dev/null; then
        ok "vLLM importable"
    else
        warn "vLLM not importable (install: pip install vllm)"
    fi

    # Model path
    if [[ -n "$QA_MODEL_PATH" ]]; then
        if [[ -d "$QA_MODEL_PATH" ]]; then
            ok "QA_MODEL_PATH: $QA_MODEL_PATH"
        else
            err "QA_MODEL_PATH directory not found: $QA_MODEL_PATH"
            issues=$((issues + 1))
        fi
    else
        warn "QA_MODEL_PATH not set (set in .bench.env or via environment)"
    fi

    step "Installing dependencies..."
    if command -v uv &>/dev/null; then
        uv sync
        uv sync --extra benchmark
        ok "Base + benchmark dependencies installed"
        echo ""
        info "Optional extras (install as needed):"
        echo "  uv sync --extra deepseek-ocr    # DeepSeek v1/v2"
        echo "  uv sync --extra markitdown       # MarkItDown"
    fi

    # Config file
    if [[ ! -f "$BENCH_ENV" ]]; then
        step "Creating .bench.env..."
        if [[ -f "$PROJECT_ROOT/.bench.env.example" ]]; then
            cp "$PROJECT_ROOT/.bench.env.example" "$BENCH_ENV"
            ok "Created .bench.env from template"
            warn "Edit .bench.env to set QA_MODEL_PATH before running benchmarks"
        fi
    else
        ok ".bench.env already exists"
    fi

    echo ""
    if [[ $issues -gt 0 ]]; then
        err "$issues issue(s) found. Fix them before running benchmarks."
        exit 1
    fi

    ok "All prerequisites satisfied!"
    echo ""
    echo "  Next steps:"
    echo "    1. Edit .bench.env and set QA_MODEL_PATH"
    echo "    2. scripts/bench.sh quick              # quick 5-sample smoke test"
    echo "    3. scripts/bench.sh full               # full benchmark"
    echo ""
    echo "  Or step-by-step:"
    echo "    scripts/benchmark/start_services.sh    # start vLLM servers"
    echo "    scripts/benchmark/run_benchmark.sh     # run evaluation"
    echo "    scripts/benchmark/generate_report.sh   # merge reports"
    echo "    scripts/benchmark/stop_services.sh     # stop servers"
    echo ""
}

# ==============================================================================
# Command: quick  (start -> 5 samples -> report -> stop)
# ==============================================================================
cmd_quick() {
    print_banner
    echo -e "  ${CYAN}Quick benchmark: 5 samples per dataset${NC}"
    echo ""
    print_config

    local auto_stop=false

    # Start servers if not running
    if ! check_servers_healthy 2>/dev/null; then
        "$BENCH_DIR/start_services.sh"
        auto_stop=true
    else
        ok "vLLM servers already running, reusing."
    fi

    # Run with 5 samples
    "$BENCH_DIR/run_benchmark.sh" --max-samples 5 "$@"

    # Report
    "$BENCH_DIR/generate_report.sh"

    # Stop if we started them
    if [[ "$auto_stop" == "true" ]]; then
        "$BENCH_DIR/stop_services.sh"
    fi
}

# ==============================================================================
# Command: full  (start -> all samples -> report -> stop)
# ==============================================================================
cmd_full() {
    print_banner
    echo -e "  ${CYAN}Full benchmark: all samples${NC}"
    echo ""
    print_config

    local auto_stop=false

    # Start servers if not running
    if ! check_servers_healthy 2>/dev/null; then
        "$BENCH_DIR/start_services.sh"
        auto_stop=true
    else
        ok "vLLM servers already running, reusing."
    fi

    # Run all samples
    "$BENCH_DIR/run_benchmark.sh" "$@"

    # Report
    "$BENCH_DIR/generate_report.sh"

    # Stop if we started them
    if [[ "$auto_stop" == "true" ]]; then
        "$BENCH_DIR/stop_services.sh"
    fi
}

# ==============================================================================
# Command: single <model>  (run one model for debugging)
# ==============================================================================
cmd_single() {
    local model="${1:-}"
    if [[ -z "$model" ]]; then
        err "Usage: scripts/bench.sh single <model> [options]"
        echo ""
        echo "  Available: baseline, deepseek-ocr, deepseek-ocr-v2, marker, markitdown"
        echo ""
        echo "  Options:"
        echo "    --datasets <name>    Dataset (default: arxivqa)"
        echo "    --max-samples <n>    Max samples (default: 10)"
        echo ""
        echo "  Servers must be running (scripts/bench.sh start)."
        exit 1
    fi
    shift

    # Default 10 samples for debug mode
    local has_max_samples=false
    for arg in "$@"; do
        [[ "$arg" == "--max-samples" ]] && has_max_samples=true
    done

    if [[ "$has_max_samples" == "false" ]]; then
        "$BENCH_DIR/run_benchmark.sh" --models "$model" --max-samples 10 "$@"
    else
        "$BENCH_DIR/run_benchmark.sh" --models "$model" "$@"
    fi
}

# ==============================================================================
# Command: clean
# ==============================================================================
cmd_clean() {
    echo ""
    warn "This will remove all benchmark results and logs:"
    echo "  - $RESULTS_BASE/"
    echo "  - $LOG_BASE/"
    echo ""
    read -rp "  Are you sure? [y/N] " confirm

    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        "$BENCH_DIR/stop_services.sh" 2>/dev/null || true
        rm -rf "$RESULTS_BASE"
        rm -rf "$LOG_BASE"
        ok "Cleaned benchmark results and logs."
    else
        info "Cancelled."
    fi
}

# ==============================================================================
# Command: help
# ==============================================================================
cmd_help() {
    print_banner
    cat <<'HELP'
  Usage: scripts/bench.sh <command> [options]

  Orchestrator Commands (convenience wrappers):
    setup              Check prerequisites and install dependencies
    quick [options]    All-in-one: start -> 5 samples -> report -> stop
    full [options]     All-in-one: start -> all samples -> report -> stop
    single <model>     Debug one OCR model (10 samples, servers must be running)
    clean              Remove all benchmark results and logs

  Delegated Commands (call sub-scripts directly):
    start              → scripts/benchmark/start_services.sh
    stop               → scripts/benchmark/stop_services.sh
    status             → scripts/benchmark/start_services.sh --status
    run [options]      → scripts/benchmark/run_benchmark.sh
    report [dataset]   → scripts/benchmark/generate_report.sh

  Run options (for run/quick/full):
    --datasets <list>  Space-separated datasets (default: "arxivqa slidevqa")
    --max-samples <n>  Limit samples per dataset (default: 0=all)
    --models <list>    Comma-separated models (default: all 5)

  Configuration:
    Copy .bench.env.example to .bench.env and set QA_MODEL_PATH.

  Workflow Examples:

    # One-command workflows
    scripts/bench.sh setup                         # first-time setup
    scripts/bench.sh quick                         # quick smoke test
    scripts/bench.sh full                          # full benchmark
    scripts/bench.sh full --datasets arxivqa       # one dataset only

    # Step-by-step (use sub-scripts directly)
    scripts/benchmark/start_services.sh            # 1. start vLLM
    scripts/benchmark/run_benchmark.sh             # 2. run evaluation
    scripts/benchmark/generate_report.sh           # 3. merge reports
    scripts/benchmark/stop_services.sh             # 4. stop vLLM

    # Debug a single model
    scripts/bench.sh start
    scripts/bench.sh single deepseek-ocr
    scripts/bench.sh stop
HELP
}

# ==============================================================================
# Trap for Ctrl+C in quick/full
# ==============================================================================
cleanup_trap() {
    echo ""
    warn "Interrupted! Cleaning up..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    "$BENCH_DIR/stop_services.sh" 2>/dev/null || true
    exit 130
}

# ==============================================================================
# Main dispatch
# ==============================================================================
main() {
    local command="${1:-help}"
    shift 2>/dev/null || true

    case "$command" in
        quick|full)
            trap cleanup_trap INT TERM
            ;;
    esac

    case "$command" in
        # Orchestrator commands
        setup)   cmd_setup "$@" ;;
        quick)   cmd_quick "$@" ;;
        full)    cmd_full "$@" ;;
        single)  cmd_single "$@" ;;
        clean)   cmd_clean "$@" ;;

        # Delegated commands
        start)   "$BENCH_DIR/start_services.sh" "$@" ;;
        stop)    "$BENCH_DIR/stop_services.sh" "$@" ;;
        status)  "$BENCH_DIR/start_services.sh" --status "$@" ;;
        run)     "$BENCH_DIR/run_benchmark.sh" "$@" ;;
        report)  "$BENCH_DIR/generate_report.sh" "$@" ;;

        help|-h|--help)  cmd_help ;;
        *)
            err "Unknown command: $command"
            echo "  Run 'scripts/bench.sh help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
