#!/usr/bin/env bash
# ==============================================================================
# _env.sh - Shared environment for benchmark scripts
#
# Provides: colors, logging helpers, config loading, PID management,
#           health-check utilities, and common path variables.
#
# Usage (source from sibling scripts):
#   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#   source "$SCRIPT_DIR/_env.sh"
# ==============================================================================

# Guard against double-sourcing
if [[ -n "${_BENCH_ENV_LOADED:-}" ]]; then return 0 2>/dev/null || true; fi
_BENCH_ENV_LOADED=1

# ── Project root ─────────────────────────────────────────────────────────────
# Resolve from this file's location: scripts/benchmark/_env.sh -> project root
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$_ENV_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Logging helpers ──────────────────────────────────────────────────────────
info()  { echo -e "${BLUE}[info]${NC} $*"; }
ok()    { echo -e "${GREEN}[ok]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}>>> $*${NC}"; }
dim()   { echo -e "${DIM}$*${NC}"; }

# ── Load user config (.bench.env) ───────────────────────────────────────────
BENCH_ENV="$PROJECT_ROOT/.bench.env"
if [[ -f "$BENCH_ENV" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$BENCH_ENV"
    set +a
    dim "Loaded config from .bench.env"
fi

# ── Configurable variables (env vars override .bench.env) ────────────────────
QA_MODEL_PATH="${QA_MODEL_PATH:-}"
QA_MODEL_NAME="${QA_MODEL_NAME:-Qwen3-VL-8B}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DATASETS="${DATASETS:-arxivqa slidevqa}"

VLLM_PORT_A="${VLLM_PORT_A:-8000}"
VLLM_PORT_B="${VLLM_PORT_B:-8001}"
VLLM_GPU_A="${VLLM_GPU_A:-2}"
VLLM_GPU_B="${VLLM_GPU_B:-3}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"

RESULTS_BASE="${RESULTS_BASE:-data/benchmark_results}"
SHARED_DATA_DIR="${SHARED_DATA_DIR:-data/benchmark_data}"

# ── Internal paths ───────────────────────────────────────────────────────────
PID_DIR="$PROJECT_ROOT/logs/benchmark/.pids"
LOG_BASE="$PROJECT_ROOT/logs/benchmark"

# ── Validation helpers ───────────────────────────────────────────────────────

require_model_path() {
    if [[ -z "$QA_MODEL_PATH" ]]; then
        err "QA_MODEL_PATH is not set."
        echo ""
        echo "  Set it in one of these ways:"
        echo "    1. Create .bench.env  (copy from .bench.env.example)"
        echo "    2. Export:  export QA_MODEL_PATH=/path/to/Qwen3-VL-8B"
        echo "    3. Inline:  QA_MODEL_PATH=/path/to/model <script>"
        echo ""
        exit 1
    fi
    if [[ ! -d "$QA_MODEL_PATH" ]]; then
        err "QA_MODEL_PATH directory does not exist: $QA_MODEL_PATH"
        exit 1
    fi
}

# ── PID management ───────────────────────────────────────────────────────────

# get_vllm_pid <name>  — echoes PID if alive, returns 1 if not running
get_vllm_pid() {
    local name=$1
    local pid_file="$PID_DIR/vllm_${name}.pid"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
        rm -f "$pid_file"   # stale
    fi
    return 1
}

save_vllm_pid() {
    local name=$1 pid=$2
    mkdir -p "$PID_DIR"
    echo "$pid" > "$PID_DIR/vllm_${name}.pid"
}

remove_vllm_pid() {
    local name=$1
    rm -f "$PID_DIR/vllm_${name}.pid"
}

# ── Network helpers ──────────────────────────────────────────────────────────

is_port_in_use() {
    local port=$1
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    elif command -v lsof &>/dev/null; then
        lsof -i :"$port" &>/dev/null && return 0
    fi
    curl -sf "http://localhost:${port}/v1/models" &>/dev/null && return 0
    return 1
}

wait_for_vllm() {
    local port=$1 name=$2
    local elapsed=0

    info "Waiting for $name (port $port) ..."
    while [[ $elapsed -lt "$HEALTH_TIMEOUT" ]]; do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            ok "$name is ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if (( elapsed % 30 == 0 )); then
            dim "  Still waiting... ${elapsed}s / ${HEALTH_TIMEOUT}s"
        fi
    done

    err "$name did not become ready within ${HEALTH_TIMEOUT}s"
    err "Check logs: $LOG_BASE/latest/vllm_${name}.log"
    return 1
}

check_servers_healthy() {
    local ok=true
    if ! curl -sf "http://localhost:${VLLM_PORT_A}/v1/models" > /dev/null 2>&1; then
        err "Server-A (port $VLLM_PORT_A) is not responding."
        ok=false
    fi
    if ! curl -sf "http://localhost:${VLLM_PORT_B}/v1/models" > /dev/null 2>&1; then
        err "Server-B (port $VLLM_PORT_B) is not responding."
        ok=false
    fi
    if [[ "$ok" == "false" ]]; then
        return 1
    fi
    return 0
}

# ── Display helpers ──────────────────────────────────────────────────────────

print_banner() {
    echo -e "${CYAN}"
    echo "  =============================================="
    echo "   Ryze-Data OCR Benchmark"
    echo "  =============================================="
    echo -e "${NC}"
}

print_config() {
    echo -e "  ${DIM}QA Model    :${NC} ${QA_MODEL_PATH:-${RED}(not set)${NC}}"
    echo -e "  ${DIM}Model Name  :${NC} $QA_MODEL_NAME"
    echo -e "  ${DIM}Max Samples :${NC} ${MAX_SAMPLES} ${DIM}(0=all)${NC}"
    echo -e "  ${DIM}Datasets    :${NC} $DATASETS"
    echo -e "  ${DIM}vLLM Ports  :${NC} $VLLM_PORT_A (GPU $VLLM_GPU_A), $VLLM_PORT_B (GPU $VLLM_GPU_B)"
    echo -e "  ${DIM}Results     :${NC} $RESULTS_BASE"
    echo ""
}

# ── Log directory ────────────────────────────────────────────────────────────

# ensure_log_dir — creates (or reuses) a timestamped log dir; sets LOG_DIR
ensure_log_dir() {
    if [[ -L "$LOG_BASE/latest" ]]; then
        LOG_DIR="$LOG_BASE/$(readlink "$LOG_BASE/latest")"
    else
        local ts
        ts=$(date +%Y%m%d_%H%M%S)
        LOG_DIR="$LOG_BASE/$ts"
        mkdir -p "$LOG_DIR"
        rm -f "$LOG_BASE/latest"
        ln -sf "$ts" "$LOG_BASE/latest"
    fi
    mkdir -p "$LOG_DIR"
}
