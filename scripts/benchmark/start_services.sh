#!/usr/bin/env bash
# ==============================================================================
# start_services.sh - Start vLLM QA servers
#
# Launches two vLLM OpenAI-compatible servers on separate GPUs, waits for
# health checks, and stores PID files for later management.
#
# Usage:
#   scripts/benchmark/start_services.sh
#
# Prerequisites:
#   - QA_MODEL_PATH must be set (via .bench.env or environment)
#   - vLLM must be installed
#   - GPUs must be available
#
# Also supports: --status, --help
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# ── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: scripts/benchmark/start_services.sh [--status] [--help]

Start vLLM QA servers for benchmark evaluation.

Options:
  --status    Show current server status and exit
  -h, --help  Show this help message

Configuration (via .bench.env or environment):
  QA_MODEL_PATH   (required) Path to Qwen3-VL-8B weights
  VLLM_GPU_A      GPU index for Server A (default: 2)
  VLLM_GPU_B      GPU index for Server B (default: 3)
  VLLM_PORT_A     Port for Server A (default: 8000)
  VLLM_PORT_B     Port for Server B (default: 8001)
  HEALTH_TIMEOUT  Max seconds to wait for startup (default: 300)

Examples:
  scripts/benchmark/start_services.sh            # start both servers
  scripts/benchmark/start_services.sh --status   # check status only
EOF
}

# ── Show server status ───────────────────────────────────────────────────────
show_status() {
    echo ""
    echo -e "${BOLD}  vLLM Server Status${NC}"
    echo "  ─────────────────────────────────────────"

    for name in a b; do
        local port_var="VLLM_PORT_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
        local port="${!port_var}"
        local pid

        if pid=$(get_vllm_pid "$name"); then
            if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
                echo -e "  Server-${name}: ${GREEN}RUNNING${NC} (PID $pid, port $port) - ${GREEN}healthy${NC}"
            else
                echo -e "  Server-${name}: ${YELLOW}RUNNING${NC} (PID $pid, port $port) - ${YELLOW}not responding${NC}"
            fi
        else
            echo -e "  Server-${name}: ${DIM}STOPPED${NC} (port $port)"
        fi
    done

    echo ""

    if command -v nvidia-smi &>/dev/null; then
        echo -e "${BOLD}  GPU Usage${NC}"
        echo "  ─────────────────────────────────────────"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null | while read -r line; do
            echo "  $line"
        done
        echo ""
    fi
}

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --status)  show_status; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *)         err "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── Main: start servers ─────────────────────────────────────────────────────
require_model_path
print_banner
print_config

# Check if already running
already_running=0
if get_vllm_pid a &>/dev/null; then
    warn "vLLM Server-A already running (PID $(get_vllm_pid a))"
    already_running=$((already_running + 1))
fi
if get_vllm_pid b &>/dev/null; then
    warn "vLLM Server-B already running (PID $(get_vllm_pid b))"
    already_running=$((already_running + 1))
fi

if [[ $already_running -eq 2 ]]; then
    ok "Both servers already running."
    echo ""
    echo "  To restart, run:"
    echo "    scripts/benchmark/stop_services.sh"
    echo "    scripts/benchmark/start_services.sh"
    echo ""
    exit 0
elif [[ $already_running -eq 1 ]]; then
    warn "One server already running. Stopping both first..."
    "$SCRIPT_DIR/stop_services.sh"
fi

# Check ports
for port in "$VLLM_PORT_A" "$VLLM_PORT_B"; do
    if is_port_in_use "$port"; then
        err "Port $port is already in use."
        err "Stop the existing process or change VLLM_PORT_A/B in .bench.env."
        exit 1
    fi
done

# Create log directory
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$LOG_BASE/$timestamp"
mkdir -p "$LOG_DIR" "$PID_DIR"

rm -f "$LOG_BASE/latest"
ln -sf "$timestamp" "$LOG_BASE/latest"

step "Starting vLLM servers..."

# ── Server A ─────────────────────────────────────────────────────────────────
info "Server-A: GPU $VLLM_GPU_A, port $VLLM_PORT_A"
CUDA_VISIBLE_DEVICES=$VLLM_GPU_A python -m vllm.entrypoints.openai.api_server \
    --model "$QA_MODEL_PATH" \
    --served-model-name "$QA_MODEL_NAME" \
    --port "$VLLM_PORT_A" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    > "$LOG_DIR/vllm_a.log" 2>&1 &
pid_a=$!
save_vllm_pid a "$pid_a"
dim "  PID=$pid_a, log=$LOG_DIR/vllm_a.log"

# ── Server B ─────────────────────────────────────────────────────────────────
info "Server-B: GPU $VLLM_GPU_B, port $VLLM_PORT_B"
CUDA_VISIBLE_DEVICES=$VLLM_GPU_B python -m vllm.entrypoints.openai.api_server \
    --model "$QA_MODEL_PATH" \
    --served-model-name "$QA_MODEL_NAME" \
    --port "$VLLM_PORT_B" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    > "$LOG_DIR/vllm_b.log" 2>&1 &
pid_b=$!
save_vllm_pid b "$pid_b"
dim "  PID=$pid_b, log=$LOG_DIR/vllm_b.log"

# ── Health checks ────────────────────────────────────────────────────────────
step "Waiting for servers to be ready (timeout: ${HEALTH_TIMEOUT}s)..."

if ! wait_for_vllm "$VLLM_PORT_A" "Server-A"; then
    err "Server-A failed to start. Cleaning up..."
    "$SCRIPT_DIR/stop_services.sh"
    exit 1
fi
if ! wait_for_vllm "$VLLM_PORT_B" "Server-B"; then
    err "Server-B failed to start. Cleaning up..."
    "$SCRIPT_DIR/stop_services.sh"
    exit 1
fi

echo ""
ok "Both vLLM servers are ready!"
echo -e "  Server-A: ${GREEN}http://localhost:${VLLM_PORT_A}/v1${NC}"
echo -e "  Server-B: ${GREEN}http://localhost:${VLLM_PORT_B}/v1${NC}"
echo -e "  Logs:     $LOG_DIR/"
echo ""
echo "  Next: scripts/benchmark/run_benchmark.sh"
echo ""
