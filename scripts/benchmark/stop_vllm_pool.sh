#!/usr/bin/env bash
# ==============================================================================
# stop_vllm_pool.sh - Stop vLLM processes started by start_vllm_pool.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

usage() {
    cat <<EOF
Usage: scripts/benchmark/stop_vllm_pool.sh [--status]

Stop vLLM pool processes tracked in:
  logs/benchmark/.pids/vllm_pool_<port>.pid
EOF
}

show_status() {
    echo ""
    echo -e "${BOLD}  vLLM Pool Status${NC}"
    echo "  ─────────────────────────────────────────"
    local found=0

    shopt -s nullglob
    for pid_file in "$PID_DIR"/vllm_pool_*.pid; do
        found=1
        port="${pid_file##*_}"
        port="${port%.pid}"
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
                echo -e "  port ${port}: ${GREEN}RUNNING${NC} (PID ${pid})"
            else
                echo -e "  port ${port}: ${YELLOW}RUNNING${NC} (PID ${pid}) - unhealthy"
            fi
        else
            echo -e "  port ${port}: ${RED}STALE${NC} (PID ${pid} not alive)"
        fi
    done
    shopt -u nullglob

    if [[ "$found" -eq 0 ]]; then
        echo -e "  ${DIM}No pool PID files found${NC}"
    fi
    echo ""
}

if [[ "${1:-}" == "--status" ]]; then
    show_status
    exit 0
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

step "Stopping vLLM pool..."

stopped=0
stale=0

shopt -s nullglob
for pid_file in "$PID_DIR"/vllm_pool_*.pid; do
    port="${pid_file##*_}"
    port="${port%.pid}"
    pid="$(cat "$pid_file")"

    if kill -0 "$pid" 2>/dev/null; then
        info "Stopping port ${port} (PID ${pid})..."
        kill "$pid" 2>/dev/null || true

        wait_count=0
        while kill -0 "$pid" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        if kill -0 "$pid" 2>/dev/null; then
            warn "Force killing PID ${pid}"
            kill -9 "$pid" 2>/dev/null || true
        fi
        stopped=$((stopped + 1))
    else
        stale=$((stale + 1))
    fi

    rm -f "$pid_file"
done
shopt -u nullglob

if [[ $stopped -eq 0 && $stale -eq 0 ]]; then
    info "No vLLM pool processes found."
else
    ok "Stopped ${stopped} process(es), removed ${stale} stale PID file(s)."
fi
