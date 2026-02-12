#!/usr/bin/env bash
# ==============================================================================
# stop_services.sh - Stop vLLM QA servers
#
# Gracefully terminates vLLM servers started by start_services.sh.
# Falls back to SIGKILL after a 10-second grace period.
#
# Usage:
#   scripts/benchmark/stop_services.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# ── Usage ────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage: scripts/benchmark/stop_services.sh

Stop vLLM QA servers that were started by start_services.sh.

Sends SIGTERM first, waits up to 10 seconds, then SIGKILL if needed.
PID files are stored in: logs/benchmark/.pids/
EOF
    exit 0
fi

# ── Main ─────────────────────────────────────────────────────────────────────
step "Stopping vLLM servers..."

stopped=0
for name in a b; do
    pid=""
    if pid=$(get_vllm_pid "$name"); then
        info "Stopping Server-${name} (PID $pid)..."
        kill "$pid" 2>/dev/null || true

        # Wait up to 10s for graceful shutdown
        wait_count=0
        while kill -0 "$pid" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            warn "Force killing Server-${name} (PID $pid)"
            kill -9 "$pid" 2>/dev/null || true
        fi

        remove_vllm_pid "$name"
        stopped=$((stopped + 1))
    fi
done

if [[ $stopped -eq 0 ]]; then
    info "No vLLM servers were running."
else
    ok "Stopped $stopped server(s)."
fi
