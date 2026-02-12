#!/usr/bin/env bash
# ==============================================================================
# start_vllm_pool.sh - Start one vLLM instance per GPU and expose endpoint list
#
# Usage:
#   scripts/benchmark/start_vllm_pool.sh \
#     --model-path /path/to/Qwen3-VL-8B \
#     --gpus 0,1,2,3 \
#     --base-port 8000
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

MODEL_PATH="${QA_MODEL_PATH:-}"
MODEL_NAME="${QA_MODEL_NAME:-Qwen3-VL-8B}"
GPU_LIST="${VLLM_GPUS:-}"
BASE_PORT="${VLLM_BASE_PORT:-8000}"
POOL_TIMEOUT="${HEALTH_TIMEOUT:-300}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

usage() {
    cat <<EOF
Usage: scripts/benchmark/start_vllm_pool.sh [options]

Start one vLLM OpenAI-compatible server per GPU.

Options:
  --model-path <path>    Path to model weights (default: QA_MODEL_PATH)
  --model-name <name>    Served model name (default: QA_MODEL_NAME or Qwen3-VL-8B)
  --gpus <list>          Comma-separated GPU indices, e.g. 0,1,2,3
  --base-port <port>     First port; each GPU uses port+index (default: 8000)
  --timeout <sec>        Health-check timeout in seconds (default: HEALTH_TIMEOUT)
  --status               Show current pool status and exit
  -h, --help             Show this message

Environment fallbacks:
  QA_MODEL_PATH, QA_MODEL_NAME, VLLM_GPUS, VLLM_BASE_PORT, HEALTH_TIMEOUT
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
        local port pid
        port="${pid_file##*_}"
        port="${port%.pid}"
        pid="$(cat "$pid_file")"

        if kill -0 "$pid" 2>/dev/null; then
            if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
                echo -e "  port ${port}: ${GREEN}RUNNING${NC} (PID ${pid}) - ${GREEN}healthy${NC}"
            else
                echo -e "  port ${port}: ${YELLOW}RUNNING${NC} (PID ${pid}) - ${YELLOW}unhealthy${NC}"
            fi
        else
            echo -e "  port ${port}: ${RED}STALE${NC} (PID ${pid} not alive)"
        fi
    done
    shopt -u nullglob

    if [[ "$found" -eq 0 ]]; then
        echo -e "  ${DIM}No pool PID files found in ${PID_DIR}${NC}"
    fi

    local latest_endpoints="$LOG_BASE/latest/vllm_pool_endpoints.txt"
    if [[ -f "$latest_endpoints" ]]; then
        echo ""
        echo -e "${BOLD}  Latest Endpoint List${NC}"
        sed 's/^/  /' "$latest_endpoints"
    fi
    echo ""
}

wait_for_endpoint() {
    local port=$1
    local label=$2
    local elapsed=0

    info "Waiting for $label (port $port)..."
    while [[ $elapsed -lt "$POOL_TIMEOUT" ]]; do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            ok "$label ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if (( elapsed % 30 == 0 )); then
            dim "  Still waiting... ${elapsed}s / ${POOL_TIMEOUT}s"
        fi
    done

    err "$label did not become ready within ${POOL_TIMEOUT}s"
    return 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --gpus) GPU_LIST="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; shift 2 ;;
        --timeout) POOL_TIMEOUT="$2"; shift 2 ;;
        --status) show_status; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) err "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    err "--model-path is required (or set QA_MODEL_PATH in .bench.env)"
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    err "Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [[ -z "$GPU_LIST" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    else
        GPU_LIST="0"
    fi
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
    err "No GPUs resolved. Use --gpus <list>."
    exit 1
fi

print_banner
echo -e "  ${DIM}Model Path :${NC} $MODEL_PATH"
echo -e "  ${DIM}Model Name :${NC} $MODEL_NAME"
echo -e "  ${DIM}GPUs       :${NC} ${GPU_ARRAY[*]}"
echo -e "  ${DIM}Base Port  :${NC} $BASE_PORT"
echo ""

mkdir -p "$PID_DIR" "$LOG_BASE"

shopt -s nullglob
for pid_file in "$PID_DIR"/vllm_pool_*.pid; do
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        err "Existing vLLM pool process is running (PID $pid)."
        err "Run scripts/benchmark/stop_vllm_pool.sh first."
        exit 1
    fi
    rm -f "$pid_file"
done
shopt -u nullglob

for idx in "${!GPU_ARRAY[@]}"; do
    port=$((BASE_PORT + idx))
    if is_port_in_use "$port"; then
        err "Port $port is already in use."
        exit 1
    fi
done

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$LOG_BASE/$timestamp"
mkdir -p "$LOG_DIR"
rm -f "$LOG_BASE/latest"
ln -sf "$timestamp" "$LOG_BASE/latest"

step "Starting vLLM pool..."

PORTS=()
for idx in "${!GPU_ARRAY[@]}"; do
    gpu="$(echo "${GPU_ARRAY[$idx]}" | xargs)"
    port=$((BASE_PORT + idx))
    log_file="$LOG_DIR/vllm_${port}.log"

    info "Starting GPU $gpu on port $port"
    CUDA_VISIBLE_DEVICES="$gpu" python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port "$port" \
        --trust-remote-code \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        > "$log_file" 2>&1 &

    pid=$!
    echo "$pid" > "$PID_DIR/vllm_pool_${port}.pid"
    PORTS+=("$port")
    dim "  PID=$pid log=$log_file"
done

for port in "${PORTS[@]}"; do
    if ! wait_for_endpoint "$port" "vLLM-${port}"; then
        err "Pool startup failed. Run scripts/benchmark/stop_vllm_pool.sh and retry."
        exit 1
    fi
done

endpoints_file="$LOG_DIR/vllm_pool_endpoints.txt"
: > "$endpoints_file"
for port in "${PORTS[@]}"; do
    echo "http://localhost:${port}/v1" >> "$endpoints_file"
done
cp "$endpoints_file" "$LOG_BASE/vllm_pool_endpoints.txt"

echo ""
ok "vLLM pool is ready (${#PORTS[@]} instances)"
echo "  Endpoints file: $endpoints_file"
echo "  Stable copy:    $LOG_BASE/vllm_pool_endpoints.txt"
echo ""
echo "  Next:"
echo "    uv run python scripts/benchmark/run_ocr_benchmark.py --dataset arxivqa"
echo ""
