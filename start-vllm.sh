#!/usr/bin/env bash
# ===================================================================
# start-vllm.sh -- start vLLM with OpenAI-compatible API enabled
# ===================================================================
set -euo pipefail

# ---- configuration ------------------------------------------------
VLLM_VENV="/home/michel/venv-vllm-ng"
MODEL_PATH="/home/michel/models/qwen3.6-27b-fp8"
PORT="${VLLM_PORT:-8000}"
HOST="0.0.0.0"
TENSOR_PARALLEL="${VLLM_TP:-2}"
GPU_MEM_UTIL="${VLLM_GPU_MEM:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_LEN:-16384}"
MAX_NUM_SEQS="${VLLM_MAX_SEQS:-256}"
PID_FILE="/tmp/vllm-${PORT}.pid"

# ---- helpers ------------------------------------------------------
is_running() {
  local pid
  pid=$(cat -- "$PID_FILE" 2>/dev/null) || return 1
  kill -0 "$pid" 2>/dev/null
}

# ---- pre-flight checks ------------------ ----------------- ------
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found — no GPU visible." >&2
  exit 1
fi

nvidia-smi -L &>/dev/null || {
  echo "ERROR: nvidia-smi failed (no driver?)" >&2
  exit 1
}

if [ -n "${VLLM_CHECK_ONLY:-}" ]; then
  echo "Pre-flight checks passed."
  exit 0
fi

# ---- stop any previous instance ------------ -------------------
if is_running; then
  echo "vLLM already running on port ${PORT} (pid=$(cat "$PID_FILE"))." >&2
  read -rp "Kill existing instance? [y/N] " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    old_pid=$(cat "$PID_FILE")
    echo "Stopping pid ${old_pid} ..."
    kill "$old_pid" 2>/dev/null || true
    sleep 2
    kill -9 "$old_pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    sleep 1
    echo "Existing instance stopped."
  else
    echo "Aborted." >&2
    exit 0
  fi
elif [ -f "$PID_FILE" ]; then
  rm -f "$PID_FILE"   # stale pid file
fi

# ---- launch ---------------------------- ---- -------------------
echo "Starting vLLM on ${HOST}:${PORT} ..."

nohup $VLLM_VENV/bin/vllm serve \
  "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --quantization fp8 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --trust-remote-code \
  --language-model-only \
  --served-model-name llama-lang/Qwen3.6-27B \
  --dtype auto \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --enable-prefix-caching \
  --reasoning-parser deepseek_r1 \
  --disable-log-stats \
  2>&1 &> /tmp/vllm-serve.log &

VLLM_PID=$!
echo "$VLLM_PID" > "$PID_FILE"
echo "vLLM started (pid ${VLLM_PID}). Waiting for health check ..."

# wait for /health to return 200
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null; then
    echo "vLLM is healthy on http://0.0.0.0:${PORT}"
    echo ""

    # ---- benchmark: streaming request with timing ----
    BODY=$(printf '{
      "model": "llama-lang/Qwen3.6-27B",
      "messages": [{"role": "user", "content": "Say hi"}],
      "max_tokens": 10,
      "temperature": 0,
      "stream": true,
      "stream_options": {"include_usage": true}
    }')

    OUTPUT=$(curl -s -w "\n---TIMING---%{time_starttransfer}---%{time_total}---" "$BODY" \
      "http://127.0.0.1:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
      2>/dev/null)

    TTFT_S=$(echo "$OUTPUT" | sed -n 's/.*---TIMING---\([0-9.]*\)-\([0-9.]*\)---.*/\1/p')
    TOTAL_S=$(echo "$OUTPUT" | sed -n 's/.*---TIMING---\([0-9.]*\)-\([0-9.]*\)---.*/\2/p')

    TTFT_MS=$(awk "BEGIN {printf \"%.1f\", ${TTFT_S:-0} * 1000}")
    TOTAL_MS=$(awk "BEGIN {printf \"%.1f\", ${TOTAL_S:-0} * 1000}")
    GEN_S=$(awk "BEGIN {v=${TOTAL_S:-0} - ${TTFT_S:-0}; printf \"%.6f\", (v>0)?v:0}")

    USAGE=$(echo "$OUTPUT" | grep -o '"usage":{[^}]*}')
    PROMPT_TOKS=$(echo "$USAGE" | grep -oP '"prompt_tokens":\s*\K[0-9]+')
    COMPLETION_TOKS=$(echo "$USAGE" | grep -oP '"completion_tokens":\s*\K[0-9]+')
    TOTAL_TOKS=$(echo "$USAGE" | grep -oP '"total_tokens":\s*\K[0-9]+')

    TOKS_OUT=$(awk "BEGIN {if(${GEN_S}>0 && ${COMPLETION_TOKS:-0}>0) printf \"%.1f\", ${COMPLETION_TOKS}/${GEN_S}; else print \"N/A\"}")
    TOKS_TOTAL=$(awk "BEGIN {if(${TOTAL_S:-0}>0 && ${TOTAL_TOKS:-0}>0) printf \"%.1f\", ${TOTAL_TOKS}/${TOTAL_S}; else print \"N/A\"}")

    echo "  TTFT:       ${TTFT_MS} ms"
    echo "  Gen time:   ${TOTAL_MS} ms"
    echo "  Context:    ${PROMPT_TOKS:-N/A} tokens"
    echo "  tok/s out:  ${TOKS_OUT}"
    echo "  tok/s total: ${TOKS_TOTAL}"

    echo ""
    echo "  vRAM:"
    nvidia-smi --query-gpu=index,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | \
      while IFS=',' read -r idx used total; do
        echo "    GPU${idx}: ${used} / ${total} GB"
      done

    echo "======="
    exit 0
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died. Check /tmp/vllm-serve.log" >&2
    exit 1
  fi
  sleep 1
done

echo "ERROR: health check timed out after 120s." >&2
echo "Check logs: /tmp/vllm-serve.log" >&2
exit 1
