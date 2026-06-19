#!/usr/bin/env bash
# ===================================================================
# daemon.sh -- run vLLM directly (no interactive prompts, no benchmark)
# Called by systemd. For manual use, run serve.sh instead.
# ===================================================================
set -euo pipefail

export LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/.venv/bin/vllm" serve \
  "${MODEL_PATH}" \
  --host "${VLLM_HOST:-0.0.0.0}" \
  --port "${VLLM_PORT:-8000}" \
  --tensor-parallel-size "${VLLM_TP:-2}" \
  --gpu-memory-utilization "${VLLM_GPU_MEM:-0.88}" \
  --max-model-len "${VLLM_MAX_LEN:-131072}" \
  --max-num-seqs "${VLLM_MAX_SEQS:-2}" \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --served-model-name llama-lang/Qwen3.6-27B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --disable-log-stats
