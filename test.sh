#!/usr/bin/env bash
# Quick smoke test for Qwen3.6-27B server
# Tests: health, models, chat, streaming, concurrency
BASE="${1:-http://localhost:8000}"
PASS=0
FAIL=0

log() {
  printf "[%s] %-30s %s\n" "$1" "$2" "$3"
  if [ "$1" = "PASS" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}

echo "=== Qwen3.6-27B Server Test ==="
echo "Target: $BASE"
echo ""

# 1. Health
HEALTH=$(curl -sf "$BASE/health" 2>/dev/null)
if [ -n "$HEALTH" ]; then log "PASS" "Health" "OK"; else log "FAIL" "Health" "No response"; fi

# 2. List models
MODELS=$(curl -sf "$BASE/v1/models" 2>/dev/null)
if echo "$MODELS" | grep -q "Qwen3.6"; then log "PASS" "Models" "Qwen3.6-27B listed"; else log "FAIL" "Models" "Not found"; fi

# 3. Chat completion
CHAT=$(curl -sf "$BASE/v1/chat/completions" -X POST -H "Content-Type: application/json" \
  -d '{"model":"llama-lang/Qwen3.6-27B","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":5,"temperature":0}' 2>/dev/null)
CHAT_TEXT=$(echo "$CHAT" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
if [ -n "$CHAT_TEXT" ] && ! echo "$CHAT_TEXT" | grep -qi "error"; then
  log "PASS" "Chat" "'$CHAT_TEXT'"
else
  log "FAIL" "Chat" "$(echo "$CHAT" | head -c 80)"
fi

# 4. Streaming
STREAM_CHUNKS=$(curl -sf -N "$BASE/v1/chat/completions" -X POST -H "Content-Type: application/json" \
  -d '{"model":"llama-lang/Qwen3.6-27B","messages":[{"role":"user","content":"Count to 3:"}],"max_tokens":10,"stream":true}' 2>/dev/null \
  | grep -c "^data:" || echo "0")
if [ "$STREAM_CHUNKS" -gt 0 ]; then
  log "PASS" "Streaming" "$STREAM_CHUNKS chunks"
else
  log "FAIL" "Streaming" "No SSE data"
fi

# 5. Concurrent requests (3 parallel)
for i in 1 2 3; do
  curl -sf "$BASE/v1/chat/completions" -X POST -H "Content-Type: application/json" \
    -d "{\"model\":\"llama-lang/Qwen3.6-27B\",\"messages\":[{\"role\":\"user\",\"content\":\"Test $i\"}],\"max_tokens\":3,\"temperature\":0}" \
    > "/tmp/qwen_test_$i.txt" 2>/dev/null &
done
wait
CONC_OK=true
for i in 1 2 3; do
  if ! grep -q "choices" "/tmp/qwen_test_$i.txt" 2>/dev/null; then CONC_OK=false; fi
done
if [ "$CONC_OK" = true ]; then
  log "PASS" "Concurrency (3x)" "All succeeded"
else
  log "FAIL" "Concurrency (3x)" "Some failed"
fi

# Summary
echo ""
TOTAL=$((PASS + FAIL))
echo "Result: $PASS/$TOTAL passed"
[ "$FAIL" -gt 0 ] && exit 1
