#!/usr/bin/env bash
# Comprehensive vLLM Test Suite
BASE="http://localhost:8000"
PASS=0
FAIL=0
RESULTS_FILE="/home/michel/vllm_test_results.txt"
> "$RESULTS_FILE"

log() {
    echo "$1" | tee -a "$RESULTS_FILE"
}

result() {
    local status="$1" name="$2" detail="$3"
    if [ "$status" = "PASS" ]; then ((PASS++)); else ((FAIL++)); fi
    printf "[%s] %-35s %s\n" "$status" "$name" "$detail" | tee -a "$RESULTS_FILE"
}

log "===== vLLM Comprehensive Test Suite ====="
log "Time: $(date -Iseconds)"
log "vLLM Version: $(/home/michel/venv-vllm-ng/bin/python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'unknown')"
log "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)"
log ""

# 1. Health check
log "--- 1. Health Check ---"
HEALTH=$(curl -s "$BASE/health" 2>&1)
if [[ "$HEALTH" == *"running"* ]] || [[ "$HEALTH" == *"true"* ]]; then
    result "PASS" "Health check" "Server healthy: $(echo $HEALTH | head -c 200)"
else
    result "FAIL" "Health check" "Not healthy: $(echo $HEALTH | head -c 200)"
fi

# 2. Model listing
log ""
log "--- 2. Model Listing ---"
MODELS=$(curl -s "$BASE/v1/models" 2>&1)
if echo "$MODELS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert 'data' in d" 2>/dev/null; then
    MODELS_LIST=$(echo "$MODELS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; [print('   - ' + m['id']) for m in json.load(sys.stdin)['data']]" 2>/dev/null)
    result "PASS" "Model listing" "Available: $MODELS_LIST"
    MODEL_ID=$(echo "$MODELS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
else
    result "FAIL" "Model listing" "$(echo $MODELS | head -c 200)"
    MODEL_ID=""
fi

log ""
log "--- 3. Basic Completion ---"
log "Test 3a: temperature=0.7, basic prompt"
COMP1=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 10,
    \"temperature\": 0.7
}" 2>&1)
if echo "$COMP1" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT1=$(echo "$COMP1" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:50])" 2>/dev/null)
    result "PASS" "text completion" "Response: '$TEXT1'..."
else
    result "FAIL" "text completion" "$(echo $COMP1 | head -c 200)"
fi

log "Test 3b: temperature=0 (deterministic)"
COMP2=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"What color is the sky?\",
    \"max_tokens\": 20,
    \"temperature\": 0
}" 2>&1)
if echo "$COMP2" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT2=$(echo "$COMP2" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:50])" 2>/dev/null)
    result "PASS" "text completion (temp=0)" "Response: '$TEXT2'..."
else
    result "FAIL" "text completion (temp=0)" "$(echo $COMP2 | head -c 200)"
fi

log ""
log "--- 4. Chat Completion ---"
log "Test 4a: Simple question"
CHAT1=$(curl -s -X POST "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"messages\": [{\"role\":\"user\",\"content\":\"What is 2+2? Answer with just the number.\"}],
    \"max_tokens\": 5,
    \"temperature\": 0
}" 2>&1)
if echo "$CHAT1" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT3=$(echo "$CHAT1" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:50])" 2>/dev/null)
    result "PASS" "chat completion" "Response: '$TEXT3'..."
else
    result "FAIL" "chat completion" "$(echo $CHAT1 | head -c 200)"
fi

log "Test 4b: Multiple messages"
CHAT2=$(curl -s -X POST "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"messages\": [
        {\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},
        {\"role\":\"user\",\"content\":\"Tell me a short joke about computers.\"}
    ],
    \"max_tokens\": 80,
    \"temperature\": 0.9
}" 2>&1)
if echo "$CHAT2" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT4=$(echo "$CHAT2" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:80])" 2>/dev/null)
    result "PASS" "chat (multi-turn)" "Response: '$TEXT4'..."
else
    result "FAIL" "chat (multi-turn)" "$(echo $CHAT2 | head -c 200)"
fi

log ""
log "--- 5. Streaming Test ---"
STREAM=$(curl -s -N -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"Count from 1 to 5:\",
    \"max_tokens\": 15,
    \"stream\": true
}" 2>/dev/null)
STREAM_COUNT=$(echo "$STREAM" | grep -c "data:" || true)
if [ "$STREAM_COUNT" -gt 0 ]; then
    result "PASS" "streaming (SSE)" "$STREAM_COUNT SSE chunks received"
    log "   Sample stream:"
    echo "$STREAM" | head -3 | sed 's/^/            /'
else
    result "FAIL" "streaming (SSE)" "No SSE data: $(echo $STREAM | head -c 100)"
fi

log ""
log "--- 6. Sampling Parameters ---"
log "Test 6a: Temperature=1.3 (creative)"
TEMP_HIGH=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"Write a creative sentence about the internet.\",
    \"max_tokens\": 30,
    \"temperature\": 1.3
}" 2>&1)
if echo "$TEMP_HIGH" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT5=$(echo "$TEMP_HIGH" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:50])" 2>/dev/null)
    result "PASS" "sampling (temp=1.3)" "Creative response: '$TEXT5'..."
else
    result "FAIL" "sampling (temp=1.3)" "$(echo $TEMP_HIGH | head -c 200)"
fi

log "Test 6b: Top_p=0.95"
TOP_P=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"What is gravity?\",
    \"max_tokens\": 30,
    \"top_p\": 0.95
}" 2>&1)
if echo "$TOP_P" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT6=$(echo "$TOP_P" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:50])" 2>/dev/null)
    result "PASS" "sampling (top_p=0.95)" "Response: '$TEXT6'..."
else
    result "FAIL" "sampling (top_p=0.95)" "$(echo $TOP_P | head -c 200)"
fi

log "Test 6c: Top_k=5"
TOP_K=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"What is math?\",
    \"max_tokens\": 30,
    \"top_k\": 5
}" 2>&1)
if echo "$TOP_K" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT7=$(echo "$TOP_K" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:50])" 2>/dev/null)
    result "PASS" "sampling (top_k=5)" "Response: '$TEXT7'..."
else
    result "FAIL" "sampling (top_k=5)" "$(echo $TOP_K | head -c 200)"
fi

log "Test 6d: Presence penalty=0.5, Frequency penalty=0.5"
PPEL=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"Tell me about cats.\",
    \"max_tokens\": 50,
    \"presence_penalty\": 0.5,
    \"frequency_penalty\": 0.5
}" 2>&1)
if echo "$PPEL" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT8=$(echo "$PPEL" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:50])" 2>/dev/null)
    result "PASS" "sampling (presence=freq=0.5)" "Response: '$TEXT8'..."
else
    result "FAIL" "sampling (presence=freq=0.5)" "$(echo $PPEL | head -c 200)"
fi

log "Test 6e: Max tokens=3 (enforcement)"
MAXT=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"Write a long essay about history.\",
    \"max_tokens\": 3,
    \"temperature\": 0.7
}" 2>&1)
if echo "$MAXT" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT9=$(echo "$MAXT" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(len(json.load(sys.stdin)['choices'][0]['text'])); print(json.load(sys.stdin)['usage'])" 2>/dev/null)
    result "PASS" "max_tokens=3" "Output length: $TEXT9"
else
    result "FAIL" "max_tokens=3" "$(echo $MAXT | head -c 200)"
fi

log ""
log "--- 7. Concurrent Requests ---"
for i in 1 2 3 4; do
    curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
        \"model\": \"$MODEL_ID\",
        \"prompt\": \"Test concurrent request $i\",
        \"max_tokens\": 5,
        \"temperature\": 0
    }" > /tmp/vllm_concurrent_$i.txt 2>&1 &
done
wait

CONCURRENT_OK=true
for i in 1 2 3 4; do
    if ! grep -q "choices" /tmp/vllm_concurrent_$i.txt 2>/dev/null; then
        CONCURRENT_OK=false
        log "   Request $i: FAILED"
    fi
done
if [ "$CONCURRENT_OK" = true ]; then
    result "PASS" "concurrent requests (4-way)" "All 4 parallel requests succeeded"
else
    result "FAIL" "concurrent requests (4-way)" "Some requests failed"
fi

log ""
log "--- 8. Logit Bias ---"
LOGIT=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"The sky is \",
    \"max_tokens\": 5,
    \"logit_bias\": {\"24039\": 100},
    \"temperature\": 0
}" 2>&1)
if echo "$LOGIT" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT10=$(echo "$LOGIT" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
    result "PASS" "logit bias" "Response: '$TEXT10'"
else
    result "FAIL" "logit bias" "$(echo $LOGIT | head -c 200)"
fi

log ""
log "--- 9. Epsilon Cutoff ---"
EPS=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"What is machine learning?\",
    \"max_tokens\": 30,
    \"temperature\": 0.7,
    \"epsilon_cutoff\": 0.001
}" 2>&1)
if echo "$EPS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('usage') is not None" 2>/dev/null 2>&1; then
    TEXT11=$(echo "$EPS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:50])" 2>/dev/null)
    result "PASS" "epsilon cutoff (0.001)" "Response: '$TEXT11'..."
else
    result "FAIL" "epsilon cutoff (0.001)" "$(echo $EPS | head -c 200)"
fi

log ""
log "--- 10. Response format (logprobs) ---"
LOGPROBS=$(curl -s -X POST "$BASE/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL_ID\",
    \"prompt\": \"What is 5+5?\",
    \"max_tokens\": 10,
    \"logprobs\": 5,
    \"temperature\": 0
}" 2>&1)
if echo "$LOGPROBS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0].get('logprobs') is not None" 2>/dev/null 2>&1; then
    result "PASS" "logprobs (top 5)" "Logprobs included in response"
else
    result "FAIL" "logprobs (top 5)" "$(echo $LOGPROBS | head -c 200)"
fi

log ""
log "========================================="
log "SUMMARY"
log "Total: $((PASS + FAIL)) tests"
log "Passed: $PASS"
log "Failed: $FAIL"
log "========================================="
log ""
log "Results written to: $RESULTS_FILE"
cat "$RESULTS_FILE"
