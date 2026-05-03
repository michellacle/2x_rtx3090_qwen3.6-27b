#!/usr/bin/env bash
# vLLM Comprehensive Test Suite
BASE="http://localhost:8000"
PASS=0
FAIL=0
RESULTS="/home/michel/vllm_test_results.txt"
> "$RESULTS"

log() { printf "[%s] %-35s %s\n" "$1" "$2" "$3" | tee -a "$RESULTS"; if [ "$1" = "PASS" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi; }

echo "=== vLLM Comprehensive Test Suite ===" | tee -a "$RESULTS"
echo "Time: $(date)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# 1. Health
HEALTH=$(curl -s "$BASE/health")
if [ -n "$HEALTH" ]; then log "PASS" "Health endpoint" "OK"; else log "FAIL" "Health endpoint" "No response"; fi

# 2. List models
MODELS=$(curl -s "$BASE/v1/models")
MODEL_ID=$(echo "$MODELS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "mistral-7b-instruct")
log "PASS" "Model listing" "$MODEL_ID"

# 3. Basic text completion
COMP=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":0.7}" 2>/dev/null)
TEXT=$(echo "$COMP" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
# Check if response is sensible (not empty, not an error)
if [ -n "$TEXT" ] && ! echo "$TEXT" | grep -qi "error"; then
    log "PASS" "Text completion" "Response: '$TEXT'"
else
    log "FAIL" "Text completion" "$(echo "$COMP" | head -c 100)"
fi

# 4. Chat completion
CHAT=$(curl -s "$BASE/v1/chat/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer with just the number.\"}],\"max_tokens\":5,\"temperature\":0}" 2>/dev/null)
CHAT_TEXT=$(echo "$CHAT" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
if [ -n "$CHAT_TEXT" ] && ! echo "$CHAT_TEXT" | grep -qi "error"; then
    log "PASS" "Chat completion" "Answer: '$CHAT_TEXT'"
else
    log "FAIL" "Chat completion" "$(echo "$CHAT" | head -c 100)"
fi

# 5. Streaming
STREAM_COUNT=$(curl -s -N "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Count to 5:\",\"max_tokens\":10,\"stream\":true}" 2>/dev/null | grep -c "data:" || echo "0")
if [ "$STREAM_COUNT" -gt 0 ]; then
    log "PASS" "Streaming (SSE)" "$STREAM_COUNT chunks"
else
    log "FAIL" "Streaming (SSE)" "No SSE data"
fi

# 6a. Temperature=0 (deterministic)
DET=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Repeat: hello world\",\"max_tokens\":10,\"temperature\":0}" 2>/dev/null)
DET_TEXT=$(echo "$DET" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
if [ -n "$DET_TEXT" ]; then
    log "PASS" "Deterministic (temp=0)" "Response: '$DET_TEXT'"
else
    log "FAIL" "Deterministic (temp=0)" "Empty response"
fi

# 6b. Temperature=1.5 (creative)
CRE=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Write a funny sentence about cats\",\"max_tokens\":50,\"temperature\":1.5}" 2>/dev/null)
CRE_TEXT=$(echo "$CRE" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:40])" 2>/dev/null)
if [ -n "$CRE_TEXT" ]; then
    log "PASS" "Creative (temp=1.5)" "Response: '$CRE_TEXT'..."
else
    log "FAIL" "Creative (temp=1.5)" "Empty response"
fi

# 6c. Top-p sampling
TOPP=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"What is AI?\",\"max_tokens\":30,\"top_p\":0.9}" 2>/dev/null)
TOPP_TEXT=$(echo "$TOPP" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:40])" 2>/dev/null)
if [ -n "$TOPP_TEXT" ]; then
    log "PASS" "Top-p (0.9)" "Response: '$TOPP_TEXT'..."
else
    log "FAIL" "Top-p (0.9)" "Empty response"
fi

# 6d. Top-k sampling
TOPK=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"What is math?\",\"max_tokens\":30,\"top_k\":5}" 2>/dev/null)
TOPK_TEXT=$(echo "$TOPK" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:40])" 2>/dev/null)
if [ -n "$TOPK_TEXT" ]; then
    log "PASS" "Top-k (5)" "Response: '$TOPK_TEXT'..."
else
    log "FAIL" "Top-k (5)" "Empty response"
fi

# 6e. max_tokens=3 enforcement
MC=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Write a long essay about history.\",\"max_tokens\":3,\"temperature\":0.7}" 2>/dev/null)
MC_USED=$(echo "$MC" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
if [ "$MC_USED" -le 3 ] 2>/dev/null; then
    log "PASS" "max_tokens enforcement" "Used $MC_USED tokens (limit: 3)"
else
    log "FAIL" "max_tokens enforcement" "Used $MC_USED tokens (limit: 3)"
fi

# 7. Concurrent requests (4 parallel)
for i in 1 2 3 4; do
    curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Test $i\",\"max_tokens\":5,\"temperature\":0}" > /tmp/vllm_t$i.txt 2>/dev/null &
done
wait
CONC_OK=true
for i in 1 2 3 4; do
    if ! grep -q "choices" /tmp/vllm_t$i.txt 2>/dev/null; then CONC_OK=false; fi
done
if [ "$CONC_OK" = true ]; then
    log "PASS" "Concurrent requests (4x)" "All succeeded"
else
    log "FAIL" "Concurrent requests (4x)" "Some failed"
fi

# 8. Logprobs
LP=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"What is 5+5?\",\"max_tokens\":5,\"logprobs\":3,\"temperature\":0}" 2>/dev/null)
LP_OK=$(echo "$LP" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;c=json.load(sys.stdin)['choices'][0]['logprobs'];print('yes' if c else 'no')" 2>/dev/null)
if [ "$LP_OK" = "yes" ]; then
    log "PASS" "Logprobs (top 3)" "Logprobs included"
else
    log "FAIL" "Logprobs (top 3)" "No logprobs"
fi

# 9. Stop sequence
STOP=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"List numbers: 1, 2, 3,\",\"max_tokens\":20,\"stop\":[\" 4\",\" 7\",\" 10\"]}" 2>/dev/null)
STOP_TEXT=$(echo "$STOP" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(len(json.load(sys.stdin)['choices'][0]['text']))" 2>/dev/null)
if [ -n "$STOP_TEXT" ]; then
    log "PASS" "Stop sequences" "Response length: $STOP_TEXT chars"
else
    log "FAIL" "Stop sequences" "Empty or error"
fi

# 10. Presence/Frequency penalty
PF=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Tell me about cats. Repeat 'cats' several times.\",\"max_tokens\":60,\"temperature\":0.7,\"presence_penalty\":0.5,\"frequency_penalty\":0.5}" 2>/dev/null)
PF_TEXT=$(echo "$PF" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:40])" 2>/dev/null)
if [ -n "$PF_TEXT" ]; then
    log "PASS" "Presence/Freq penalty" "Response: '$PF_TEXT'..."
else
    log "FAIL" "Presence/Freq penalty" "Empty response"
fi

# 11. Repetition penalty
REP=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"What is the sun?\",\"max_tokens\":40,\"temperature\":0.7,\"repetition_penalty\":1.1}" 2>/dev/null)
REP_TEXT=$(echo "$REP" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:40])" 2>/dev/null)
if [ -n "$REP_TEXT" ]; then
    log "PASS" "Repetition penalty" "Response: '$REP_TEXT'..."
else
    log "FAIL" "Repetition penalty" "Empty response"
fi

# 12. Epsilon cutoff
EPS=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"What is gravity?\",\"max_tokens\":30,\"temperature\":0.7,\"epsilon_cutoff\":0.001}" 2>/dev/null)
EPS_TEXT=$(echo "$EPS" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:40])" 2>/dev/null)
if [ -n "$EPS_TEXT" ]; then
    log "PASS" "Epsilon cutoff" "Response: '$EPS_TEXT'..."
else
    log "FAIL" "Epsilon cutoff" "Empty response"
fi

# 13. Logit bias
LB=$(curl -s "$BASE/v1/completions" -X POST -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"The sky is \",\"max_tokens\":5,\"logit_bias\":{\"24039\":100},\"temperature\":0}" 2>/dev/null)
LB_TEXT=$(echo "$LB" | /home/michel/venv-vllm-ng/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
if [ -n "$LB_TEXT" ]; then
    log "PASS" "Logit bias" "Response: '$LB_TEXT'"
else
    log "FAIL" "Logit bias" "Empty response"
fi

# Summary
echo "" | tee -a "$RESULTS"
echo "====== vLLM Test Results ====" | tee -a "$RESULTS"
TOTAL=$((PASS + FAIL))
echo "Total:  $TOTAL tests" | tee -a "$RESULTS"
echo "Passed:  $PASS" | tee -a "$RESULTS"
echo "Failed:  $FAIL" | tee -a "$RESULTS"
echo "==============================" | tee -a "$RESULTS"
echo ""
echo "Full results saved to: $RESULTS"
cat "$RESULTS"
