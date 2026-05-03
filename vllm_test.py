#!/usr/bin/env python3
"""Comprehensive vLLM test suite for Mistral-7B-Instruct on dual RTX 3090s."""

import requests
import time
import json
import concurrent.futures
import subprocess

RESULTS = []

def test(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                RESULTS.append({'name': name, 'passed': True, 'error': None, 'time': elapsed})
                print(f"  PASS [{elapsed:.2f}s] {name}")
                return result
            except Exception as e:
                elapsed = time.time() - start
                RESULTS.append({'name': name, 'passed': False, 'error': str(e)[:200], 'time': elapsed})
                print(f"  FAIL [{elapsed:.2f}s] {name}: {e}")
                raise
        return wrapper
    return decorator

# Tests
@test("Health endpoint")
def test_health():
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200
    data = resp.json()
    print(f"    status: {data.get('status', 'unknown')}")
    return data

@test("List models")
def test_list_models():
    resp = requests.get("http://localhost:8000/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    model = data['data'][0]
    print(f"    model: {model['root']}")
    return data

@test("Basic completion")
def test_completion():
    resp = requests.post("http://localhost:8000/v1/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0.7
    })
    assert resp.status_code == 200
    data = resp.json()
    assert 'choices' in data
    text = data['choices'][0]['text']
    print(f"    Response: '{text}'")
    return data

@test("Chat completion (greedy)")
def test_chat_greedy():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        "max_tokens": 10,
        "temperature": 0.0
    })
    assert resp.status_code == 200
    data = resp.json()
    content = data['choices'][0]['message']['content']
    print(f"    Response: '{content}'")
    return data

@test("Chat completion (creative)")
def test_chat_creative():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a haiku about AI."}
        ],
        "max_tokens": 50,
        "temperature": 0.9
    })
    assert resp.status_code == 200
    data = resp.json()
    content = data['choices'][0]['message']['content']
    print(f"    Response: '{content}'")
    return data

@test("Streaming completion")
def test_streaming():
    resp = requests.post("http://localhost:8000/v1/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt": "Count from 1 to 5:",
        "max_tokens": 30,
        "temperature": 0,
        "stream": True
    }, stream=True)
    assert resp.status_code == 200
    chunks_text = []
    chunk_count = 0
    for line in resp.iter_lines():
        if line and line.startswith(b'data:'):
            data_str = line[5:].decode()
            if data_str == '[DONE]':
                break
            chunk = json.loads(data_str)
            chunk_count += 1
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                text = delta.get('text', '')
                if text:
                    chunks_text.append(text)
    result = ''.join(chunks_text)
    print(f"    Chunks: {chunk_count}, Text: '{result}'")
    return {'chunks': chunk_count, 'text': result}

@test("Chat streaming")
def test_chat_streaming():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Say hello in 3 words"}],
        "max_tokens": 20,
        "stream": True
    }, stream=True)
    assert resp.status_code == 200
    chunk_count = 0
    for line in resp.iter_lines():
        if line and line.startswith(b'data:'):
            data_str = line[5:].decode()
            if data_str == '[DONE]':
                break
            chunk_count += 1
    print(f"    Streaming chunks: {chunk_count}")
    return chunk_count

@test("High temperature (2.0)")
def test_high_temp():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Write one unexpected word."}],
        "max_tokens": 5,
        "temperature": 2.0
    })
    assert resp.status_code == 200
    content = resp.json()['choices'][0]['message']['content']
    print(f"    Response: '{content}'")
    return content

@test("Top-k sampling (k=10)")
def test_topk():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Name any animal: "}],
        "max_tokens": 10,
        "top_k": 10,
        "temperature": 1.0
    })
    assert resp.status_code == 200
    content = resp.json()['choices'][0]['message']['content']
    print(f"    Response: '{content}'")
    return content

@test("Top-p sampling (p=0.95)")
def test_topp():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Name any animal: "}],
        "max_tokens": 10,
        "top_p": 0.95,
        "temperature": 0.7
    })
    assert resp.status_code == 200
    content = resp.json()['choices'][0]['message']['content']
    print(f"    Response: '{content}'")
    return content

@test("Logit bias")
def test_logit_bias():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "The cat sat on the "}],
        "max_tokens": 5,
        "logit_bias": {"1914": 100}
    })
    assert resp.status_code == 200
    content = resp.json()['choices'][0]['message']['content']
    print(f"    Response: '{content}'")
    return content

@test("Stop sequences")
def test_stop_sequences():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Count from 1: "}],
        "max_tokens": 30,
        "stop": ["5"]
    })
    assert resp.status_code == 200
    content = resp.json()['choices'][0]['message']['content']
    print(f"    Response (should not contain 5): '{content}'")
    return content

@test("Deterministic output")
def test_deterministic():
    results = []
    for i in range(3):
        resp = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": "What is the capital of France? Answer only."}],
            "max_tokens": 10,
            "temperature": 0.0,
            "seed": 42
        })
        assert resp.status_code == 200
        content = resp.json()['choices'][0]['message']['content']
        results.append(content)
    print(f"    3 runs: {results}")
    return results

@test("Concurrent requests (12 parallel)")
def test_concurrent():
    def make_request(i):
        start = time.time()
        resp = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": f"Reply briefly. Test request {i}"}],
            "max_tokens": 15,
            "temperature": 0.7
        }, timeout=30)
        elapsed = time.time() - start
        if resp.status_code != 200:
            return {"error": resp.status_code, "text": resp.text[:100], "time": elapsed}
        return {
            'id': f"req_{i}",
            'content': resp.json()['choices'][0]['message']['content'][:50],
            'time': elapsed
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(make_request, i) for i in range(12)]
        results_list = []
        for f in concurrent.futures.as_completed(futures):
            results_list.append(f.result())
    
    times = [r['time'] for r in results_list if 'time' in r]
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    avg_time = sum(times) / len(times) if times else 0
    print(f"    12 replies, latencies: min={min_time:.2f}s, avg={avg_time:.2f}s, max={max_time:.2f}s")
    return results_list

@test("Logprobs (top-5)")
def test_logprobs():
    resp = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "The sky is "}],
        "max_tokens": 5,
        "temperature": 0.7,
        "logprobs": True,
        "top_logprobs": 5
    })
    assert resp.status_code == 200
    data = resp.json()
    choices = data['choices'][0]
    logprobs_list = choices.get('logprobs', {}).get('content', [])
    print(f"    Logprobs received: {len(logprobs_list)} tokens")
    if logprobs_list:
        first = logprobs_list[0]
        print(f"    First token top choices: {[(l['token'], l['logprob']) for l in first.get('top_logprobs', [])]}")
    return data

# Run all tests
if __name__ == "__main__":
    print("=" * 80)
    print("  vLLM TEST SUITE - Mistral-7B-Instruct on dual RTX 3090")
    print("=" * 80)
    print()
    
    tests = [
        test_health,
        test_list_models,
        test_completion,
        test_chat_greedy,
        test_chat_creative,
        test_streaming,
        test_chat_streaming,
        test_high_temp,
        test_topk,
        test_topp,
        test_logit_bias,
        test_stop_sequences,
        test_deterministic,
        test_concurrent,
        test_logprobs,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception:
            pass
    
    passed = sum(1 for r in RESULTS if r['passed'])
    failed = sum(1 for r in RESULTS if not r['passed'])
    total_time = sum(r['time'] for r in RESULTS)
    
    print()
    print("=" * 80)
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(RESULTS)} total")
    print(f"  Total test time: {total_time:.1f}s")
    print("=" * 80)
    
    if failed > 0:
        print("\nFAILED TESTS:")
        for r in RESULTS:
            if not r['passed']:
                print(f"  - {r['name']}: {r['error']}")
    
    print("\nGPU Memory Usage:")
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv'], 
                          capture_output=True, text=True)
    print(result.stdout)
