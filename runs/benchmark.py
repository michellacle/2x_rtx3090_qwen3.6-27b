#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib import request


class Color:
    C = "\033[0;36m"
    B = "\033[1m"
    Y = "\033[1;33m"
    R = "\033[0;31m"
    X = "\033[0m"


def info(msg: str) -> None:
    print(f"{Color.C}[INFO]{Color.X} {msg}")


def warn(msg: str) -> None:
    print(f"{Color.Y}[WARN]{Color.X} {msg}")


def err(msg: str) -> None:
    print(f"{Color.R}[ERROR]{Color.X} {msg}", file=sys.stderr)


def post_json(url: str, payload: dict, timeout: int = 300):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        status = resp.getcode()
        data = resp.read().decode("utf-8", errors="replace")
    return status, data


def get_json(url: str, timeout: int = 10):
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)


def detect_model(base_url: str) -> str:
    info("Auto-detecting model...")
    try:
        data = get_json(f"{base_url}/models", timeout=10)
        models = data.get("data", [])
        model = models[0]["id"] if models else "unknown"
    except Exception:
        model = "unknown"
        warn("Could not auto-detect model from /models endpoint. Use -m to specify one.")
    info(f"Model: {Color.B}{model}{Color.X}")
    return model


def detect_max_model_len(base_url: str) -> int | None:
    info("Fetching max model length...")
    try:
        data = get_json(f"{base_url}/models", timeout=10)
        models = data.get("data", [])
        return models[0].get("max_model_len") if models else None
    except Exception:
        warn("Could not fetch max_model_len from /models endpoint.")
        return None


def detect_api_mode(base_url: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "_"}],
        "max_tokens": 1,
    }
    try:
        status, _ = post_json(f"{base_url}/chat/completions", payload, timeout=5)
        # Prefer OpenAI-compatible chat whenever the endpoint is reachable.
        # A 400/401/etc means the route exists; only treat 404/405 as missing.
        if status in (404, 405):
            raise RuntimeError("chat route missing")
        mode = "openai"
    except Exception:
        gen_payload = {
            "prompt": "_",
            "max_tokens": 1,
            "top_p": 1.0,
            "temperature": 1.0,
        }
        try:
            status, _ = post_json(f"{base_url}/generate", gen_payload, timeout=5)
            mode = "vllm" if status == 200 else "openai"
        except Exception:
            mode = "openai"
    info(f"Detected API mode: {mode}")
    return mode


def load_prompts(prompt_file: str, prompt: str, num_prompts: int):
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f][:num_prompts]
    else:
        prompts = [prompt] * num_prompts
    prompts = [p for p in prompts if p.strip()]
    return prompts


def get_tokenizer(tokenizer_name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(tokenizer_name)
    except Exception:
        return None


def send_request(
    prompt: str,
    base_url: str,
    model: str,
    max_tokens: int,
    top_p: float,
    temperature: float,
    api_mode: str,
    tokenizer,
):
    try:
        if api_mode == "openai":
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(max_tokens),
                "top_p": float(top_p),
                "temperature": float(temperature),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                f"{base_url}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t_t0 = time.time()
            ttft_ms = None
            usage = {}
            with request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    chunk = line.decode("utf-8", errors="replace").strip()
                    if not chunk or not chunk.startswith("data:"):
                        continue
                    text = chunk[len("data:"):]
                    if text.strip() == "[DONE]":
                        continue
                    obj = json.loads(text)
                    if obj.get("choices") and len(obj["choices"]) > 0 and obj["choices"][0].get("delta") \
                            and obj["choices"][0]["delta"].get("content") is not None and ttft_ms is None:
                        ttft_ms = round((time.time() - t_t0) * 1000, 1)
                    if "usage" in obj:
                        usage = obj["usage"]
            # fallback: TTFT not captured per-chunk, use total response time
            if ttft_ms is None:
                ttft_ms = round((time.time() - t_t0) * 1000, 1)
            return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), int(ttft_ms), int(usage.get("total_tokens", 0))
        payload = {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "temperature": float(temperature),
        }
        _, raw = post_json(f"{base_url}/generate", payload, timeout=300)
        data = json.loads(raw)
        text = ""
        if isinstance(data.get("text"), list) and data["text"]:
            text = data["text"][0] or ""
        out_tokens = len(tokenizer.encode(text)) if (text and tokenizer is not None) else 0
        return 0, out_tokens, 0, 0
    except Exception:
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description="vLLM / OpenAI-compatible endpoint benchmark")
    parser.add_argument("-u", "--url", default="http://gpus:8000/v1")
    parser.add_argument("-m", "--model", default="")
    parser.add_argument("-n", "--num-prompts", type=int, default=100)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick benchmark with 10 prompts (unless --num-prompts is explicitly set).",
    )
    parser.add_argument("-r", "--rate-limit", type=float, default=0)
    parser.add_argument("-t", "--max-tokens", type=int, default=512)
    parser.add_argument("-p", "--prompt-file", default="")
    parser.add_argument("-P", "--prompt", default="")
    parser.add_argument("-c", "--concurrency", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--api-mode", choices=["openai", "vllm"], default="")
    args = parser.parse_args()

    if args.quick and args.num_prompts == 100:
        args.num_prompts = 10

    if args.prompt_file and args.prompt:
        err("Use only --prompt-file or --prompt, not both.")
        sys.exit(1)
    if not args.prompt_file and not args.prompt:
        err("Specify --prompt or --prompt-file.")
        sys.exit(1)
    if args.prompt_file and not os.path.isfile(args.prompt_file):
        err(f"File not found: {args.prompt_file}")
        sys.exit(1)

    model = args.model or detect_model(args.url)
    if not model or model == "unknown":
        err("No model specified and could not auto-detect one. Use -m to specify a model.")
        sys.exit(1)
    max_model_len = detect_max_model_len(args.url)
    max_ctx = None
    if max_model_len is not None:
        max_ctx = int(max_model_len)
        info(f"Max context length: {Color.B}{max_ctx}{Color.X}")
    prompts = load_prompts(args.prompt_file, args.prompt, args.num_prompts)
    if not prompts:
        err("No prompts loaded.")
        sys.exit(1)

    info(f"Benchmarking {Color.B}{model}{Color.X} on {args.url}")
    info(
        f"Prompts: {len(prompts)} | Max tokens: {args.max_tokens} | "
        f"Concurrency: {args.concurrency} | Rate: {args.rate_limit} req/s"
    )
    print()

    api_mode = args.api_mode or detect_api_mode(args.url, model)
    if args.api_mode:
        info(f"Using API mode: {api_mode}")
    print()

    tokenizer = get_tokenizer(args.tokenizer)
    if api_mode == "vllm" and tokenizer is None:
        warn(f"Tokenizer '{args.tokenizer}' unavailable; output token counts may be 0.")

    info("Starting benchmark...")
    print()

    start = time.time()
    results = [None] * len(prompts)
    lock = threading.Lock()
    completed = 0

    def run_one(i: int):
        nonlocal completed
        t0 = time.time_ns()
        it, ot, ttft, total = send_request(
            prompt=prompts[i],
            base_url=args.url,
            model=model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            api_mode=api_mode,
            tokenizer=tokenizer,
        )
        t1 = time.time_ns()
        ms = max(1, int((t1 - t0) / 1_000_000))
        with lock:
            completed += 1
            print(f"\r  Progress: {completed} / {len(prompts)}", end="", flush=True)
        return i, it, ot, ttft, total, ms

    if args.concurrency <= 1:
        for i in range(len(prompts)):
            if args.rate_limit > 0:
                time.sleep(round(1.0 / args.rate_limit, 4))
            idx, it, ot, ttft, total, ms = run_one(i)
            results[idx] = (it, ot, ttft, total, ms)
        print()
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = []
            for i in range(len(prompts)):
                if args.rate_limit > 0:
                    time.sleep(round(i / args.rate_limit, 4))
                futures.append(ex.submit(run_one, i))
            for f in as_completed(futures):
                try:
                    idx, it, ot, ttft, total, ms = f.result()
                except Exception:
                    warn("A request had issues")
                    continue
                results[idx] = (it, ot, ttft, total, ms)
        print()

    duration_s = max(0.001, time.time() - start)
    vals = [v for v in results if v is not None]
    if not vals:
        err("No valid request results were recorded.")
        sys.exit(1)

    total_input = sum(v[0] for v in vals)
    total_output = sum(v[1] for v in vals)
    total_tokens = total_input + total_output
    ttfts = sorted(v[2] for v in vals)
    times = sorted(v[4] for v in vals)
    n = len(vals)

    avg_ttft = sum(ttfts) / n
    avg = sum(times) / n
    p50 = times[int(n * 0.50)]
    p90 = times[int(n * 0.90)]
    p95 = times[min(int(n * 0.95), n - 1)]
    p99 = times[min(int(n * 0.99), n - 1)]
    p50_ttft = ttfts[int(n * 0.50)]
    p90_ttft = ttfts[int(n * 0.90)]
    p95_ttft = ttfts[min(int(n * 0.95), n - 1)]
    p99_ttft = ttfts[min(int(n * 0.99), n - 1)]
    gen_times_ms = [max(1, max(0, v[4] - v[2])) for v in vals]
    total_gen_s = sum(gen_times_ms) / 1000.0
    tps = n / duration_s
    tpm = total_output / duration_s
    tokps = total_tokens / duration_s
    avg_tokps_out = total_output / total_gen_s if total_gen_s > 0 else 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"benchmark_results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "request_id",
                "model",
                "prompt_hash",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "ttft_ms",
                "ctx_pct",
                "response_time_ms",
                "gen_time_ms",
                "tpm",
            ]
        )
        for i, (it, ot, ttft, total, ms) in enumerate(vals, start=1):
            total_tok = it + ot
            req_tpm = int(ot * 60 / max(ms, 1))
            h = hashlib.md5(prompts[i - 1].encode("utf-8")).hexdigest()
            ctx_pct = round(it / max_ctx * 100, 1) if max_ctx else None
            w.writerow([i, model, h, it, ot, total_tok, ttft, ctx_pct, ms, max(0, ms - ttft), req_tpm])

    info(f"Results: {Color.B}{csv_path}{Color.X}")
    print()
    print("==============================================")
    print("  BENCHMARK SUMMARY")
    print("==============================================")
    print(f"  Model:            {model}")
    print(f"  Max ctx:          {Color.B}{max_ctx or 'N/A'}{Color.X}")
    print(f"  Endpoint:         {args.url}")
    print(f"  Requests:         {n}")
    print(f"  Duration:         {duration_s:.2f}s")
    print("------------------------------------------------")
    print(f"  Total input tokens:      {total_input}")
    print(f"  Total output tokens:     {total_output}")
    print(f"  Total tokens:            {total_tokens}")
    print("------------------------------------------------")
    print(f"  Throughput:              {tps:.2f} req/s")
    print(f"  Output throughput:       {tokps:.2f} tok/s")
    print(f"  Output tok/s (gen-only): {avg_tokps_out:.1f}")
    print(f"  Output tok/min:          {tpm:.2f}")
    print("------------------------------------------------")
    print(f"  Avg TTFT:            {avg_ttft:.1f} ms")
    print(f"  TTFT P50:            {p50_ttft:.1f} ms | P90: {p90_ttft:.1f} | P95: {p95_ttft:.1f} | P99: {p99_ttft:.1f}")
    print("------------------------------------------------")
    print(f"  Avg response time:   {avg:.0f} ms")
    print(f"  Response P50:        {p50} ms | P90: {p90} | P95: {p95} | P99: {p99}")
    print("==============================================")
    print()

    info("First 20 requests:")
    print("------ ---- ---- --- ---- ------ ------")
    print(f"   {'Req#':<6} {'In':<6} {'Out':<6} {'Ctx%':<6} {'TTFT':<8} {'Time':<8} {'tok/s':<8}")
    print("------ ---- ---- --- ---- ------ ------")
    for i, (it, ot, ttft, total, ms) in enumerate(vals[:20], start=1):
        tok_s = int(ot * 1000 / max(ms, 1))
        gen = max(0, ms - ttft)
        tok_s_gen = int(ot * 1000 / max(gen, 1)) if gen > 0 else 0
        ctx_pct = round(it / max_ctx * 100, 1) if max_ctx else None
        ctx_s = f"{ctx_pct}%" if ctx_pct is not None else "N/A"
        print(f"    {i:<6} {it:<8} {ot:<8} {ctx_s:<8} {ttft:<8.1f} {ms:<8} {tok_s_gen:<8}")
    if n > 20:
        print(f"    ... and {n - 20} more (see CSV)")
    print()
    print(f"Full CSV: {csv_path}")


if __name__ == "__main__":
    main()
