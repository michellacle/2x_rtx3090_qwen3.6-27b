# vllm-tools

Scripts to manage a vLLM OpenAI-compatible API server running vLLM 0.19.1.

## Usage

### Start

```bash
bash ~/code/vllm-tools/start-vllm.sh
```

Options (set as environment variables before running):

| Variable          | Default | Description                                  |
| ----------------- | ------- | -------------------------------------------- |
| `VLLM_PORT`       | 8000    | Port to listen on                            |
| `VLLM_TP`         | 2       | Tensor parallelism (GPUs to use)             |
| `VLLM_GPU_MEM`    | 0.95    | Fraction of GPU memory to use                |
| `VLLM_MAX_LEN`    | 16384   | Maximum model length (tokens)                 |
| `VLLM_MAX_SEQS`   | 256     | Maximum number of sequences                  |
| `VLLM_CHECK_ONLY` | (empty) | Run pre-flight checks and exit (don't start) |

### Stop

```bash
bash ~/code/vllm-tools/stop-vllm.sh
```

### Test OpenAI-compatible API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-lang/Qwen3.6-27B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10,
    "temperature": 0.7
  }'
```

### Available endpoints

- `GET /health` -- health check
- `GET /v1/models` -- list models
- `POST /v1/chat/completions` -- chat completions
- `POST /v1/completions` -- text completions
- `POST /v1/embeddings` -- embeddings (if supported by model)

## Files

| File                 | Purpose                                |
| ------------------- | -------------------------------------- |
| `start-vllm.sh`     | Launch vLLM with OpenAI API enabled    |
| `stop-vllm.sh`     | Stop the running instance               |

## Logging

Output goes to `/tmp/vllm-serve.log`.
PID is stored in `/tmp/vllm-$PORT.pid`.
