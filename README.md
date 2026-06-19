# Qwen3.6-27B on 2x RTX 3090

Single-purpose LLM server. One model, one hardware configuration, zero bloat.

- **Model:** Qwen3.6-27B (FP8)
- **Hardware:** 2x NVIDIA RTX 3090 (24 GB each)
- **Engine:** vLLM 0.23.0 with FlashInfer
- **API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, etc.)

## Install as systemd service (recommended)

```bash
sudo bash install.sh
```

Options:

```bash
sudo bash install.sh --model /path/to/model --port 9000
sudo bash install.sh --hf-repo Qwen/Qwen3.6-27B-FP8    # custom HF repo
sudo bash install.sh --skip-download                    # model already on disk
sudo bash install.sh --dry-run                          # preview without installing
```

Hugging Face token (required for download):

```bash
# Option 1: set before running
export HF_TOKEN=hf_...
sudo -E bash install.sh

# Option 2: the installer will prompt you interactively
sudo bash install.sh
```

Manage the service:

```bash
systemctl status qwen3.6-27b            # check status
journalctl -u qwen3.6-27b -f           # follow logs
systemctl restart qwen3.6-27b           # restart
sudo bash uninstall.sh                  # remove service
```

## Manual run

```bash
# Start (with startup benchmark)
bash serve.sh

# Stop
bash kill-vllm.sh

# Test
bash test.sh

# Check GPUs
bash gpu-status.sh

# Clean logs
bash clean-logs.sh

# Pre-flight check only (don't start)
VLLM_CHECK_ONLY=1 bash serve.sh
```

## Configuration

All settings are environment variables. See `.env.example` for the full list.

| Variable       | Default  | Description                        |
|--------------- |----------|------------------------------------|
| `VLLM_PORT`    | 8000     | HTTP port                          |
| `VLLM_TP`      | 2        | Tensor parallel size (GPUs)        |
| `VLLM_GPU_MEM` | 0.88     | GPU memory utilization fraction    |
| `VLLM_MAX_LEN` | 131072   | Max context length (tokens)        |
| `VLLM_MAX_SEQS`| 2        | Max concurrent sequences           |

Override inline: `VLLM_PORT=9000 VLLM_GPU_MEM=0.92 bash serve.sh`

## API

OpenAI-compatible endpoints:

- `GET /health` — health check
- `GET /v1/models` — list models
- `POST /v1/chat/completions` — chat
- `POST /v1/completions` — completions
- `POST /v1/embeddings` — embeddings (if supported)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-lang/Qwen3.6-27B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

## Features

- **FP8 KV cache** — reduces memory, enables longer contexts
- **Multi-token prediction** — 3 speculative tokens via MTP
- **Prefix caching** — fast repeated prefixes (e.g. system prompts)
- **Qwen3 reasoning parser** — structured reasoning output
- **Qwen3 coder tool parser** — function calling support

## Files

| File                | Purpose                                  |
|---------------------|------------------------------------------|
| `install.sh`        | Install as systemd service (+ download)  |
| `uninstall.sh`      | Remove systemd service                   |
| `download_model.py` | Download model from Hugging Face         |
| `daemon.sh`         | Systemd entry point (no interactive UI)  |
| `serve.sh`          | Manual start with benchmark              |
| `test.sh`           | Quick smoke test (5 checks)              |
| `kill-vllm.sh`      | Stop the server                          |
| `gpu-status.sh`     | GPU health and memory usage              |
| `clean-logs.sh`     | Clean up log files                       |
| `.env.example`      | Configuration reference                  |

## Logging

- **Systemd:** `journalctl -u qwen3.6-27b -f`
- **Manual:** `/tmp/vllm-serve.log`
- **PID file:** `/tmp/vllm-<PORT>.pid`

## Design philosophy

Most LLM serving tools try to be universal — support every model on every hardware. This results in complex configs, hidden defaults, and fragile setups.

This repo does one thing: serve Qwen3.6-27B on 2x RTX 3090s. Every parameter is tuned for this specific combination. If you have different hardware, fork and adjust.
