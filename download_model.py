#!/usr/bin/env python3
"""download_model.py -- download a model from Hugging Face with progress.

Usage:
    python3 download_model.py REPO_ID --local-dir PATH [--token TOKEN]

Examples:
    python3 download_model.py Qwen/Qwen3.6-27B-FP8 --local-dir ~/models/qwen3.6-27b-fp8
    HF_TOKEN=xxx python3 download_model.py Qwen/Qwen3.6-27B-FP8 --local-dir ~/models/qwen3.6-27b-fp8
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("repo_id", help="Hugging Face repo (e.g. Qwen/Qwen3.6-27B-FP8)")
    parser.add_argument("--local-dir", required=True, help="Local directory to download to")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    local_dir = Path(args.local_dir).expanduser()

    # Check if already downloaded
    if (local_dir / "config.json").exists():
        print(f"Model already exists at {local_dir}")
        return 0

    # Need token for gated models
    if not token:
        print("ERROR: No Hugging Face token found.")
        print("Set HF_TOKEN environment variable or pass --token.")
        print("Get a token at: https://huggingface.co/settings/tokens")
        return 1

    print(f"Downloading {args.repo_id} to {local_dir} ...")
    print("(This may take a while depending on model size and bandwidth)")

    local_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"ERROR: Download failed: {e}", file=sys.stderr)
        return 1

    # Verify
    if (local_dir / "config.json").exists():
        total_gb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e9
        print(f"Downloaded successfully: {local_dir} ({total_gb:.1f} GB)")
        return 0
    else:
        print("ERROR: Download completed but config.json not found.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
