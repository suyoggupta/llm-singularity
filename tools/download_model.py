#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

"""Download a HuggingFace model for llm-singularity.

Usage:
    python3 tools/download_model.py <model_id> [--token <hf_token>]

Prints the local path to the downloaded model on stdout.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download HF model")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g. meta-llama/Llama-3.2-1B)")
    parser.add_argument("--token", default=None, help="HuggingFace auth token for gated models")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    print(f"Downloading {args.model_id} to {cache_dir}...", file=sys.stderr)

    path = snapshot_download(
        args.model_id,
        cache_dir=cache_dir,
        token=args.token,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
    )

    # Print the path to stdout (C++ reads this)
    print(path)


if __name__ == "__main__":
    main()
