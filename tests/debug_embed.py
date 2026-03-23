#!/usr/bin/env python3
"""Check if BF16 to F32 conversion matches between our code and Python."""

import struct
import numpy as np
from safetensors import safe_open

MODEL_DIR = "/lustre/fs1/portfolios/coreai/users/suyogg/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Read raw bytes of embedding weight
for i in range(1, 5):
    path = f"{MODEL_DIR}/model-0000{i}-of-00004.safetensors"
    with safe_open(path, framework="pt") as f:
        if "model.embed_tokens.weight" in f.keys():
            t = f.get_tensor("model.embed_tokens.weight")
            print(f"Found embed_tokens in shard {i}")
            print(f"Shape: {t.shape}, dtype: {t.dtype}")

            # Convert to f32
            t_f32 = t.float()
            print(f"\nEmbed[128000, :8] (BOS token) as float32:")
            print(f"  {t_f32[128000, :8].numpy()}")

            print(f"\nEmbed[0, :8] as float32:")
            print(f"  {t_f32[0, :8].numpy()}")

            # Show raw BF16 bytes for token 128000
            raw = t[128000, :4].numpy()
            print(f"\nRaw BF16 values for token 128000[:4]: {raw}")
            # Manual BF16 to F32
            for j in range(4):
                bf16_val = raw[j]
                # torch bfloat16 -> get uint16 representation
                bf16_bytes = struct.pack('e', float(bf16_val))  # Not quite right for bf16
                print(f"  val[{j}] = {float(bf16_val):.8f} (f32: {t_f32[128000, j].item():.8f})")
            break
