#!/usr/bin/env python3
"""Compare embedding values between our C++ model and HF reference.

Runs our server in the background, queries it, and compares against HF.
Instead of that, let's directly check the BF16 conversion logic.
"""
import struct
import numpy as np

# Reproduce the C++ BF16-to-F32 conversion
def bf16_to_f32_cpp_style(bf16_bytes):
    """Simulate our C++ convert_bf16_to_f32."""
    results = []
    for i in range(0, len(bf16_bytes), 2):
        bf16_val = struct.unpack('<H', bf16_bytes[i:i+2])[0]  # uint16 LE
        # C++ code: uint32_t bits = (uint32_t)bf16_val << 16;
        f32_bits = bf16_val << 16
        f32_val = struct.unpack('<f', struct.pack('<I', f32_bits))[0]
        results.append(f32_val)
    return results

# Read raw bytes from safetensors for embed_tokens
MODEL_DIR = "/lustre/fs1/portfolios/coreai/users/suyogg/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

import json

for shard in range(1, 5):
    path = f"{MODEL_DIR}/model-0000{shard}-of-00004.safetensors"
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = json.loads(f.read(header_size))

        if "model.embed_tokens.weight" in header_json:
            info = header_json["model.embed_tokens.weight"]
            dtype = info["dtype"]
            shape = info["shape"]
            offsets = info["data_offsets"]
            print(f"Found embed_tokens in shard {shard}")
            print(f"  dtype: {dtype}, shape: {shape}")
            print(f"  offsets: {offsets}")

            data_start = 8 + header_size
            # Read raw bytes for token 128000 (BOS)
            vocab_size, hidden = shape
            elem_size = 2 if dtype == "BF16" else 4
            token_offset = 128000 * hidden * elem_size
            f.seek(data_start + offsets[0] + token_offset)
            raw = f.read(hidden * elem_size)

            # Convert first 8 elements
            if dtype == "BF16":
                our_f32 = bf16_to_f32_cpp_style(raw[:16])
                print(f"\n  BOS token raw BF16 (first 8 uint16): "
                      f"{[struct.unpack('<H', raw[i:i+2])[0] for i in range(0, 16, 2)]}")
                print(f"  Our BF16->F32 conversion (first 8): {our_f32}")

                # Compare with torch
                import torch
                bf16_tensor = torch.frombuffer(bytearray(raw[:16]), dtype=torch.bfloat16)
                torch_f32 = bf16_tensor.float().numpy()
                print(f"  Torch BF16->F32 reference (first 8): {list(torch_f32)}")

                # Check if they match
                match = all(abs(a - b) < 1e-7 for a, b in zip(our_f32, torch_f32))
                print(f"  Match: {match}")
                if not match:
                    for i in range(8):
                        print(f"    [{i}] ours={our_f32[i]:.10f} ref={torch_f32[i]:.10f} "
                              f"diff={abs(our_f32[i]-torch_f32[i]):.2e}")
            break
