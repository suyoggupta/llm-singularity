#!/usr/bin/env python3
"""Compare layer-0 intermediate values between HF and our inference.

Tests: embedding → RMSNorm → Q/K/V projections to find where divergence starts.
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/lustre/fs1/portfolios/coreai/users/suyogg/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"


def main():
    print("Loading HF model (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Use chat template like our server does
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Chat-templated input ({len(input_ids)} tokens): {text[:100]}...")
    print(f"First 10 token IDs: {input_ids[:10]}")

    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # Extract intermediate values using hooks
    intermediates = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[0].detach()
            else:
                intermediates[name] = output.detach()
        return hook

    # Register hooks
    model.model.embed_tokens.register_forward_hook(hook_fn("embed"))
    model.model.layers[0].input_layernorm.register_forward_hook(hook_fn("layer0_input_norm"))
    model.model.layers[0].self_attn.q_proj.register_forward_hook(hook_fn("layer0_q"))
    model.model.layers[0].self_attn.k_proj.register_forward_hook(hook_fn("layer0_k"))
    model.model.layers[0].self_attn.v_proj.register_forward_hook(hook_fn("layer0_v"))
    model.model.norm.register_forward_hook(hook_fn("final_norm"))
    model.lm_head.register_forward_hook(hook_fn("logits"))

    with torch.no_grad():
        outputs = model(input_tensor)

    # Print key values that our C++ code should match
    print("\n=== REFERENCE VALUES (last token) ===")

    for name in ["embed", "layer0_input_norm", "layer0_q", "layer0_k", "layer0_v", "final_norm", "logits"]:
        val = intermediates[name][0, -1, :8].numpy()  # last token, first 8 dims
        full = intermediates[name][0, -1, :].numpy()
        print(f"\n{name} (last token, first 8 dims):")
        print(f"  {val}")
        print(f"  shape: {intermediates[name].shape}")
        print(f"  stats: min={full.min():.6f}, max={full.max():.6f}, mean={full.mean():.6f}, std={full.std():.6f}")

    # Final logits
    logits = intermediates["logits"][0, -1, :].numpy()
    top5 = np.argsort(logits)[-5:][::-1]
    print("\nTop-5 predicted tokens:")
    for idx in top5:
        print(f"  {idx} ({tokenizer.decode([idx])!r}): {logits[idx]:.4f}")

    # Also check: what does the model predict with just BOS?
    print("\n=== SANITY CHECK: BOS-only input ===")
    bos_input = torch.tensor([[tokenizer.bos_token_id or 128000]], dtype=torch.long)
    with torch.no_grad():
        bos_out = model(bos_input)
    bos_logits = bos_out.logits[0, 0, :].numpy()
    bos_top5 = np.argsort(bos_logits)[-5:][::-1]
    print("BOS-only top-5:")
    for idx in bos_top5:
        print(f"  {idx} ({tokenizer.decode([idx])!r}): {bos_logits[idx]:.4f}")


if __name__ == "__main__":
    main()
