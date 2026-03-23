#!/usr/bin/env python3
"""Check if embedding → RMSNorm → Q_proj matches HF.

This simulates what our C++ code should do for a single BOS token,
computing each step manually and comparing against HF model output.
"""

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

MODEL_DIR = "/lustre/fs1/portfolios/coreai/users/suyogg/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms * weight


def main():
    print("Loading weights from safetensors...")
    embed_w = None
    layer0_input_norm_w = None
    layer0_q_proj_w = None

    import glob
    st_files = sorted(glob.glob(f"{MODEL_DIR}/model*.safetensors"))
    for st_path in st_files:
        with safe_open(st_path, framework="pt") as f:
            for name in f.keys():
                if name == "model.embed_tokens.weight":
                    embed_w = f.get_tensor(name).float().numpy()
                elif name == "model.layers.0.input_layernorm.weight":
                    layer0_input_norm_w = f.get_tensor(name).float().numpy()
                elif name == "model.layers.0.self_attn.q_proj.weight":
                    layer0_q_proj_w = f.get_tensor(name).float().numpy()

    print(f"embed_w shape: {embed_w.shape}")  # [128256, 4096]
    print(f"layer0_input_norm_w shape: {layer0_input_norm_w.shape}")  # [4096]
    print(f"layer0_q_proj_w shape: {layer0_q_proj_w.shape}")  # [4096, 4096]

    # Step 1: Embedding lookup for BOS token (128000)
    token_id = 128000
    hidden = embed_w[token_id]  # [4096]
    print(f"\nStep 1 - Embedding[{token_id}] first 8: {hidden[:8]}")

    # Step 2: RMSNorm
    normed = rms_norm(hidden, layer0_input_norm_w)
    print(f"\nStep 2 - RMSNorm first 8: {normed[:8]}")

    # Step 3: Q projection
    # HF linear: output = input @ weight^T (weight shape is [out, in])
    # Our C++ GEMM: C = A @ B^T where B = weight [out, in]
    # So: q = normed @ q_proj_w^T
    q = normed @ layer0_q_proj_w.T
    print(f"\nStep 3 - Q = normed @ q_proj.T first 8: {q[:8]}")
    print(f"  Q shape: {q.shape}")

    # Also compute with no transpose to see if that's the bug
    q_no_T = normed @ layer0_q_proj_w
    print(f"\nAlternate - Q = normed @ q_proj (NO transpose) first 8: {q_no_T[:8]}")

    # Now compare with HF model
    print("\n\nLoading HF model for comparison...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, device_map="cpu")
    model.eval()

    intermediates = {}
    def hook(name):
        def fn(m, inp, out):
            if isinstance(out, tuple):
                intermediates[name] = out[0].detach()
            else:
                intermediates[name] = out.detach()
        return fn

    model.model.embed_tokens.register_forward_hook(hook("embed"))
    model.model.layers[0].input_layernorm.register_forward_hook(hook("norm"))
    model.model.layers[0].self_attn.q_proj.register_forward_hook(hook("q"))

    with torch.no_grad():
        model(torch.tensor([[token_id]]))

    hf_embed = intermediates["embed"][0, 0].numpy()
    hf_norm = intermediates["norm"][0, 0].numpy()
    hf_q = intermediates["q"][0, 0].numpy()

    print(f"\nHF embed first 8: {hf_embed[:8]}")
    print(f"HF norm first 8:  {hf_norm[:8]}")
    print(f"HF Q first 8:     {hf_q[:8]}")

    print(f"\nOur embed matches HF: {np.allclose(hidden, hf_embed, atol=1e-5)}")
    print(f"Our norm matches HF:  {np.allclose(normed, hf_norm, atol=1e-4)}")
    print(f"Our Q (with T) matches HF:    {np.allclose(q, hf_q, atol=1e-3)}")
    print(f"Our Q (no T) matches HF:      {np.allclose(q_no_T, hf_q, atol=1e-3)}")

    if not np.allclose(q, hf_q, atol=1e-3):
        diff = np.abs(q - hf_q)
        print(f"  Max diff (with T): {diff.max():.6f}")
    if not np.allclose(q_no_T, hf_q, atol=1e-3):
        diff = np.abs(q_no_T - hf_q)
        print(f"  Max diff (no T):   {diff.max():.6f}")


if __name__ == "__main__":
    main()
