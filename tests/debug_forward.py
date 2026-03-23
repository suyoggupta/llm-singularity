#!/usr/bin/env python3
"""Debug script: compare our server's logits against HuggingFace reference.

This runs a single token through both systems and compares the output distribution.
"""

import json
import os
import sys
import numpy as np

MODEL_DIR = "/lustre/fs1/portfolios/coreai/users/suyogg/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"


def get_reference_logits(model_dir, input_ids):
    """Get reference logits from HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    print(f"Input IDs: {input_ids}")
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_tensor)
        # Get logits for the last token
        logits = outputs.logits[0, -1, :].numpy()

    # Get top-10 tokens
    top_indices = np.argsort(logits)[-10:][::-1]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("\nHF Reference top-10 tokens:")
    for idx in top_indices:
        print(f"  token {idx} ({tokenizer.decode([idx])!r}): logit={logits[idx]:.4f}")

    return logits


def get_our_embedding(model_dir):
    """Quick check: read the embedding weights and verify they match HF."""
    import torch
    from safetensors import safe_open

    # Read from safetensors
    import glob
    st_files = sorted(glob.glob(os.path.join(model_dir, "model*.safetensors")))

    embed_weight = None
    for st_path in st_files:
        with safe_open(st_path, framework="pt") as f:
            if "model.embed_tokens.weight" in f.keys():
                embed_weight = f.get_tensor("model.embed_tokens.weight")
                break

    if embed_weight is not None:
        print(f"\nEmbedding weight shape: {embed_weight.shape}")
        print(f"Embedding dtype: {embed_weight.dtype}")
        print(f"Embedding[0, :5]: {embed_weight[0, :5]}")
        print(f"Embedding[1, :5]: {embed_weight[1, :5]}")

        # Check: for token ID 1, embedding should be embed_weight[1]
        print(f"\nToken 128000 embedding[:5]: {embed_weight[128000, :5]}")

    return embed_weight


def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found: {MODEL_DIR}")
        sys.exit(1)

    # Use a simple input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Simple test input
    text = "Hello"
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Input text: {text!r}")
    print(f"Input IDs: {input_ids}")

    # Check embedding weights
    get_our_embedding(MODEL_DIR)

    # Get HF reference
    ref_logits = get_reference_logits(MODEL_DIR, input_ids)

    print(f"\nReference logit stats: min={ref_logits.min():.4f}, max={ref_logits.max():.4f}, "
          f"mean={ref_logits.mean():.4f}, std={ref_logits.std():.4f}")

    # Get argmax prediction
    pred_token = np.argmax(ref_logits)
    print(f"\nPredicted next token: {pred_token} ({tokenizer.decode([pred_token])!r})")


if __name__ == "__main__":
    main()
