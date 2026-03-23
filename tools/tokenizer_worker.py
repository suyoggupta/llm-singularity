#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

"""Long-running tokenizer subprocess for llm-singularity.

Communicates via JSON lines over stdin/stdout.

Requests (one JSON per line on stdin):
  {"cmd": "encode", "text": "Hello world"}
  {"cmd": "decode", "ids": [1, 2, 3]}
  {"cmd": "info"}
  {"cmd": "apply_chat_template", "messages": [{"role": "user", "content": "Hi"}]}

Responses (one JSON per line on stdout):
  {"ids": [1, 2, 3]}
  {"text": "Hello world"}
  {"eos_token_id": 2, "bos_token_id": 1, "vocab_size": 32000}
  {"text": "<formatted prompt>"}
"""

import json
import sys


def main():
    model_path = sys.argv[1]

    # Try transformers first (handles all tokenizer types)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers",
              file=sys.stderr)
        sys.exit(1)

    # Signal ready
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")

            if cmd == "encode":
                ids = tokenizer.encode(req["text"], add_special_tokens=False)
                print(json.dumps({"ids": ids}), flush=True)

            elif cmd == "decode":
                text = tokenizer.decode(req["ids"], skip_special_tokens=True)
                print(json.dumps({"text": text}), flush=True)

            elif cmd == "info":
                eos = tokenizer.eos_token_id
                bos = tokenizer.bos_token_id
                # eos_token_id can be a list in some configs
                if isinstance(eos, list):
                    eos = eos[0]
                if isinstance(bos, list):
                    bos = bos[0]
                print(json.dumps({
                    "eos_token_id": eos if eos is not None else -1,
                    "bos_token_id": bos if bos is not None else -1,
                    "vocab_size": len(tokenizer),
                }), flush=True)

            elif cmd == "apply_chat_template":
                messages = req["messages"]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                print(json.dumps({"text": text}), flush=True)

            else:
                print(json.dumps({"error": f"unknown cmd: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
