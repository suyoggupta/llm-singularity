#!/usr/bin/env python3
"""End-to-end test: starts the server, sends a request, checks output."""

import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL_DIR = os.environ.get(
    "TEST_MODEL_PATH",
    "/lustre/fs1/portfolios/coreai/users/suyogg/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)
BINARY = os.environ.get(
    "LLM_SERVE_BINARY",
    os.path.join(os.path.dirname(__file__), "..", "build", "app", "llm-serve-llama"),
)
PORT = 18234  # use a non-standard port to avoid conflicts


def wait_for_server(port, timeout=300):
    """Wait until the server's /health endpoint responds."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def test_chat_completion(port):
    """Send a simple chat completion and check the response is coherent."""
    payload = {
        "model": "llama",
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        "max_tokens": 20,
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()

    # Check structure
    assert "choices" in data, "Missing 'choices' in response"
    assert len(data["choices"]) > 0, "Empty choices"

    content = data["choices"][0].get("message", {}).get("content", "")
    print(f"Generated content: '{content}'")

    # The response should contain "4" somewhere
    assert "4" in content, f"Expected '4' in response, got: '{content}'"
    print("PASS: Chat completion returns correct answer")


def test_streaming(port):
    """Test SSE streaming returns tokens incrementally."""
    payload = {
        "model": "llama",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": True,
    }
    r = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=payload,
        timeout=120,
        stream=True,
    )
    assert r.status_code == 200

    tokens = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str == "[DONE]":
            break
        chunk = json.loads(data_str)
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            tokens.append(content)

    full_text = "".join(tokens)
    print(f"Streamed tokens: {tokens}")
    print(f"Full streamed text: '{full_text}'")
    assert len(tokens) > 0, "No tokens received via streaming"
    print("PASS: Streaming returns tokens")


def test_completions_endpoint(port):
    """Test the /v1/completions endpoint."""
    payload = {
        "model": "llama",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0.0,
    }
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json=payload,
        timeout=120,
    )
    print(f"Completions status: {r.status_code}")
    print(f"Completions response: {r.text}")
    assert r.status_code == 200
    data = r.json()
    text = data["choices"][0].get("text", "")
    print(f"Completion text: '{text}'")
    # Should mention Paris
    assert len(text) > 0, "Empty completion"
    print("PASS: Completions endpoint works")


def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"SKIP: Model directory not found: {MODEL_DIR}")
        sys.exit(0)

    if not os.path.isfile(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        sys.exit(1)

    print(f"Binary: {BINARY}")
    print(f"Model: {MODEL_DIR}")
    print(f"Port: {PORT}")

    # Start server
    print("\nStarting server...")
    proc = subprocess.Popen(
        [BINARY, "--model-dir", MODEL_DIR, "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        print("Waiting for server to be ready...")
        if not wait_for_server(PORT, timeout=300):
            # Dump server output
            proc.kill()
            stdout, _ = proc.communicate(timeout=5)
            print(f"Server output:\n{stdout.decode()}")
            print("FAIL: Server did not start within timeout")
            sys.exit(1)

        print("Server is ready!\n")

        # Run tests
        passed = 0
        failed = 0

        for test_fn in [test_chat_completion, test_streaming, test_completions_endpoint]:
            print(f"\n--- {test_fn.__name__} ---")
            try:
                test_fn(PORT)
                passed += 1
            except Exception as e:
                print(f"FAIL: {e}")
                failed += 1

        print(f"\n{'='*40}")
        print(f"Results: {passed} passed, {failed} failed")
        sys.exit(1 if failed > 0 else 0)

    finally:
        print("\nStopping server...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
