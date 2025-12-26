"""
Quick HuggingFace Space connectivity test using gradio_client.

What it does:
- Uses HF_TOKEN (env or hardcoded fallback) to authenticate.
- Connects to the MedSAM Space and prints available named endpoints.
- Calls the Space root to verify it responds.

Usage:
    python hf_space_test.py

You can override the Space or token via env:
    HF_SPACE_URL="https://aniketg6-medsam-inference.hf.space" HF_TOKEN="hf_..." python hf_space_test.py
"""

import os
import sys
import json
import requests

try:
    from gradio_client import Client
except Exception as e:
    print(f"gradio_client import failed: {e}")
    sys.exit(1)


def main():
    # Get Space URL and token from environment variables
    default_space = "https://aniketg6-medsam-inference.hf.space"
    space_url = os.getenv("HF_SPACE_URL", default_space)
    
    # Token must be provided via environment variable (no hardcoded fallback)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("ERROR: HF_TOKEN or HUGGINGFACE_TOKEN environment variable is required")
        print("Usage: HF_TOKEN='your_token' python hf_space_test.py")
        sys.exit(1)

    print(f"Space URL: {space_url}")
    print(f"HF_TOKEN length: {len(hf_token) if hf_token else 0}")
    if hf_token:
        print(f"HF_TOKEN prefix/suffix: {hf_token[:6]}...{hf_token[-4:]}")
    else:
        print("No HF_TOKEN provided.")

    # 1) Simple HTTP GET to see if the Space is awake
    try:
        resp = requests.get(space_url, timeout=10)
        print(f"HTTP GET {space_url} -> {resp.status_code}")
        print(f"Response (first 200 chars): {resp.text[:200]!r}")
    except Exception as e:
        print(f"HTTP GET failed: {e}")

    # 2) Gradio client connect and list APIs
    try:
        print("Initializing gradio Client...")
        client = Client(space_url, hf_token=hf_token)
        api_info = client.view_api()
        print("✓ Connected via gradio_client")
        if not isinstance(api_info, dict):
            print(f"api_info is not a dict (type={type(api_info).__name__}): {api_info}")
        else:
            named = api_info.get("named_endpoints", {})
            print(f"Named endpoints (count={len(named)}):")
            for name, meta in named.items():
                print(f"  - {name}: {meta}")
    except Exception as e:
        print(f"gradio_client connection or API listing failed: {e}")
        # If there's a response body attached, print it
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                print(f"Response status: {getattr(resp, 'status', getattr(resp, 'status_code', 'n/a'))}")
                body = getattr(resp, 'text', None) or getattr(resp, 'body', None)
                if body:
                    print(f"Response body (first 500 chars): {str(body)[:500]}")
            except Exception as dbg_err:
                print(f"(debug printing response failed: {dbg_err})")
        sys.exit(1)


if __name__ == "__main__":
    main()

