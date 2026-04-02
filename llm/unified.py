"""
Unified LLM interface for this project.

Supports multiple providers behind a single function:

    from llm import call_llm
    text = call_llm("Your prompt here")

Currently supported providers
-----------------------------
- OpenAI (chat/completions-style HTTP API)
- Groq (Groq cloud chat API)

Provider & model selection
--------------------------
Configuration is via environment variables:

    LLM_PROVIDER=openai|groq         # which backend to use

For OpenAI:
    OPENAI_API_KEY=...               # required
    OPENAI_BASE_URL=...              # optional, default: https://api.openai.com/v1
    OPENAI_MODEL=...                 # optional, default: gpt-4o-mini

For Groq:
    GROQ_API_KEY=...                 # required
    GROQ_BASE_URL=...                # optional, default: https://api.groq.com/openai/v1
    GROQ_MODEL=...                   # optional, default: llama-3.1-8b-instant

`call_llm(prompt)` always returns a plain string with the model’s text output.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Call the configured LLM provider with a simple text prompt.

    Returns the model's textual response (stripped).
    """
    provider = (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider == "openai":
        return _call_openai(prompt).strip()
    if provider == "groq":
        return _call_groq(prompt).strip()
    raise ValueError(f"Unsupported LLM_PROVIDER='{provider}'. Use 'openai' or 'groq'.")


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _call_openai(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"OpenAI API error {e.code}: {body[:500]}") from e

    data = json.loads(raw)
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response format: {raw[:500]}") from exc


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------

def _call_groq(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.2")),
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"Groq API error {e.code}: {body[:500]}") from e

    data = json.loads(raw)
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:
        raise RuntimeError(f"Unexpected Groq response format: {raw[:500]}") from exc


if __name__ == "__main__":
    # Simple manual test:
    #   set LLM_PROVIDER, OPENAI_* or GROQ_* env vars, then run:
    #   python -m llm.unified
    test_prompt = "Say 'hello' and nothing else."
    print(call_llm(test_prompt))

