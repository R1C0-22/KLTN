"""
Ollama adapter for local free LLM usage.

This adapter provides 3 callables used by the rest of your code:
  - generate_fn(prompt: str) -> str                       (analogical reasoning / generation)
  - score_fn(prompt: str, events: Sequence[Any]) -> list[float]  (long-term scoring)
  - predict_fn(prompt: str) -> str                       (final entity prediction)

It calls the Ollama HTTP API at:
  http://localhost:11434
with model tag specified by OLLAMA_MODEL (default: llama3.2:1b).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from shutil import which
from typing import Any, Sequence


_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b").strip()


def _post_json(path: str, payload: dict[str, Any], timeout_s: int = 180) -> dict[str, Any]:
    url = f"{_OLLAMA_BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except urllib.error.HTTPError as e:
        # Common case: model is not pulled yet -> auto-pull and retry.
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""

        if e.code == 404 and "model" in body.lower() and "not found" in body.lower():
            auto_pull = os.environ.get("OLLAMA_AUTO_PULL", "0").strip().lower() in {"1", "true", "yes"}
            if auto_pull:
                ollama_exe = _find_ollama_executable()
                if ollama_exe:
                    subprocess.run([ollama_exe, "pull", _OLLAMA_MODEL], check=False)
                    # Retry once
                    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")
                    return json.loads(raw)
                raise RuntimeError(
                    f"Ollama model '{_OLLAMA_MODEL}' is not downloaded yet and auto-pull "
                    f"is enabled, but we couldn't find `ollama.exe` on this machine.\n"
                    f"Please download the model in the Ollama UI, then retry."
                ) from e

            raise RuntimeError(
                f"Ollama model '{_OLLAMA_MODEL}' is not downloaded yet.\n"
                f"Please open the Ollama app and download that model (e.g. '{_OLLAMA_MODEL}'), "
                f"then retry. If you prefer CLI: `ollama pull {_OLLAMA_MODEL}`."
            ) from e

        raise


def _extract_first_json_array(text: str) -> list[float]:
    """
    Extract the first JSON array from a model response and parse floats.
    Example target: [0.12, -0.3, 0.44]
    """
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        raise ValueError(f"Could not find JSON array in model output: {text[:200]}")
    arr_text = match.group(0)
    parsed = json.loads(arr_text)
    if not isinstance(parsed, list):
        raise ValueError("Extracted JSON is not a list.")
    return [float(x) for x in parsed]


def _ensure_ollama_available(tries: int = 3, sleep_s: float = 1.0) -> None:
    # Quick health ping to fail early if Ollama isn't running.
    for attempt in range(tries):
        try:
            # GET /api/tags is lightweight and returns quickly if server is up.
            url = f"{_OLLAMA_BASE_URL}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as resp:
                _ = resp.read(32)
            return
        except Exception:
            if attempt == tries - 1:
                raise RuntimeError(
                    "Ollama server not reachable. Start it with `ollama serve`, "
                    "or just open the Ollama app and ensure it's running."
                )
            time.sleep(sleep_s)


def _find_ollama_executable() -> str | None:
    """Try to locate `ollama.exe` for auto-pull.

    We can't rely on PATH always on Windows.
    """
    exe = which("ollama")
    if exe:
        return exe

    candidates = [
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe",
    ]
    user = os.environ.get("USERNAME", "")
    if user:
        candidates.extend(
            [
                rf"C:\Users\{user}\AppData\Local\Programs\Ollama\ollama.exe",
                rf"C:\Users\{user}\AppData\Roaming\Ollama\ollama.exe",
            ]
        )

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def generate_fn(prompt: str) -> str:
    """Generate free-form text from Ollama."""
    _ensure_ollama_available()
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Keep deterministic-ish for experiments.
            "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
        },
    }
    out = _post_json("/api/generate", payload)
    return str(out.get("response", "")).strip()


def score_fn(prompt: str, events: Sequence[Any]) -> list[float]:
    """
    Score each event by calling Ollama once on the provided prompt.

    The prompt you generate elsewhere should instruct the model to output
    ONLY a JSON array of numbers with the correct length.
    """
    _ensure_ollama_available()
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
        },
    }
    out = _post_json("/api/generate", payload)
    text = str(out.get("response", "")).strip()
    scores = _extract_first_json_array(text)

    expected = len(events)
    if len(scores) < expected:
        raise ValueError(
            f"score_fn expected at least {expected} scores from model, got {len(scores)}.\n"
            f"Model output (first 300 chars): {text[:300]}"
        )
    if len(scores) > expected:
        # Be robust: some models sometimes output extra values. Keep the first N.
        scores = scores[:expected]
    return scores


def predict_fn(prompt: str) -> str:
    """
    Predict the next object entity.

    The predictor prompt elsewhere should instruct the model to output ONLY
    the entity string. We still do light cleanup.
    """
    _ensure_ollama_available()
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.1")),
        },
    }
    out = _post_json("/api/generate", payload)
    text = str(out.get("response", "")).strip()

    # Remove surrounding quotes if any.
    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        text = text[1:-1].strip()
    # Use first line.
    return text.splitlines()[0].strip()

