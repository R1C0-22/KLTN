"""
Run the full `predict_next_object()` pipeline using Ollama locally.

This script relies on the permanent Ollama fallbacks added to:
  - analogical.analogical_reasoning
  - long_term.long_term_filter
  - inference.final_prediction

It only needs:
  - Ollama installed + server running
  - a model pulled in Ollama (it will download on first run):
      export OLLAMA_MODEL=llama3.2:1b   (or llama3.2:3b)
"""

from __future__ import annotations

import os
import sys
import json
import urllib.request
from pathlib import Path


def main() -> None:
    # Ensure imports work when running this file directly from the project root.
    _project_root = str(Path(__file__).resolve().parents[1])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from inference import predict_next_object

    # Pick a small model for your 4GB VRAM.
    if "OLLAMA_MODEL" not in os.environ:
        default_model = "gemma3:1b"
        fallback_model = "llama3.2:1b"
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

        try:
            tags_url = f"{base_url}/api/tags"
            with urllib.request.urlopen(tags_url, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            available = [m.get("name") for m in payload.get("models", []) if isinstance(m, dict)]

            if default_model in available:
                os.environ["OLLAMA_MODEL"] = default_model
            elif fallback_model in available:
                os.environ["OLLAMA_MODEL"] = fallback_model
            elif available:
                os.environ["OLLAMA_MODEL"] = available[0]
        except Exception:
            # If Ollama server isn't reachable for tag listing, keep the default.
            os.environ["OLLAMA_MODEL"] = default_model
    else:
        # user already set OLLAMA_MODEL
        pass

    # Minimal synthetic history so we can run without loading dataset files.
    # query_event.data is used as the history pool.
    data = [
        ("Alice", "meet", "Bob", "2014-01-01"),
        ("Alice", "meet", "Carol", "2014-01-10"),
        ("Alice", "visit", "Dave", "2014-01-20"),
        ("Eve", "meet", "Bob", "2014-01-05"),
    ]

    query_event = {
        "subject": "Alice",
        "relation": "meet",
        "object": "?",
        "timestamp": "2014-01-15",
        "data": data,
    }

    pred = predict_next_object(query_event)
    print("Predicted object:", pred)


if __name__ == "__main__":
    main()

