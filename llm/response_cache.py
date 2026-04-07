"""
Optional on-disk cache for LLM text responses (Colab / repeated runs).

Set LLM_CACHE_DIR to a writable directory (e.g. Google Drive path). When
unset, all cache operations are no-ops.

Namespaces used by llm.cloud_adapter:
  - "generate" — analogical reasoning
  - "score" — long-term PDC (raw model output before JSON parse)
  - "predict" — final entity prediction (normalized text)
  - "logprobs" — paper §3.3 candidate scoring (JSON: predicted + probs)

Cache keys include LLM_PROVIDER and the active model id so different models
do not share entries.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _model_fingerprint() -> str:
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    model = (
        os.environ.get("HF_MODEL_ID")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("GROQ_MODEL")
        or ""
    ).strip()
    return f"{provider}:{model}"


def _cache_root() -> Path | None:
    raw = os.environ.get("LLM_CACHE_DIR", "").strip()
    if not raw:
        return None
    root = Path(raw)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _digest(namespace: str, payload: str) -> str:
    body = f"{namespace}\n{_model_fingerprint()}\n{payload}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def cache_get(namespace: str, payload: str) -> str | None:
    root = _cache_root()
    if root is None:
        return None
    path = root / f"{_digest(namespace, payload)}.txt"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def cache_set(namespace: str, payload: str, value: str) -> None:
    root = _cache_root()
    if root is None:
        return
    path = root / f"{_digest(namespace, payload)}.txt"
    path.write_text(value, encoding="utf-8")
