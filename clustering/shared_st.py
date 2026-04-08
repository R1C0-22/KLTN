"""
Shared SentenceTransformer instance for clustering + candidate similarity (§3.1).

Avoids loading the same embedding model twice per prediction (clustering then
rank_events_by_similarity), which was noisy/slow on Colab.
"""

from __future__ import annotations

import os
from typing import Any

_MODEL: Any = None
_CACHE_KEY: tuple[str, str | None] | None = None


def get_shared_sentence_transformer(
    model_name: str,
    *,
    device: str | None = None,
) -> Any:
    """Return a cached SentenceTransformer for (model_name, device)."""
    global _MODEL, _CACHE_KEY
    from sentence_transformers import SentenceTransformer

    key = (model_name, device)
    if _MODEL is not None and _CACHE_KEY == key:
        return _MODEL

    hf_token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )
    if hf_token:
        try:
            model = SentenceTransformer(
                model_name, device=device, token=hf_token
            )
        except TypeError:
            model = SentenceTransformer(
                model_name, device=device, use_auth_token=hf_token
            )
    else:
        model = SentenceTransformer(model_name, device=device)

    _MODEL = model
    _CACHE_KEY = key
    return model


def clear_shared_sentence_transformer_cache() -> None:
    """Free memory when switching embedding models in one process (tests)."""
    global _MODEL, _CACHE_KEY
    _MODEL = None
    _CACHE_KEY = None
