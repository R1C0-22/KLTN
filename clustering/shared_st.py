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
    def _load(cache_folder: str | None = None) -> Any:
        kwargs: dict[str, Any] = {"device": device}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        if hf_token:
            try:
                return SentenceTransformer(model_name, token=hf_token, **kwargs)
            except TypeError:
                return SentenceTransformer(model_name, use_auth_token=hf_token, **kwargs)
        return SentenceTransformer(model_name, **kwargs)

    try:
        model = _load()
    except OSError as exc:
        # Colab + Google Drive can intermittently fail with:
        #   OSError: [Errno 5] Input/output error
        # when HF hub cache lives on Drive. Retry once with local cache.
        msg = str(exc).lower()
        is_io_err = getattr(exc, "errno", None) == 5 or "input/output error" in msg
        if not is_io_err:
            raise

        fallback_cache = os.environ.get("ST_LOCAL_CACHE_DIR", "/content/hf_cache").strip()
        os.makedirs(fallback_cache, exist_ok=True)
        print(
            "[shared_st] HF cache I/O error detected; retrying SentenceTransformer "
            f"with local cache_folder={fallback_cache}",
            flush=True,
        )
        model = _load(cache_folder=fallback_cache)

    _MODEL = model
    _CACHE_KEY = key
    return model


def clear_shared_sentence_transformer_cache() -> None:
    """Free memory when switching embedding models in one process (tests)."""
    global _MODEL, _CACHE_KEY
    _MODEL = None
    _CACHE_KEY = None
