"""
Shared runtime helpers for evaluation scripts.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@contextmanager
def patched_env(patch: dict[str, str]) -> Iterator[None]:
    """Temporarily patch environment variables."""
    old = {k: os.environ.get(k) for k in patch}
    try:
        for k, v in patch.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def ensure_eval_runtime() -> None:
    """Set minimal defaults so evaluation scripts run on Colab/HF.

    Priority:
      1) Explicit API providers (OpenAI/Groq) if keys are present.
      2) Otherwise use local HF provider with HF_MODEL_ID.
    """
    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_groq = bool(os.environ.get("GROQ_API_KEY", "").strip())

    if has_openai or has_groq:
        # User is running with cloud APIs; keep existing provider/model.
        return

    os.environ.setdefault("LLM_PROVIDER", "hf")
    os.environ.setdefault("HF_MODEL_ID", os.environ.get("EVAL_HF_MODEL", DEFAULT_HF_MODEL))
    os.environ.setdefault("LLM_SCORER", "llm.cloud_adapter:score_fn")
    os.environ.setdefault("LLM_GENERATOR", "llm.cloud_adapter:generate_fn")
    os.environ.setdefault("LLM_PREDICTOR", "llm.cloud_adapter:predict_fn")
    os.environ.setdefault("LLM_PREDICTOR_LOGPROBS", "llm.cloud_adapter:predict_with_logprobs_fn")
