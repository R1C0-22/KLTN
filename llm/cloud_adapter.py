"""
Cloud LLM adapter for Colab / API-based models.

This module exposes three callables with the same interface as the former
Ollama adapter so that the rest of the codebase can stay unchanged:

    - generate_fn(prompt: str) -> str
    - score_fn(prompt: str, events: Sequence[Any]) -> list[float]
    - predict_fn(prompt: str) -> str

Under the hood it uses `llm.call_llm`, which is already configurable
via environment variables:

    LLM_PROVIDER = openai | groq

For OpenAI-compatible endpoints (OpenAI, Together, DeepInfra, etc.),
set:

    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...   # e.g. https://api.openai.com/v1
    OPENAI_MODEL=...      # e.g. gpt-4o-mini, gpt-4.1, llama-3.1-8b-instant

For Groq:

    GROQ_API_KEY=...
    GROQ_BASE_URL=...
    GROQ_MODEL=...

On Colab you typically just export those env vars at the notebook level.

Optional disk cache (repeatable experiments, fewer HF forward passes):
    LLM_CACHE_DIR=/path/to/dir
    Keys include LLM_PROVIDER and HF_MODEL_ID / OPENAI_MODEL / GROQ_MODEL.
    Per-step opt-out: LLM_CACHE_PREDICT=0 (final predict_fn only);
    LLM_CACHE_LOGPROBS=0 (predict_with_logprobs_fn only).

Long-term PDC scoring (`score_fn`) now uses logprobs (paper §3.2, Table 7):
    The model selects the most helpful event by label number; we read
    the logprob for each label to derive per-event scores for DTF.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence, List

from common import env_truthy

from .response_cache import cache_get, cache_set
from .unified import call_llm, call_llm_logprobs


def generate_fn(prompt: str) -> str:
    """Free-form generation used for analogical reasoning text."""
    cached = cache_get("generate", prompt)
    if cached is not None:
        return cached.strip()

    out = call_llm(prompt)
    if not isinstance(out, str):
        out = str(out)
    text = out.strip()
    cache_set("generate", prompt, text)
    return text


def score_fn(prompt: str, events: Sequence[Any]) -> List[float]:
    """PDC scoring using logprobs (paper §3.2, Table 7, Eq. 1).

    The paper maps each historical event to a numerical label (1, 2, ...),
    obtains the LLM's log-probability for each label token, and applies
    softmax to derive the effectiveness probability distribution.

    This uses the same mechanism as ``predict_with_logprobs_fn`` (§3.3)
    and replaces the previous text-generation + JSON-parsing approach,
    which caused degenerate all-zero scores on quantized models.
    """
    n = len(events)
    labels = [str(i) for i in range(1, n + 1)]

    cached_raw = cache_get("score_lp", prompt)
    if cached_raw is not None:
        try:
            cached = json.loads(cached_raw)
            if isinstance(cached, list) and len(cached) == n:
                return [float(x) for x in cached]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    if env_truthy("LLM_VERBOSE"):
        print(
            f"[llm] score_fn(logprob): n_events={n} prompt_chars={len(prompt)}",
            flush=True,
        )

    logprobs = call_llm_logprobs(prompt, labels)
    cache_set("score_lp", prompt, json.dumps(logprobs))
    return logprobs


def _strip_outer_quotes(text: str) -> str:
    """Remove one pair of surrounding quotes; keep all lines (prediction may be line 2+)."""
    text = text.strip()
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
    ):
        text = text[1:-1].strip()
    return text


def predict_fn(prompt: str) -> str:
    """
    Final object prediction used by `predict_next_object`.

    The prediction prompt asks for a candidate **index** (1..|Oq|) or entity name.
    We return the **full** model output (after stripping outer quotes). Do not
    truncate to the first line — chatty models put the index on the last line
    (see IMPROVE.MD / paper §3.3 parsing).

    Cached when LLM_CACHE_DIR is set (same mechanism as generate_fn / score_fn).
    Set LLM_CACHE_PREDICT=0 to bypass cache for this step.

    Optional: ``HF_PREDICT_MAX_NEW_TOKENS`` overrides ``HF_MAX_NEW_TOKENS`` for this call only
    (short rationale + index; default in ``setup()`` is enough for most cases).
    """
    use_cache = env_truthy("LLM_CACHE_PREDICT", default=True)
    if use_cache:
        cached = cache_get("predict", prompt)
        if cached is not None:
            return _strip_outer_quotes(cached)

    saved_max = os.environ.get("HF_MAX_NEW_TOKENS")
    predict_max = os.environ.get("HF_PREDICT_MAX_NEW_TOKENS", "").strip()
    if predict_max.isdigit():
        os.environ["HF_MAX_NEW_TOKENS"] = predict_max
    try:
        out = call_llm(prompt)
    finally:
        if saved_max is None:
            os.environ.pop("HF_MAX_NEW_TOKENS", None)
        else:
            os.environ["HF_MAX_NEW_TOKENS"] = saved_max

    if not isinstance(out, str):
        out = str(out)
    text = _strip_outer_quotes(out)

    if use_cache:
        cache_set("predict", prompt, text)
    return text


def predict_with_logprobs_fn(
    prompt: str,
    candidates: List[str],
) -> tuple[str, List[float]]:
    """
    Final object prediction using logprob-based scoring (Paper §3.3).
    
    Paper Algorithm: "We map each candidate entity to a numerical token,
    obtain the corresponding logarithmic output La from the LLM, and convert
    it into a normalized probability using the softmax function, resulting
    in the probability distribution of each candidate answer."
    
    Parameters
    ----------
    prompt : str
        The prediction prompt (should end with "Your choice is:")
    candidates : List[str]
        List of candidate entity names
    
    Returns
    -------
    tuple[str, List[float]]
        (predicted_entity, probability_distribution)
        The predicted entity is the one with highest probability.
    """
    import math

    use_logprob_cache = env_truthy("LLM_CACHE_LOGPROBS", default=True)
    cache_payload = prompt + "\n" + json.dumps(candidates, ensure_ascii=False)
    if use_logprob_cache:
        cached = cache_get("logprobs", cache_payload)
        if cached is not None:
            try:
                payload = json.loads(cached)
                pred = str(payload["predicted"])
                probs = [float(x) for x in payload["probs"]]
                if len(probs) == len(candidates):
                    return pred, probs
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

    # Map candidates to numerical labels (1, 2, 3, ...)
    # Paper: "we map each candidate entity to a numerical token"
    labels = [str(i) for i in range(1, len(candidates) + 1)]
    
    # Get logprobs for each label
    # Paper: "obtain the corresponding logarithmic output La from the LLM"
    logprobs = call_llm_logprobs(prompt, labels)
    
    # Convert to probabilities using softmax
    # Paper: "convert it into a normalized probability using the softmax function"
    max_logprob = max(logprobs) if logprobs else 0.0
    exps = [math.exp(lp - max_logprob) for lp in logprobs]
    total = sum(exps)
    if total == 0:
        probs = [1.0 / len(candidates)] * len(candidates)
    else:
        probs = [e / total for e in exps]
    
    # Select highest probability
    # Paper: "sort the probability results and select the highest probability result"
    best_idx = probs.index(max(probs))
    predicted = candidates[best_idx] if candidates else ""

    if use_logprob_cache:
        cache_set(
            "logprobs",
            cache_payload,
            json.dumps(
                {"predicted": predicted, "probs": probs},
                ensure_ascii=False,
            ),
        )

    return predicted, probs


if __name__ == "__main__":
    print("generate_fn:", generate_fn("Say 'hello' and nothing else."))
    print("predict_fn:", predict_fn("Return only the word China."))

