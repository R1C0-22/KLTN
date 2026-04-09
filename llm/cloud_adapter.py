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

Long-term PDC scoring (`score_fn`) extras (Hugging Face):
    HF_SCORE_MAX_NEW_TOKENS=...     # optional floor; auto minimum scales with chunk size so JSON is not truncated
    LLM_SCORE_PARSE_FALLBACK=1      # optional; if model output is not a JSON array, use deterministic pseudo-scores
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Sequence, List

from .response_cache import cache_get, cache_set
from .unified import call_llm, call_llm_logprobs


def _min_score_max_new_tokens(n_events: int) -> int:
    """Lower bound on new tokens so a JSON array of n floats can finish (closing `]`)."""
    n = max(1, int(n_events))
    # Keep this conservative but short: we only need a numeric JSON array.
    # Large values (e.g. 500+) make local HF runs look "hung" on Colab.
    return max(48, min(384, 4 * n + 24))


def _effective_score_max_new_tokens(n_events: int) -> int:
    """Merge explicit HF_SCORE_MAX_NEW_TOKENS with a minimum derived from chunk size."""
    raw = os.environ.get("HF_SCORE_MAX_NEW_TOKENS", "").strip()
    user = int(raw) if raw.isdigit() else 0
    need = _min_score_max_new_tokens(n_events)
    # If user sets HF_SCORE_MAX_NEW_TOKENS, treat it as an explicit override.
    # This keeps notebook behavior predictable when tuning runtime vs quality.
    return max(32, user) if user else need


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


def _extract_first_json_array(text: str) -> List[float]:
    """
    Extract the first JSON array from a model response and parse as floats.

    Used by `score_fn`. The `filter_prompt.txt` template already instructs
    the model to output ONLY a JSON array, but we stay defensive here.
    """
    text = text.strip()
    # Strip optional markdown fence so JSON is parseable.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Prefer balanced-bracket extraction: non-greedy `\\[[\\s\\S]*?\\]`
    # can stop at the first `]` inside nested structures or mis-parse arrays.
    start = text.find("[")
    if start == -1:
        raise ValueError(f"Could not find '[' in model output: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                arr_text = text[start : i + 1]
                parsed = json.loads(arr_text)
                if not isinstance(parsed, list):
                    raise ValueError("Extracted JSON is not a list.")
                return [float(x) for x in parsed]
    # Truncated generation (max_new_tokens too low): try to salvage floats after "[".
    tail = text[start + 1 :]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", tail)
    if nums:
        return [float(x) for x in nums]
    raise ValueError(f"Unclosed JSON array in model output: {text[:200]!r}")


def _fallback_scores(prompt: str, events: Sequence[Any]) -> List[float]:
    """Deterministic pseudo-logits when the model does not return parseable JSON."""
    scores: List[float] = []
    for ev in events:
        s = repr(ev) + "|" + prompt[:200]
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        v = int(h[:8], 16) / 0xFFFFFFFF
        scores.append((v * 2.0) - 1.0)
    return scores


def score_fn(prompt: str, events: Sequence[Any]) -> List[float]:
    """
    Long-term history scoring used by `compute_scores_with_llm`.

    The prompt (from `prompts/filter_prompt.txt`) must instruct the model
    to output ONLY a JSON array of real-valued logits, one per event.
    """
    cached_raw = cache_get("score", prompt)
    if cached_raw is not None:
        raw = cached_raw
    else:
        n_ev = len(events)
        score_limit = str(_effective_score_max_new_tokens(n_ev))
        if os.environ.get("LLM_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
            print(
                f"[llm] score_fn: n_events={n_ev} HF_MAX_NEW_TOKENS={score_limit} "
                f"prompt_chars={len(prompt)}",
                flush=True,
            )
        saved_tok = os.environ.get("HF_MAX_NEW_TOKENS")
        os.environ["HF_MAX_NEW_TOKENS"] = score_limit
        try:
            raw = call_llm(prompt)
        finally:
            if saved_tok is None:
                os.environ.pop("HF_MAX_NEW_TOKENS", None)
            else:
                os.environ["HF_MAX_NEW_TOKENS"] = saved_tok
        raw = raw if isinstance(raw, str) else str(raw)
        cache_set("score", prompt, raw)

    if not isinstance(raw, str):
        raw = str(raw)
    use_fallback = os.environ.get("LLM_SCORE_PARSE_FALLBACK", "").strip().lower() in (
        "1", "true", "yes", "on"
    )

    try:
        scores = _extract_first_json_array(raw.strip())
    except ValueError:
        if use_fallback:
            return _fallback_scores(prompt, events)
        raise

    expected = len(events)
    if len(scores) < expected:
        if use_fallback:
            # Pad missing scores with fallback values for remaining events
            missing = _fallback_scores(prompt, list(events)[len(scores):])
            scores.extend(missing)
        else:
            raise ValueError(
                f"Expected at least {expected} scores from model, got {len(scores)}."
            )
    elif len(scores) > expected:
        scores = scores[:expected]

    return scores


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
    use_cache = os.environ.get("LLM_CACHE_PREDICT", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
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

    use_logprob_cache = os.environ.get("LLM_CACHE_LOGPROBS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
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
    # Very small smoke test (will fail if API keys / provider are not set).
    print("generate_fn:", generate_fn("Say 'hello' and nothing else."))
    demo_scores = score_fn(
        "[0.1, 0.2, 0.3]",
        events=[1, 2, 3],
    )
    print("score_fn (dummy parse):", demo_scores)
    print("predict_fn:", predict_fn("Return only the word China."))

