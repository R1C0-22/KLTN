"""
Cloud LLM adapter for Colab / API-based models.

This module exposes three callables with the same interface as the former
Ollama adapter so that the rest of the codebase can stay unchanged:

    - generate_fn(prompt: str) -> str
    - score_fn(prompt: str, events: Sequence[Any]) -> list[float]
    - predict_fn(prompt: str) -> str

Under the hood it uses `Code.llm.call_llm`, which is already configurable
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
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence, List

from Code.llm import call_llm


def generate_fn(prompt: str) -> str:
    """Free-form generation used for analogical reasoning text."""
    out = call_llm(prompt)
    if not isinstance(out, str):
        out = str(out)
    return out.strip()


def _extract_first_json_array(text: str) -> List[float]:
    """
    Extract the first JSON array from a model response and parse as floats.

    Used by `score_fn`. The `filter_prompt.txt` template already instructs
    the model to output ONLY a JSON array, but we stay defensive here.
    """
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        raise ValueError(f"Could not find JSON array in model output: {text[:200]}")
    arr_text = match.group(0)
    parsed = json.loads(arr_text)
    if not isinstance(parsed, list):
        raise ValueError("Extracted JSON is not a list.")
    return [float(x) for x in parsed]


def score_fn(prompt: str, events: Sequence[Any]) -> List[float]:
    """
    Long-term history scoring used by `compute_scores_with_llm`.

    The prompt (from `prompts/filter_prompt.txt`) must instruct the model
    to output ONLY a JSON array of real-valued logits, one per event.
    """
    raw = call_llm(prompt)
    if not isinstance(raw, str):
        raw = str(raw)
    scores = _extract_first_json_array(raw.strip())

    expected = len(events)
    if len(scores) < expected:
        raise ValueError(
            f"Expected at least {expected} scores from model, got {len(scores)}."
        )
    if len(scores) > expected:
        scores = scores[:expected]
    return scores


def predict_fn(prompt: str) -> str:
    """
    Final object prediction used by `predict_next_object`.

    The prediction prompt (`prompts/prediction_prompt.txt`) tells the model
    to return ONLY the predicted entity string.
    """
    out = call_llm(prompt)
    if not isinstance(out, str):
        out = str(out)
    text = out.strip()

    # Remove surrounding quotes if any (some chat models quote JSON-style).
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
    ):
        text = text[1:-1].strip()

    # Use first line as the entity.
    return text.splitlines()[0].strip()


if __name__ == "__main__":
    # Very small smoke test (will fail if API keys / provider are not set).
    print("generate_fn:", generate_fn("Say 'hello' and nothing else."))
    demo_scores = score_fn(
        "[0.1, 0.2, 0.3]",
        events=[1, 2, 3],
    )
    print("score_fn (dummy parse):", demo_scores)
    print("predict_fn:", predict_fn("Return only the word China."))

