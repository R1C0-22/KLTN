"""
Dummy LLM scorer for local testing.

Sets `LLM_SCORER="long_term.dummy_scorer:score_fn"` to enable
`compute_scores_with_llm(history)` without an actual LLM/API call.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence


def score_fn(prompt: str, events: Sequence[Any]) -> list[float]:
    # Deterministic pseudo-logit scores based on content hashes.
    # Higher means "more helpful".
    scores: list[float] = []
    for ev in events:
        s = repr(ev) + "|" + prompt[:200]
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        # Map first 8 hex chars to roughly [-1, 1]
        v = int(h[:8], 16) / 0xFFFFFFFF  # [0,1]
        scores.append((v * 2.0) - 1.0)
    return scores

