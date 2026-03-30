"""
Dummy final predictor for local testing.

Sets up so `predict_next_object()` can run without a real LLM by using:
  LLM_PREDICTOR="Code.inference.dummy_predictor:predict_fn"

The function parses the JSON array of candidate objects from the prompt and
returns the last candidate (most recent in the prompt's construction).
"""

from __future__ import annotations

import json
import re
from typing import Sequence


def predict_fn(prompt: str) -> str:
    # Look for: Candidate Objects: [...]
    m = re.search(r"Candidate Objects:\s*(\[[\s\S]*?\])", prompt)
    if not m:
        return "UNKNOWN"
    try:
        candidates = json.loads(m.group(1))
        if isinstance(candidates, list) and candidates:
            # Return last candidate
            return str(candidates[-1])
    except Exception:
        pass
    return "UNKNOWN"

