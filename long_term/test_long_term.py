"""
Quick smoke test for long-term dynamic threshold filtering.

This test does not call a real LLM. It uses the dummy scorer:
  LLM_SCORER=Code.long_term.dummy_scorer:score_fn
"""

from __future__ import annotations

import os

import sys
from pathlib import Path

# Make sure `import Code.*` works when running this file directly:
#   python Code\long_term\test_long_term.py
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Code.long_term import compute_scores_with_llm, filter_long_term


def main() -> None:
    # Enable dummy scoring.
    os.environ.setdefault("LLM_SCORER", "Code.long_term.dummy_scorer:score_fn")

    # Synthetic history: (s, r, o, t)
    # Note: include repeated timestamps so some time-step groups have F>1.
    # Otherwise, for F=1 the paper's dynamic threshold c_j becomes 1 and
    # the within-group softmax probability is also 1, so nothing is filtered.
    history = [
        ("A", "meet", "B", "2014-01-01"),
        ("A", "visit", "C", "2014-02-15"),
        ("A", "meet", "D", "2014-02-15"),
        ("A", "visit", "E", "2014-03-01"),
    ]

    query_event = ("A", "meet", "?", "2014-02-20")
    scores = compute_scores_with_llm(history, query_event)
    assert len(scores) == len(history)

    # Use the query timestamp as t_q for dynamic thresholding, following the
    # paper's definition.
    filtered = filter_long_term(history, scores, query_time=query_event[3])
    print("scores:", [round(x, 4) for x in scores])
    print("filtered:", filtered)

    assert len(filtered) <= len(history)


if __name__ == "__main__":
    main()

