"""
Smoke test for `predict_next_object()` using dummy LLM components.

This test does not call any external LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import sys
from pathlib import Path

# Ensure imports work when running this file directly from the project root.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from inference import predict_next_object


@dataclass
class QueryEvent:
    subject: str
    relation: str
    object: str
    timestamp: str
    data: list[tuple[str, str, str, str]]


def main() -> None:
    os.environ["LLM_GENERATOR"] = "analogical.dummy_generator:generate_fn"
    os.environ["LLM_SCORER"] = "long_term.dummy_scorer:score_fn"
    os.environ["LLM_PREDICTOR"] = "inference.dummy_predictor:predict_fn"

    # Synthetic temporal KG (quadruples)
    data = [
        ("Alice", "meet", "Bob", "2014-01-01"),
        ("Alice", "meet", "Carol", "2014-01-10"),
        ("Alice", "visit", "Dave", "2014-01-20"),
        ("Eve", "meet", "Bob", "2014-01-05"),
    ]

    query = QueryEvent(
        subject="Alice",
        relation="meet",
        object="?",
        timestamp="2014-01-15",
        data=data,
    )

    pred = predict_next_object(query)
    assert pred in {"Bob", "Carol", "UNKNOWN"}
    print("Prediction:", pred)
    print("Test passed.")


if __name__ == "__main__":
    main()

