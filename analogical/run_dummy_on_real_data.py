"""
Run analogical reasoning generation on real dataset WITHOUT any real LLM.

This uses the dummy generator shipped in:
  Code/analogical/dummy_generator.py

It does NOT implement the full AnRe retrieval/selection pipeline; instead it:
  1) loads real ICEWS quadruples from Code/data/
  2) picks one query event
  3) retrieves history events involving the query subject from the train split
  4) filters by the same relation as the query
  5) selects a few "similar events"
  6) calls generate_analogical_reasoning(query_event, similar_events)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure imports work when running this file directly from the project root.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from preprocessing import load_dataset  # noqa: E402
from history import get_entity_history, filter_by_relation  # noqa: E402
from analogical import generate_analogical_reasoning  # noqa: E402


def _parse_date(ts: str) -> datetime | None:
    ts = str(ts).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def main() -> None:
    # Use the dummy generator so we have *no* external LLM dependency.
    os.environ.setdefault("LLM_GENERATOR", "analogical.dummy_generator:generate_fn")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "ICEWS05-15"

    train_quads = load_dataset(data_dir, splits=["train"])
    valid_quads = load_dataset(data_dir, splits=["valid"])
    test_quads = load_dataset(data_dir, splits=["test"])

    # Pick a query event (valid if available, else test).
    query = valid_quads[0] if valid_quads else test_quads[0]

    # Retrieve subject history from training data and filter by relation.
    history = get_entity_history(query.subject, train_quads)
    rel_history = filter_by_relation(history, query.relation)

    query_dt = _parse_date(query.timestamp)
    if query_dt is None:
        similar_events = rel_history[:3]
    else:
        # Only keep events that happened on/before the query date.
        before = []
        for ev in rel_history:
            ev_dt = _parse_date(ev.timestamp)
            if ev_dt is not None and ev_dt <= query_dt:
                before.append(ev)
        similar_events = before[-3:] if before else rel_history[:3]

    if not similar_events:
        raise RuntimeError("Could not find any similar events to feed the generator.")

    reasoning = generate_analogical_reasoning(query, similar_events)
    print("=== Query Event ===")
    print((query.subject, query.relation, query.object, query.timestamp))
    print("\n=== Similar Events ===")
    for ev in similar_events:
        print((ev.subject, ev.relation, ev.object, ev.timestamp))
    print("\n=== Generated Reasoning ===")
    print(reasoning)


if __name__ == "__main__":
    main()

