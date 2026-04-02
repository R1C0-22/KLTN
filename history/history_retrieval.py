"""
Temporal Knowledge Graph history retrieval utilities.

Implements:
  - get_entity_history(entity, data): retrieve all events involving an entity
  - filter_by_relation(history, relation): keep only events with a specific relation

Both functions are designed to work with either:
  - Quadruple objects returned by `preprocessing.load_dataset`, or
  - Plain 4-tuples: (subject, relation, object, timestamp)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence

from common import event_fields, parse_timestamp


def get_entity_history(entity: str, data: Sequence[Any]) -> list[Any]:
    """Retrieve all events related to `entity` from `data`.

    Events are returned in chronological order (oldest -> newest).

    An event is considered "related" if:
      - event.subject == entity OR event.object == entity
    (exact match after stripping whitespace)
    """
    needle = entity.strip()
    related: list[Any] = []
    with_idx: list[tuple[int, Any, str, datetime | None]] = []

    for idx, ev in enumerate(data):
        s, r, o, t = event_fields(ev)
        if s.strip() == needle or o.strip() == needle:
            dt = parse_timestamp(t)
            with_idx.append((idx, ev, t, dt))

    # Sort by parsed timestamp when possible; otherwise keep them after all parsed ones.
    # Ties are stable by original index.
    with_idx.sort(key=lambda x: (x[3] is None, x[3] or datetime.min, x[2], x[0]))

    for _, ev, _, _ in with_idx:
        related.append(ev)

    return related


def filter_by_relation(history: Iterable[Any], relation: str) -> list[Any]:
    """Filter `history` to only keep events with the same relation."""
    rel_norm = relation.strip().lower()
    filtered: list[Any] = []
    for ev in history:
        _, r, _, _ = event_fields(ev)
        if r.strip().lower() == rel_norm:
            filtered.append(ev)
    return filtered


if __name__ == "__main__":
    # Minimal self-check (does not require external datasets)
    demo_data = [
        ("USA", "meet", "China", "2014-01-01"),
        ("USA", "meet", "Mexico", "2013-12-01"),
        ("Canada", "visit", "USA", "2015-01-01"),
    ]

    hist = get_entity_history("USA", demo_data)
    only_meet = filter_by_relation(hist, "meet")

    print("History (USA):")
    for ev in hist:
        print(" ", ev)

    print("\nFiltered by relation='meet':")
    for ev in only_meet:
        print(" ", ev)

