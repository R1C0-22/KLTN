"""
Short-term history selection for temporal knowledge graphs.

Implements:
  - get_short_term(history, l)

Given a list/sequence of events (each event can be either a Quadruple-like
object with attributes (subject, relation, object, timestamp) or a plain
4-tuple/list (s, r, o, t)), this module:
  1) Sorts events by timestamp (oldest -> newest).
  2) Selects the last `l` events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from common import event_fields, parse_timestamp


def get_short_term(history: Sequence[Any], l: int = 20) -> list[Any]:
    """Select the short-term history: last `l` events by timestamp.

    Parameters
    ----------
    history:
        Sequence of events (Quadruple-like objects or (s, r, o, t) tuples).
    l:
        Number of last (most recent) events to keep. Default is 20.

    Returns
    -------
    List of events of length <= l, ordered from oldest -> newest.
    """
    if l is None:
        l = 20
    l = int(l)
    if l <= 0:
        return []
    if not history:
        return []

    sortable: list[tuple[int, Any, datetime | None, str]] = []
    for idx, ev in enumerate(history):
        _, _, _, t = event_fields(ev)
        dt = parse_timestamp(t)
        sortable.append((idx, ev, dt, str(t)))

    # Primary sort: parsed timestamp presence (None goes last)
    # Secondary sort: actual datetime when available; otherwise keep stable order.
    sortable.sort(key=lambda x: (x[2] is None, x[2] or datetime.min, x[0]))
    ordered = [ev for _, ev, _, _ in sortable]
    return ordered[-l:]


if __name__ == "__main__":
    demo = [
        ("A", "meet", "B", "2014-01-02"),
        ("A", "meet", "C", "2014-01-01"),
        ("A", "visit", "D", "2013-12-31"),
        ("A", "visit", "E", "2014-01-03"),
    ]
    print(get_short_term(demo, l=2))
