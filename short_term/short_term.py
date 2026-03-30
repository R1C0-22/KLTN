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


def _event_fields(event: Any) -> tuple[str, str, str, str]:
    """Extract (s, r, o, t) from either a Quadruple or a 4-tuple."""
    if (
        hasattr(event, "subject")
        and hasattr(event, "relation")
        and hasattr(event, "object")
        and hasattr(event, "timestamp")
    ):
        return (
            str(event.subject),
            str(event.relation),
            str(event.object),
            str(event.timestamp),
        )

    if isinstance(event, (tuple, list)) and len(event) >= 4:
        s, r, o, t = event[0], event[1], event[2], event[3]
        return str(s), str(r), str(o), str(t)

    raise TypeError(
        "Unsupported event type. Expected a Quadruple-like object with "
        "(subject, relation, object, timestamp) or a 4-tuple/list (s, r, o, t)."
    )


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse common ICEWS/GDELT timestamp formats for sorting."""
    ts = str(ts).strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


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
        _, _, _, t = _event_fields(ev)
        dt = _parse_timestamp(t)
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
