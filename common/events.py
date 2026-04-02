from __future__ import annotations

"""
Common utilities for Temporal Knowledge Graph events.

The project uses the same (subject, relation, object, timestamp) pattern
across multiple modules. This file centralizes:

  - Extraction of (s, r, o, t) from different event representations.
  - Parsing of timestamp strings into `datetime` objects.

Keeping these helpers in one place avoids subtle inconsistencies and
duplicate timestamp format lists scattered around the codebase.
"""

from datetime import datetime, timedelta
from typing import Any, Tuple


def event_fields(event: Any) -> tuple[str, str, str, str]:
    """Extract (subject, relation, object, timestamp) from an event.

    Supported shapes:
      - Quadruple-like objects with attributes: subject, relation, object, timestamp
      - 4-tuples / 4-lists: (s, r, o, t)

    This helper is intentionally strict and *not* aware of dict-based
    query_event structures used in the inference module.
    """
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


def parse_timestamp(ts: str) -> datetime | None:
    """Parse common ICEWS/GDELT timestamp formats for sorting.

    Returns `datetime` or None if parsing fails.
    """
    ts = str(ts).strip()
    # ICEWS/GDELT preprocessing in this repo sometimes keeps time as an
    # integer snapshot id (e.g. "0", "12", ...). Treat those as ordered
    # timesteps by mapping them to a monotonically increasing datetime.
    if ts.isdigit():
        seconds = int(ts)
        return datetime(1970, 1, 1) + timedelta(seconds=seconds)
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


