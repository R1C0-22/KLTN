"""
Common utilities for Temporal Knowledge Graph events.

Centralizes:
  - Extraction of (s, r, o, t) from different event representations
  - Parsing of timestamp strings into ``datetime`` objects
  - Shared helpers (``env_truthy``, ``log``) used across modules
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any


def event_fields(event: Any) -> tuple[str, str, str, str]:
    """Extract (subject, relation, object, timestamp) from an event.

    Supported shapes:
      - Quadruple-like objects with attributes: subject, relation, object, timestamp
      - dict with keys: subject/s, relation/r, object/o, timestamp/t/time
      - 4-tuples / 4-lists: (s, r, o, t)
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

    if isinstance(event, dict):
        s = event.get("subject") or event.get("s")
        r = event.get("relation") or event.get("r")
        o = event.get("object") or event.get("o")
        t = event.get("timestamp") or event.get("t") or event.get("time")
        if s is not None and r is not None and o is not None and t is not None:
            return str(s), str(r), str(o), str(t)

    if isinstance(event, (tuple, list)) and len(event) >= 4:
        s, r, o, t = event[0], event[1], event[2], event[3]
        return str(s), str(r), str(o), str(t)

    raise TypeError(
        "Unsupported event type. Expected a Quadruple-like object, "
        "dict with s/r/o/t keys, or a 4-tuple/list (s, r, o, t)."
    )


def parse_timestamp(ts: str) -> datetime | None:
    """Parse common ICEWS/GDELT timestamp formats for sorting.

    Integer snapshot IDs (e.g. ``"3243"``) are mapped to day offsets from
    2005-01-01 (ICEWS05-15 start date), consistent with
    ``preprocessing.verbalize._format_date``.
    """
    ts = str(ts).strip()
    if ts.isdigit():
        return datetime(2005, 1, 1) + timedelta(days=int(ts))
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def env_truthy(name: str, default: bool = False) -> bool:
    """Check if an environment variable is truthy (``1``, ``true``, ``yes``, ``on``)."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def log(msg: str) -> None:
    """Print with flush for real-time output in Colab."""
    print(msg, flush=True)


