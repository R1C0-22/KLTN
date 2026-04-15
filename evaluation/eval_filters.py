"""
Evaluation filtering helpers for TKG forecasting metrics.

Implements three modes:
  - none: no filtering
  - static: filter by (s, r, o) membership in train/valid/test
  - time-aware: filter by (s, r, o, t) membership in train/valid/test
"""

from __future__ import annotations

from typing import Any, Sequence

from common import event_fields


FilterMode = str


def normalize_filter_mode(raw: str | None) -> FilterMode:
    """Normalize filter mode and validate supported values."""
    mode = (raw or "none").strip().lower()
    if mode not in {"none", "static", "time-aware"}:
        raise ValueError(f"Unsupported EVAL_FILTER={raw!r}; use none|static|time-aware")
    return mode


def build_filter_index(
    *,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    test_data: Sequence[Any],
    mode: FilterMode,
) -> set[tuple[str, ...]]:
    """Build lookup index for filtered metrics."""
    if mode == "none":
        return set()

    all_events = list(train_data) + list(valid_data) + list(test_data)
    if mode == "static":
        out: set[tuple[str, ...]] = set()
        for ev in all_events:
            s, r, o, _t = event_fields(ev)
            out.add((s.strip(), r.strip(), o.strip()))
        return out

    # time-aware
    out_t: set[tuple[str, ...]] = set()
    for ev in all_events:
        s, r, o, t = event_fields(ev)
        out_t.add((s.strip(), r.strip(), o.strip(), t.strip()))
    return out_t


def filter_ranked_predictions(
    *,
    query: tuple[str, str, str, str],
    ground_truth: str,
    ranked_candidates: Sequence[tuple[str, float]],
    index: set[tuple[str, ...]],
    mode: FilterMode,
) -> list[str]:
    """Apply filtered ranking protocol to ranked predictions."""
    sq, rq, _oq, tq = query
    s = sq.strip()
    r = rq.strip()
    t = tq.strip()
    gt = ground_truth.strip()

    ranked = [cand.strip() for cand, _ in ranked_candidates]
    if mode == "none":
        return ranked

    filtered: list[str] = []
    for cand in ranked:
        if cand == gt:
            filtered.append(cand)
            continue

        if mode == "static":
            if (s, r, cand) not in index:
                filtered.append(cand)
            continue

        # time-aware
        if (s, r, cand, t) not in index:
            filtered.append(cand)

    return filtered


def compute_rank(predictions: Sequence[str], ground_truth: str) -> int | None:
    """Compute filtered rank with duplicate-gold correction."""
    gt = ground_truth.strip()
    idx = next((i for i, p in enumerate(predictions) if p == gt), None)
    if idx is None:
        return None
    before = [p for p in predictions[:idx] if p != gt]
    return len(before) + 1

