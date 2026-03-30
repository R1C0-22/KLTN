"""
Final prediction module for Temporal Knowledge Graph Forecasting.

Implements an AnRe-style pipeline that combines:
  - short-term history selection
  - long-term history filtering (dynamic threshold + LLM scoring)
  - analogical reasoning generation

Then constructs a final prompt and calls an LLM to predict the missing
object entity in a query event.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


def _event_fields(event: Any) -> tuple[str, str, str, str]:
    """Extract (s, r, o, t) from either a Quadruple-like object or a tuple."""
    if (
        hasattr(event, "subject")
        and hasattr(event, "relation")
        and hasattr(event, "object")
        and hasattr(event, "timestamp")
    ):
        return (str(event.subject), str(event.relation), str(event.object), str(event.timestamp))
    if isinstance(event, (tuple, list)) and len(event) >= 4:
        s, r, o, t = event[0], event[1], event[2], event[3]
        return str(s), str(r), str(o), str(t)
    raise TypeError("query_event must be Quadruple-like or a tuple/list (s,r,o,t).")


def _parse_timestamp(ts: str) -> datetime | None:
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


def _load_callable_from_env(var_name: str) -> Callable[[str], str]:
    spec = os.environ.get(var_name, "").strip()
    if not spec:
        raise EnvironmentError(
            f"{var_name} is not set. Expected format 'module_path:function_name' "
            f"with signature callable(prompt: str) -> str."
        )
    if ":" not in spec:
        raise ValueError(f"{var_name} must be in format 'module_path:function_name'.")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"{var_name} resolved to non-callable: {spec}")
    return fn  # type: ignore[return-value]


def _verbalize_query_with_mask(s: str, r: str, t: str, mask: str = "?") -> str:
    # We reuse the verbalization logic for known relations, but keep `object`
    # masked so it reads as a prediction question.
    from Code.preprocessing import verbalize_event

    return verbalize_event(s, r, mask, t)


def _choose_similarity_events(events: Sequence[Any], m: int = 3) -> list[Any]:
    if not events:
        return []
    # Use the most recent m events (chronological order assumed already).
    return list(events[-m:])


def _extract_predicted_object(llm_output: str, candidates: Sequence[str]) -> str:
    out = llm_output.strip()
    if not candidates:
        return out

    # Prefer an exact candidate match if present.
    for c in candidates:
        if c and c in out:
            return c

    # Also try extracting from JSON-like outputs.
    try:
        maybe = json.loads(out)
        if isinstance(maybe, str):
            return maybe
    except Exception:
        pass

    # Fall back: take first line token.
    first = out.splitlines()[0].strip()
    for c in candidates:
        if first == c:
            return c
    return first


def _load_history_data_from_query_event(query_event: Any) -> list[Any]:
    """
    Retrieve dataset quadruples for history building.

    The function supports:
      - query_event.data (or query_event["data"]): already-loaded quadruples
      - env var `TKG_DATA_DIR`: directory containing train/valid/test
    """
    if hasattr(query_event, "data"):
        return list(query_event.data)
    if isinstance(query_event, dict) and "data" in query_event:
        return list(query_event["data"])

    data_dir = os.environ.get("TKG_DATA_DIR", "").strip()
    if not data_dir:
        raise EnvironmentError(
            "No data provided. Pass query_event.data or set TKG_DATA_DIR "
            "to a dataset directory like Code/data/ICEWS05-15."
        )

    from Code.preprocessing import load_dataset

    # Default: use train split as history pool.
    return load_dataset(data_dir, splits=["train"])


def predict_next_object(query_event: Any) -> str:
    """Predict the missing object entity for a query event (s, r, ?, t).

    Expected query_event formats:
      - Quadruple-like with fields: subject, relation, object, timestamp
      - tuple/list: (subject, relation, object, timestamp)

    The missing object should be either '?' or None. Other placeholder
    strings are allowed but candidates are extracted from history events.

    LLM configuration:
      - LLM_SCORER: required by long-term filtering (Code.long_term)
      - LLM_GENERATOR: required by analogical reasoning generation
      - LLM_PREDICTOR (preferred) or LLM_GENERATOR fallback:
            callable(prompt: str) -> str
    """
    from Code.preprocessing import verbalize_event
    from Code.history import get_entity_history, filter_by_relation
    from Code.short_term import get_short_term
    from Code.long_term import compute_scores_with_llm, filter_long_term
    from Code.analogical import generate_analogical_reasoning

    s, r, o, t = _event_fields(query_event)
    mask = "?" if o is None or str(o).strip() in {"?", "None", "null"} else str(o)

    # Load history pool
    data = _load_history_data_from_query_event(query_event)

    # Retrieve entity-related events and filter by same relation
    entity_history = get_entity_history(s, data)
    rel_history = filter_by_relation(entity_history, r)

    # Time restriction: keep events strictly before/equal to query time.
    query_dt = _parse_timestamp(t)
    if query_dt is not None:
        rel_before: list[Any] = []
        for ev in rel_history:
            _, _, _, ev_t = _event_fields(ev)
            ev_dt = _parse_timestamp(ev_t)
            if ev_dt is not None and ev_dt <= query_dt:
                rel_before.append(ev)
        rel_history = rel_before

    # Short-term: last l events
    l = int(os.environ.get("SHORT_TERM_L", "20"))
    short_history = get_short_term(rel_history, l=l)

    # Long-term: dynamic threshold filtering based on LLM scores
    # (We score the entire relation-filtered history, then keep those passing thresholds.)
    scores = compute_scores_with_llm(rel_history)
    long_filtered = filter_long_term(rel_history, scores)

    # Similar events for analogical reasoning: use last m from the long-term filtered set.
    m = int(os.environ.get("ANALOGICAL_SIMILAR_M", "3"))
    similar_events = _choose_similarity_events(long_filtered, m=m)

    # Prepare analogical reasoning text (LLM-driven)
    query_for_reasoning = (s, r, mask, t)
    analogical_reasoning = generate_analogical_reasoning(query_for_reasoning, similar_events)

    # Build candidate objects from combined histories
    candidate_objects: list[str] = []
    seen: set[str] = set()

    def _add_obj(ev: Any) -> None:
        _, _, obj, _ = _event_fields(ev)
        obj = obj.strip()
        if obj and obj not in seen and obj not in {"?", "None", "null"}:
            seen.add(obj)
            candidate_objects.append(obj)

    for ev in short_history:
        _add_obj(ev)
    for ev in long_filtered:
        _add_obj(ev)

    candidates_json = json.dumps(candidate_objects, ensure_ascii=False)

    # Verbalize histories for final prompt
    short_lines = []
    for i, ev in enumerate(short_history, start=1):
        ev_s, ev_r, ev_o, ev_t = _event_fields(ev)
        short_lines.append(f"{i}. {verbalize_event(ev_s, ev_r, ev_o, ev_t)}")

    long_lines = []
    for i, ev in enumerate(long_filtered, start=1):
        ev_s, ev_r, ev_o, ev_t = _event_fields(ev)
        long_lines.append(f"{i}. {verbalize_event(ev_s, ev_r, ev_o, ev_t)}")

    query_sentence = _verbalize_query_with_mask(s, r, t, mask=mask)

    final_prompt = (
        "You are solving a Temporal Knowledge Graph Forecasting task.\n\n"
        f"Query (missing object): {query_sentence}\n\n"
        "Candidate objects are provided. Predict the most likely object entity for the query.\n"
        f"Candidate Objects: {candidates_json}\n\n"
        "Short-term History (most recent events):\n"
        + ("\n".join(short_lines) if short_lines else "- none -")
        + "\n\n"
        "Long-term Filtered History (LLM-scored):\n"
        + ("\n".join(long_lines) if long_lines else "- none -")
        + "\n\n"
        "Analogical Reasoning (for similar events):\n"
        + analogical_reasoning
        + "\n\n"
        "Task:\n"
        "Return exactly ONE object entity string selected from Candidate Objects.\n"
        "Output ONLY the entity string (no quotes, no extra text)."
    )

    # Call predictor LLM
    predictor_spec = os.environ.get("LLM_PREDICTOR", "").strip()
    if predictor_spec:
        # prefer LLM_PREDICTOR
        os.environ["LLM_PREDICTOR"] = predictor_spec
        predictor = _load_callable_from_env("LLM_PREDICTOR")
    else:
        # fallback to LLM_GENERATOR
        predictor = _load_callable_from_env("LLM_GENERATOR")

    llm_output = predictor(final_prompt)
    return _extract_predicted_object(str(llm_output), candidate_objects)


if __name__ == "__main__":
    # Minimal dummy run for sanity.
    # Set:
    #   - LLM_GENERATOR for analogical reasoning generation
    #   - LLM_SCORER for long-term scoring
    #   - LLM_PREDICTOR for final prediction (or LLM_GENERATOR fallback)
    import sys

    if not os.environ.get("TKG_DATA_DIR"):
        # If user runs directly, point to ICEWS05-15 by default.
        os.environ["TKG_DATA_DIR"] = str(
            Path(__file__).resolve().parent.parent.parent / "data" / "ICEWS05-15"
        )

    # This is a placeholder query event.
    q = ("China", "meet", "?", "2014-01-01")
    print(predict_next_object(q))

