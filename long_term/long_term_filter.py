"""
Long-term history filtering for Temporal Knowledge Graph Forecasting.

This module implements the paper's Dual History Extraction (§3.2):
  - Short-term history retriever: get last l events
  - Long-term history retriever: PDC + DTF filtering

Key Algorithm (§3.2.2):
  1. HL = Hi - HS (subtract short-term from full history first!)
  2. Partition HL by timestamp into time-step groups H^{tj}
  3. For each time-step, use PDC to score events with LLM
  4. Apply DTF to filter events with p(hl) >= threshold c_j
  5. Retrieve backwards from tli-1 until sufficient length L

Required functions:
  - compute_scores_with_llm(history, query_event)
  - dynamic_threshold(F, delta_t, T, alpha)
  - filter_long_term(history, scores)
  - extract_dual_history(full_history, query_event, l, L)

Reference: Tang et al., ACL 2025, Algorithm 1 lines 7-9, 13-15
"""

from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from common import event_fields, parse_timestamp


def _extract_event_fields(event: Any) -> tuple[str, str, str, str]:
    # Delegate to the shared helper for standard Quadruple / tuple events.
    return event_fields(event)


def _softmax(logits: Sequence[float]) -> list[float]:
    # Softmax via log-sum-exp for numerical stability.
    max_logit = max(logits) if logits else 0.0
    exps = [math.exp(x - max_logit) for x in logits]
    denom = sum(exps) if exps else 0.0
    if denom == 0.0:
        return [0.0 for _ in logits]
    return [v / denom for v in exps]


def _load_prompt_template() -> str:
    # `Code/` is the project root; prompts live under `Code/prompts/`.
    code_root = Path(__file__).resolve().parents[1]
    prompt_path = code_root / "prompts" / "filter_prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Missing prompt template: {prompt_path}. "
            "Create prompts/filter_prompt.txt or update the module to point to your prompt."
        )
    return prompt_path.read_text(encoding="utf-8")


def _load_llm_scorer_from_env() -> Any:
    spec = os.environ.get("LLM_SCORER", "").strip()
    if not spec:
        # Default to cloud adapter which uses `llm.call_llm`.
        # On Colab, configure LLM_PROVIDER / OPENAI_* / GROQ_* env vars.
        spec = "llm.cloud_adapter:score_fn"

    if ":" not in spec:
        raise ValueError("LLM_SCORER must be in format 'module_path:function_name'.")

    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"LLM_SCORER resolved to non-callable: {spec}")
    return fn


def _make_question_from_query_event(query_event: Any) -> str:
    """
    Create a natural-language question like the paper's Figure 8.

    Input query_event should represent a masked query (s, r, ?, t).
    We generate a sentence via `preprocessing.verbalize_event` then convert it to a question.
    """
    from preprocessing import verbalize_event

    s, r, o, t = _extract_event_fields(query_event)
    # Force mask for safety
    masked_sentence = verbalize_event(s, r, "?", t).strip()
    if masked_sentence.endswith("."):
        masked_sentence = masked_sentence[:-1]

    # Convert the tail "... ?" into "... whom?" (or "... what?").
    # We avoid slice-based string replacement (which can drop the needed
    # whitespace, producing tokens like "endorsedwhom?").
    import re

    lower = masked_sentence.lower()
    if lower.endswith(" about ?"):
        return re.sub(r"\s*\?\s*$", " what?", masked_sentence, flags=re.I)

    # Default: objects are entities in TKG => "whom?"
    return re.sub(r"\s*\?\s*$", " whom?", masked_sentence, flags=re.I)


def compute_scores_with_llm(history: Sequence[Any], query_event: Any) -> list[float]:
    """Compute per-event effectiveness *logits/scores* using an LLM scorer.

    Parameters
    ----------
    history:
        Sequence of events. Each event must be Quadruple-like or a 4-tuple/list.
    query_event:
        Masked query event (s, r, ?, t) used to form the Question in the prompt,
        matching the paper's PDC setup (Figure 8 / Table 7).

    Returns
    -------
    logits:
        List of real-valued logits/scores aligned with the order of `history`.

    Notes
    -----
    The paper normalizes these scores with a softmax *within each time-step
    group* H^{t_j} when computing p(hl). Therefore this function returns raw
    logits; :func:`filter_long_term` performs the per-group softmax.
    """
    if not history:
        return []

    prompt_template = _load_prompt_template()
    score_fn = _load_llm_scorer_from_env()

    labeled_events = []
    for i, ev in enumerate(history, start=1):
        s, r, o, t = _extract_event_fields(ev)
        labeled_events.append(f"{i}. ({s}, {r}, {o}, {t})")

    # The template is user-defined; support common placeholders.
    # Following the paper, we build a natural-language question from the masked query.
    labeled_history = "\n".join(labeled_events)
    query_text = _make_question_from_query_event(query_event)
    prompt = prompt_template.format(
        history=labeled_history,
        events=labeled_history,
        query=query_text,
        n=len(history),
    )

    # The scorer returns logits/scores for each event in the same order.
    logits = score_fn(prompt, list(history))
    expected = len(history)
    if len(logits) < expected:
        raise ValueError(
            f"LLM scorer returned {len(logits)} scores but history has {expected} events."
        )
    if len(logits) > expected:
        # Be robust to models that output extra values.
        logits = list(logits)[:expected]

    return [float(x) for x in logits]


def dynamic_threshold(F: int, delta_t: float, T: float, alpha: float) -> float:
    """Dynamic Threshold Filtering confidence threshold.

    Follows the paper's formula (Eq. 2):
      c_j = 1/F + (1 - 1/F) * ( (Delta_t) / T )^alpha

    Parameters
    ----------
    F : int
        Number of events in the historical set at time step j.
    delta_t : float
        Time difference between query time and this time step.
    T : float
        Total time difference between query time and the earliest time.
    alpha : float
        Variation factor controlling the growth rate.
    """
    if F <= 0:
        raise ValueError("F must be > 0")
    if T <= 0:
        # Degenerate case; use the lower bound 1/F.
        return 1.0 / float(F)

    base = 1.0 / float(F)
    ratio = float(delta_t) / float(T)
    # Paper intends non-negative ratio; clamp to avoid negative power issues.
    ratio = max(ratio, 0.0)
    return base + (1.0 - base) * (ratio**alpha)


def filter_long_term(
    history: Sequence[Any],
    scores: Sequence[float],
    *,
    query_time: str | None = None,
) -> list[Any]:
    """Filter long-term history events using dynamic thresholds.

    Because the paper partitions history into time steps {t_j}, we group
    events in `history` by their exact parsed timestamp. For each timestamp
    group, we compute its dynamic threshold c_j and keep events with
    p(hl) >= c_j, where p(hl) is the softmax-normalized probability over
    logits *within the same time-step group*.

    The returned list keeps chronological order (oldest -> newest).
    """
    if not history:
        return []
    if len(history) != len(scores):
        raise ValueError("history and scores must have the same length.")

    # Parse timestamps for ordering and for Δt/T computations.
    parsed: list[tuple[int, Any, float | None]] = []
    for idx, ev in enumerate(history):
        _, _, _, ts = event_fields(ev)
        dt = parse_timestamp(ts)
        parsed.append((idx, ev, dt.timestamp() if dt is not None else None))

    # Keep only parseable timestamps for dynamic threshold computation.
    # If nothing is parseable, return history unchanged.
    ts_values = [p[2] for p in parsed if p[2] is not None]
    if not ts_values:
        return list(history)

    # Use explicit query time t_q when provided; otherwise fall back to the
    # latest timestamp in history (backwards compatible but less faithful).
    if query_time is not None:
        q_dt = parse_timestamp(query_time)
        if q_dt is None:
            # If query_time is malformed, degrade gracefully to history-based t_q.
            t_q = max(ts_values)
        else:
            t_q = q_dt.timestamp()
    else:
        t_q = max(ts_values)

    t_min = min(ts_values)
    T = t_q - t_min
    if T <= 0:
        return list(history)

    # Group events by their timestamp seconds.
    groups: dict[float, list[int]] = {}
    for idx, _, sec in parsed:
        if sec is None:
            continue
        groups.setdefault(sec, []).append(idx)

    # Determine threshold per group.
    alpha = 2.75
    thresholds: dict[float, float] = {}
    for sec, idxs in groups.items():
        F = len(idxs)
        delta_t = t_q - sec
        thresholds[sec] = dynamic_threshold(F=F, delta_t=delta_t, T=T, alpha=alpha)

    # Filter by comparing each event's within-group softmax probability
    # with its group's threshold.
    keep_by_idx = set()
    for sec, idxs in groups.items():
        c = thresholds[sec]
        group_logits = [float(scores[idx]) for idx in idxs]
        group_probs = _softmax(group_logits)
        for idx_local, idx in enumerate(idxs):
            if group_probs[idx_local] >= c:
                keep_by_idx.add(idx)

    # Return in chronological order.
    chronological = sorted(
        [
            (i, ev, parse_timestamp(event_fields(ev)[3]) or datetime.min)
            for i, ev, _ in parsed
        ],
        key=lambda x: x[2],
    )
    return [ev for i, ev, _ in chronological if i in keep_by_idx]


def subtract_short_term(
    full_history: Sequence[Any],
    short_term: Sequence[Any],
) -> list[Any]:
    """Subtract short-term history from full history.
    
    Paper §3.2.2: "the long-term history chain is obtained from
    the set HL = Hi - HS"
    
    Returns events in full_history that are NOT in short_term.
    """
    short_set = set()
    for ev in short_term:
        s, r, o, t = event_fields(ev)
        short_set.add((s.strip(), r.strip().lower(), o.strip(), t.strip()))
    
    result = []
    for ev in full_history:
        s, r, o, t = event_fields(ev)
        key = (s.strip(), r.strip().lower(), o.strip(), t.strip())
        if key not in short_set:
            result.append(ev)
    
    return result


def extract_dual_history(
    full_history: Sequence[Any],
    query_event: Any,
    l: int = 20,
    L: int = 100,
    alpha: float = 2.75,
) -> tuple[list[Any], list[Any]]:
    """Extract both short-term and long-term history following paper Algorithm 1.
    
    Paper §3.2 Dual History Extraction:
    1. HS = get_short_term(Hi, l) - last l events
    2. HL = Hi - HS - subtract short-term first!
    3. Partition HL by timestamp, apply PDC + DTF
    4. Retrieve backwards until sufficient length L
    5. Return (HS, HL) for concatenation
    
    Parameters
    ----------
    full_history : Sequence[Any]
        Complete history Hi of the entity (NOT filtered by relation!)
    query_event : Any
        Masked query event (s, r, ?, t)
    l : int
        Short-term history length (default 20)
    L : int
        Total history length target (default 100)
    alpha : float
        DTF variation factor (default 2.75)
    
    Returns
    -------
    (short_term, long_term) : tuple[list[Any], list[Any]]
        Short-term and long-term histories, both in chronological order
    """
    from short_term import get_short_term
    
    short_term = get_short_term(full_history, l=l)
    
    long_term_pool = subtract_short_term(full_history, short_term)
    
    if not long_term_pool:
        return short_term, []
    
    scores = compute_scores_with_llm(long_term_pool, query_event)
    
    _, _, _, query_time = event_fields(query_event)
    long_term_filtered = filter_long_term(
        long_term_pool, 
        scores, 
        query_time=query_time,
    )
    
    target_long_term_len = max(0, L - len(short_term))
    
    if len(long_term_filtered) > target_long_term_len:
        long_term_filtered = long_term_filtered[-target_long_term_len:]
    
    return short_term, long_term_filtered


def combine_dual_history(
    short_term: Sequence[Any],
    long_term: Sequence[Any],
) -> list[Any]:
    """Combine short-term and long-term into final history chain.
    
    Paper §3.2.2: "Finally, we concatenate HS and HL to obtain
    the combined long-term and short-term history chain Hi."
    
    Returns events in chronological order.
    """
    from datetime import datetime
    
    all_events = list(long_term) + list(short_term)
    
    all_events.sort(
        key=lambda ev: parse_timestamp(event_fields(ev)[3]) or datetime.min
    )
    
    return all_events

