"""
Long-term history filtering for Temporal Knowledge Graph Forecasting.

This module implements the paper's dynamic threshold filtering step for the
Probability Distribution Calculation (PDC) / Dynamic Threshold Filtering
described in §3.2.2.

Required functions:
  - compute_scores_with_llm(history)
  - dynamic_threshold(F, delta_t, T, alpha)
  - filter_long_term(history, scores)

Notes on `compute_scores_with_llm`
----------------------------------
The original paper computes a probability p(hl) for each historical event hl
using LLM token/logit scores and softmax normalization.

This repository does not ship an LLM client wrapper, so this function expects
an external "LLM scorer" callable to be provided via an environment variable:

    LLM_SCORER="some_python_module:some_function"

The callable must have this signature:
    score_fn(prompt: str, events: list[Any]) -> list[float]

where the returned list is a set of per-event *logits* (or any real-valued
scores that can be softmax-normalized).

`compute_scores_with_llm` loads the prompt template from:
    prompts/filter_prompt.txt
relative to the repository root.
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

from Code.common import event_fields, parse_timestamp


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
    repo_root = Path(__file__).resolve().parent.parent.parent
    prompt_path = repo_root / "prompts" / "filter_prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Missing prompt template: {prompt_path}. "
            "Create prompts/filter_prompt.txt or update the module to point to your prompt."
        )
    return prompt_path.read_text(encoding="utf-8")


def _load_llm_scorer_from_env() -> Any:
    spec = os.environ.get("LLM_SCORER", "").strip()
    if not spec:
        # Default to cloud adapter which uses `Code.llm.call_llm`.
        # On Colab, configure LLM_PROVIDER / OPENAI_* / GROQ_* env vars.
        spec = "Code.llm.cloud_adapter:score_fn"

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
    We generate a sentence via `verbalize_event` then convert it to a question.
    """
    from Code.preprocessing import verbalize_event

    s, r, o, t = _extract_event_fields(query_event)
    # Force mask for safety
    masked_sentence = verbalize_event(s, r, "?", t).strip()
    if masked_sentence.endswith("."):
        masked_sentence = masked_sentence[:-1]

    # Heuristic question conversion depending on connector near the mask.
    # Prefer "whom" because objects are entities in TKG.
    lower = masked_sentence.lower()
    if lower.endswith(" with ?"):
        return masked_sentence[:-2] + "whom?"
    if lower.endswith(" to ?"):
        return masked_sentence[:-2] + "whom?"
    if lower.endswith(" by ?"):
        return masked_sentence[:-2] + "whom?"
    if lower.endswith(" from ?"):
        return masked_sentence[:-2] + "whom?"
    if lower.endswith(" against ?"):
        return masked_sentence[:-2] + "whom?"
    if lower.endswith(" about ?"):
        return masked_sentence[:-2] + "what?"

    # Generic fallback.
    if masked_sentence.endswith(" ?"):
        return masked_sentence[:-2] + "whom?"
    return masked_sentence + " Whom?"


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

