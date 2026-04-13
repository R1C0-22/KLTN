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
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common import event_fields, parse_timestamp, env_truthy, DEFAULT_EMBED_MODEL


_extract_event_fields = event_fields


def _softmax(logits: Sequence[float]) -> list[float]:
    # Softmax via log-sum-exp for numerical stability.
    max_logit = max(logits) if logits else 0.0
    exps = [math.exp(x - max_logit) for x in logits]
    denom = sum(exps) if exps else 0.0
    if denom == 0.0:
        return [0.0 for _ in logits]
    return [v / denom for v in exps]


def _cap_events_per_timestep(events: Sequence[Any]) -> list[Any]:
    """Bound LLM PDC cost when one calendar day has very many events (common on ICEWS).

    Env: ``LLM_SCORE_MAX_EVENTS_PER_TIMESTEP`` (default 64). ``0`` or negative = no cap.
    Keeps the *last* ``cap`` events in timestep order (typically most recent within the day).
    """
    ev = list(events)
    raw = os.environ.get("LLM_SCORE_MAX_EVENTS_PER_TIMESTEP", "64").strip()
    try:
        cap = int(raw)
    except ValueError:
        cap = 64
    if cap <= 0 or len(ev) <= cap:
        return ev
    return ev[-cap:]


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
    """Create a natural-language question like the paper's Figure 8."""
    from preprocessing import verbalize_masked_query

    s, r, _o, t = _extract_event_fields(query_event)
    return verbalize_masked_query(s, r, t)


def _embedding_similarity_scores(
    history_chunk: Sequence[Any],
    query_event: Any,
) -> list[float]:
    """Cosine similarity between query and each event using the shared SentenceTransformer.

    Fallback for quantized local LLMs (e.g. Llama 3 8B 4-bit) that produce
    degenerate all-zero PDC scores.  The embedding model is already loaded
    for clustering (§3.1), so this adds negligible overhead.
    """
    from clustering.shared_st import get_shared_sentence_transformer
    from preprocessing import verbalize_event

    model = get_shared_sentence_transformer(DEFAULT_EMBED_MODEL, device=None)

    qs, qr, _qo, qt = _extract_event_fields(query_event)
    query_text = verbalize_event(qs, qr, "?", qt).strip()

    event_texts = []
    for ev in history_chunk:
        s, r, o, t = _extract_event_fields(ev)
        event_texts.append(verbalize_event(s, r, o, t).strip())

    embs = model.encode(
        [query_text] + event_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sims = np.dot(embs[1:], embs[0])
    return [float(x) for x in sims]


def _compute_scores_one_chunk(
    history_chunk: Sequence[Any],
    query_event: Any,
    *,
    prompt_template: str,
    score_fn: Any,
) -> list[float]:
    """One LLM scoring call for *history_chunk* only (local indices 1..len(chunk)).

    Events are verbalized into natural language (paper Figure 8 / Table 7)
    so the LLM can assess semantic relevance to the query.

    When the LLM returns degenerate scores (all identical, common with 4-bit
    quantized models), falls back to embedding cosine similarity.
    """
    from preprocessing import verbalize_event

    labeled_events = []
    for i, ev in enumerate(history_chunk, start=1):
        s, r, o, t = _extract_event_fields(ev)
        labeled_events.append(f"{i}. {verbalize_event(s, r, o, t)}")

    labeled_history = "\n".join(labeled_events)
    query_text = _make_question_from_query_event(query_event)
    prompt = prompt_template.format(
        history=labeled_history,
        events=labeled_history,
        query=query_text,
        n=len(history_chunk),
    )

    logits = score_fn(prompt, list(history_chunk))
    expected = len(history_chunk)
    if len(logits) < expected:
        raise ValueError(
            f"LLM scorer returned {len(logits)} scores but chunk has {expected} events."
        )
    if len(logits) > expected:
        logits = list(logits)[:expected]

    if len(logits) > 1:
        mn, mx = min(logits), max(logits)
        if (mx - mn) < 1e-9:
            logits = _embedding_similarity_scores(history_chunk, query_event)

    return [float(x) for x in logits]


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

    For local / Hugging Face models with limited context (e.g. 8k), scoring
    *all* long-term events in one prompt can overflow context and produce
    invalid generations (no JSON array). We therefore score in contiguous
    chunks along ``history`` order.

    Caching is handled inside ``score_fn`` (keyed by the full prompt text),
    which automatically invalidates when the prompt template changes.

    Environment:
        LLM_SCORE_CHUNK_SIZE — max events per scoring call (default: 32).
        Set to 0 or negative to use one call for the full history (old behaviour).
    """
    if not history:
        return []

    chunk_size = int(os.environ.get("LLM_SCORE_CHUNK_SIZE", "32"))
    if chunk_size <= 0:
        chunk_size = len(history)

    prompt_template = _load_prompt_template()
    score_fn = _load_llm_scorer_from_env()

    all_logits: list[float] = []
    for start in range(0, len(history), chunk_size):
        chunk = history[start : start + chunk_size]
        logits = _compute_scores_one_chunk(
            chunk,
            query_event,
            prompt_template=prompt_template,
            score_fn=score_fn,
        )
        all_logits.extend(logits)

    return all_logits


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


def _get_date_key(dt: datetime) -> str:
    """Extract DATE (day) key from datetime for time-step grouping.
    
    Paper §3.2.2: Groups events by "time step" which in ICEWS/GDELT
    datasets corresponds to DAYS, not exact timestamps.
    """
    return dt.strftime("%Y-%m-%d")


def filter_long_term(
    history: Sequence[Any],
    scores: Sequence[float],
    *,
    query_time: str | None = None,
    alpha: float = 2.75,
) -> list[Any]:
    """Filter long-term history events using dynamic thresholds.

    Paper §3.2.2 - Dynamic Threshold Filtering (DTF):
    Partitions history into time steps {t_j} where each time step is a DAY
    (not exact timestamp). For each time-step group, we:
    1. Compute softmax probabilities within the group (PDC)
    2. Calculate dynamic threshold c_j based on time difference
    3. Keep events where p(hl) >= c_j

    The returned list keeps chronological order (oldest -> newest).
    
    Parameters
    ----------
    history : Sequence[Any]
        Long-term history events (HL = Hi - HS)
    scores : Sequence[float]
        LLM-computed effectiveness logits for each event
    query_time : str | None
        Query timestamp for computing time differences
    alpha : float
        DTF variation factor (default 2.75 per paper §6.1)
    """
    if not history:
        return []
    if len(history) != len(scores):
        raise ValueError("history and scores must have the same length.")

    # Parse timestamps for ordering and for Δt/T computations.
    parsed: list[tuple[int, Any, datetime | None, str | None]] = []
    for idx, ev in enumerate(history):
        _, _, _, ts = event_fields(ev)
        dt = parse_timestamp(ts)
        date_key = _get_date_key(dt) if dt is not None else None
        parsed.append((idx, ev, dt, date_key))

    # Keep only parseable timestamps for dynamic threshold computation.
    ts_values = [p[2] for p in parsed if p[2] is not None]
    if not ts_values:
        return list(history)

    # Use explicit query time t_q when provided; otherwise fall back to the
    # latest timestamp in history (backwards compatible but less faithful).
    if query_time is not None:
        q_dt = parse_timestamp(query_time)
        if q_dt is None:
            t_q = max(ts_values)
        else:
            t_q = q_dt
    else:
        t_q = max(ts_values)

    t_min = min(ts_values)
    
    # T = total time span in DAYS (paper uses day-level granularity)
    T_days = (t_q - t_min).days
    if T_days <= 0:
        T_days = 1  # Avoid division by zero; minimum 1 day span

    # Group events by their DATE (day), not exact timestamp.
    # Paper §3.2.2: "Partition HL into historical sets for each time step"
    groups: dict[str, list[int]] = {}
    for idx, _, dt, date_key in parsed:
        if date_key is None:
            continue
        groups.setdefault(date_key, []).append(idx)

    # Determine threshold per time-step group using paper's formula (Eq. 2).
    # c_j = 1/F + (1 - 1/F) * (Δt / T)^α
    keep_by_idx = set()
    
    for date_key, idxs in groups.items():
        # F = size of events at this time step
        F = len(idxs)
        
        # Δt = time difference between query and this time step (in days)
        group_date = datetime.strptime(date_key, "%Y-%m-%d")
        delta_t_days = (t_q - group_date).days
        if delta_t_days < 0:
            delta_t_days = 0
        
        # Calculate dynamic threshold c_j
        c_j = dynamic_threshold(F=F, delta_t=delta_t_days, T=T_days, alpha=alpha)
        
        # Compute softmax probabilities WITHIN this time-step group (PDC)
        group_logits = [float(scores[idx]) for idx in idxs]
        group_probs = _softmax(group_logits)
        
        # Keep events where p(hl) >= c_j
        for idx_local, idx in enumerate(idxs):
            if group_probs[idx_local] >= c_j:
                keep_by_idx.add(idx)

    # Return in chronological order.
    chronological = sorted(
        [(i, ev, dt or datetime.min) for i, ev, dt, _ in parsed],
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


def _partition_by_timestep(
    history: Sequence[Any],
) -> list[tuple[str, list[Any]]]:
    """Partition history into time-step groups, sorted by date (newest first).
    
    Paper §3.2.2: "We partition HL into historical sets for each time step,
    denoted as H^{tj}"
    
    Returns list of (date_key, events) tuples, sorted reverse chronologically.
    """
    groups: dict[str, list[Any]] = {}
    
    for ev in history:
        _, _, _, ts = event_fields(ev)
        dt = parse_timestamp(ts)
        if dt is not None:
            date_key = _get_date_key(dt)
            groups.setdefault(date_key, []).append(ev)
    
    # Sort by date, newest (closest to query) first
    sorted_groups = sorted(groups.items(), key=lambda x: x[0], reverse=True)
    return sorted_groups


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
    3. Partition HL by timestamp into time-step groups H^{tj}
    4. For each time step, apply PDC (LLM scoring) + DTF (dynamic threshold)
    5. Start retrieving from time step tli-1 in REVERSE chronological order
       until the long-term history length is sufficient
    6. Sort final result in chronological order
    
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

    disable_short = env_truthy("DISABLE_SHORT_TERM")
    disable_long = env_truthy("DISABLE_LONG_TERM")

    # Step 1: Get short-term history (last l events)
    short_term = [] if disable_short else get_short_term(full_history, l=l)
    
    # Step 2: HL = Hi - HS (subtract short-term from full history)
    long_term_pool = subtract_short_term(full_history, short_term)
    
    if disable_long:
        return short_term, []

    if not long_term_pool:
        return short_term, []
    
    # Target length for long-term history
    target_long_term_len = max(0, L - len(short_term))
    if target_long_term_len == 0:
        return short_term, []
    
    # Get query time for computing time differences
    _, _, _, query_time = event_fields(query_event)
    q_dt = parse_timestamp(query_time)
    if q_dt is None:
        # Fallback: use latest event time
        for ev in reversed(long_term_pool):
            dt = parse_timestamp(event_fields(ev)[3])
            if dt is not None:
                q_dt = dt
                break
    
    if q_dt is None:
        # Can't compute time differences, return empty
        return short_term, []
    
    # Find earliest timestamp in HL for computing T
    earliest_dt = None
    for ev in long_term_pool:
        dt = parse_timestamp(event_fields(ev)[3])
        if dt is not None:
            if earliest_dt is None or dt < earliest_dt:
                earliest_dt = dt
    
    if earliest_dt is None:
        return short_term, []
    
    T_days = (q_dt - earliest_dt).days
    if T_days <= 0:
        T_days = 1
    
    # Step 3: Partition HL by time-step (day), sorted reverse chronologically
    timestep_groups = _partition_by_timestep(long_term_pool)
    
    # Step 4-5: Retrieve backwards from tli-1 until sufficient length
    # Paper: "We start retrieving from time step tli-1 in reverse chronological
    #         order until the long-term history length is sufficient"
    long_term_selected: list[Any] = []

    # Each timestep (calendar day) here triggers at least one full PDC scoring call
    # (``compute_scores_with_llm``). Dense entities (e.g. heads of state) can span
    # hundreds of days → hundreds of sequential LLM forwards on Colab (tens of minutes
    # per query). Cap iterations for interactive runs; set ``0`` for paper-faithful
    # unlimited passes (slow).
    _raw_max_ts = os.environ.get("MAX_DTF_TIMESTEP_ITERATIONS", "0").strip()
    try:
        max_dtf_timesteps = int(_raw_max_ts)
    except ValueError:
        max_dtf_timesteps = 0

    _verbose = env_truthy("LLM_VERBOSE")

    for ts_idx, (date_key, events) in enumerate(timestep_groups):
        if len(long_term_selected) >= target_long_term_len:
            break
        if max_dtf_timesteps > 0 and ts_idx >= max_dtf_timesteps:
            if _verbose:
                print(
                    f"[extract_dual_history] stopping DTF at timestep index {ts_idx} "
                    f"(MAX_DTF_TIMESTEP_ITERATIONS={max_dtf_timesteps}); "
                    f"collected {len(long_term_selected)}/{target_long_term_len} long-term events",
                    flush=True,
                )
            break
        scored_events = _cap_events_per_timestep(events)
        if _verbose:
            # Per-timestep progress is critical on Colab T4, where a single timestep
            # can take minutes due to multiple sequential LLM forwards.
            chunk = os.environ.get("LLM_SCORE_CHUNK_SIZE", "32")
            cap = os.environ.get("LLM_SCORE_MAX_EVENTS_PER_TIMESTEP", "64")
            print(
                f"[extract_dual_history] timestep={ts_idx} date={date_key} "
                f"events={len(events)} capped={len(scored_events)} "
                f"(chunk={chunk}, cap_per_day={cap}) "
                f"selected={len(long_term_selected)}/{target_long_term_len}",
                flush=True,
            )
        scores = compute_scores_with_llm(scored_events, query_event)

        F = len(scored_events)
        group_date = datetime.strptime(date_key, "%Y-%m-%d")
        delta_t_days = (q_dt - group_date).days
        if delta_t_days < 0:
            delta_t_days = 0

        c_j = dynamic_threshold(F=F, delta_t=delta_t_days, T=T_days, alpha=alpha)

        probs = _softmax(scores)

        for ev, prob in zip(scored_events, probs):
            if prob >= c_j:
                long_term_selected.append(ev)
    
    # Truncate if we have more than needed
    if len(long_term_selected) > target_long_term_len:
        long_term_selected = long_term_selected[:target_long_term_len]
    
    # Step 6: Sort in chronological order before returning
    long_term_selected.sort(
        key=lambda ev: parse_timestamp(event_fields(ev)[3]) or datetime.min
    )
    
    return short_term, long_term_selected


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

