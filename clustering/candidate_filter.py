"""
Candidate History Filter for Temporal Knowledge Graph Forecasting.

Implements §3.1 of the AnRe framework:
  - Find similar events Ei for entities in the same cluster
  - Filter events that have ≥ MIN_HISTORY_CONTEXTS historical contexts
  - Rank similar events by semantic similarity to the query
  - Build candidate answer set Oq from all entities in Hq

Reference: Algorithm 1 lines 4-16 and Section 3.1 of Tang et al., ACL 2025
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from common import DEFAULT_EMBED_MODEL, event_fields, parse_timestamp

logger = logging.getLogger(__name__)

_parse_timestamp = parse_timestamp
_datetime_min = datetime.min

MIN_HISTORY_CONTEXTS = 300


@dataclass
class SimilarEventCandidate:
    """Similar event ei with dual-history chain Hai (§3.2) for analogical replay."""
    entity: str
    event: Any
    history: list[Any]
    similarity_score: float


def get_similar_events_for_entity(
    entity: str,
    relation: str,
    all_events: Sequence[Any],
    query_time: str | None = None,
) -> list[Any]:
    """Find events where entity si has the same relation rq.
    
    Paper §3.1: Ei = {(si, r, o, t) ∈ TKG_{t<n+1} | r = rq}
    
    Returns events in chronological order.
    """
    rel_norm = relation.strip().lower()
    entity_norm = entity.strip()
    matching_events = []
    
    for ev in all_events:
        s, r, o, t = event_fields(ev)
        if s.strip() == entity_norm and r.strip().lower() == rel_norm:
            if query_time is not None:
                ev_dt = _parse_timestamp(t)
                q_dt = _parse_timestamp(query_time)
                if ev_dt is not None and q_dt is not None and ev_dt >= q_dt:
                    continue
            matching_events.append(ev)
    
    matching_events.sort(
        key=lambda ev: _parse_timestamp(event_fields(ev)[3]) or _datetime_min
    )
    return matching_events


def get_entity_history_count(
    entity: str,
    all_events: Sequence[Any],
    before_time: str,
) -> int:
    """Count historical events involving entity before a given time.
    
    Paper §3.1 requires at least 300 relevant historical contexts.
    """
    before_dt = _parse_timestamp(before_time)
    if before_dt is None:
        return 0
    
    entity_norm = entity.strip()
    count = 0
    for ev in all_events:
        s, r, o, t = event_fields(ev)
        ev_dt = _parse_timestamp(t)
        if ev_dt is None or ev_dt >= before_dt:
            continue
        if s.strip() == entity_norm or o.strip() == entity_norm:
            count += 1
    
    return count


def filter_events_by_history_requirement(
    events: Sequence[Any],
    all_data: Sequence[Any],
    min_contexts: int = MIN_HISTORY_CONTEXTS,
) -> list[Any]:
    """Filter similar events by requiring at least min_contexts history.
    
    Paper §3.1: "we filter Ei by requiring at least 300 relevant
    historical contexts before the event timestamp."
    """
    filtered = []
    for ev in events:
        s, r, o, t = event_fields(ev)
        count = get_entity_history_count(s, all_data, t)
        if count >= min_contexts:
            filtered.append(ev)
    
    return filtered


def rank_events_by_similarity(
    events: Sequence[Any],
    query_event: Any,
    model: SentenceTransformer | None = None,
) -> list[tuple[Any, float]]:
    """Rank events by semantic similarity to the query.
    
    Paper §3.1: "These events are then ranked according to the
    semantic similarity to q, and the query with the highest
    similarity is selected as the similar event ei."
    
    Query and events are embedded as verbalized sentences (masked query for q).
    
    Returns list of (event, similarity_score) sorted descending.
    """
    if not events:
        return []
    
    if model is None:
        model = SentenceTransformer(DEFAULT_EMBED_MODEL)
    
    from preprocessing import verbalize_event

    def event_to_text(ev: Any) -> str:
        s, r, o, t = event_fields(ev)
        return verbalize_event(s, r, o, t).strip()

    qs, qr, _qo, qt = event_fields(query_event)
    query_text = verbalize_event(qs, qr, "?", qt).strip()
    
    event_texts = [event_to_text(ev) for ev in events]
    all_texts = [query_text] + event_texts
    
    embeddings = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
    query_emb = embeddings[0]
    event_embs = embeddings[1:]
    
    similarities = np.dot(event_embs, query_emb)
    
    ranked = sorted(
        zip(events, similarities),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def find_similar_events_from_cluster(
    query_event: Any,
    cluster_entities: list[str],
    all_data: Sequence[Any],
    top_a: int = 1,
    min_contexts: int = MIN_HISTORY_CONTEXTS,
    min_history_length: int = 100,
    short_term_l: int = 20,
    dual_history_target_L: int = 100,
    dtf_alpha: float = 2.75,
    model: SentenceTransformer | None = None,
) -> list[SimilarEventCandidate]:
    """Find similar events from entities in the same cluster.
    
    Paper Algorithm 1, lines 4-10 and §3.2–3.3:
    For each si ∈ X where si ≠ sq:
      - Build Ei, filter ≥300 contexts, rank by similarity to q
      - Select **one** ei per si (highest similarity among those that qualify)
      - Hai = dual history (HS + HL) for (si, rq, ?, ti) per §3.2
      - Skip if |Hai| < L (§3.3)
    Globally keep the top_a candidates by similarity score.
    
    Parameters
    ----------
    query_event : Any
        The query (sq, rq, ?, tn+1)
    cluster_entities : list[str]
        Entities in the same cluster as sq
    all_data : Sequence[Any]
        All TKG events (training data)
    top_a : int
        Number of top similar events to return (default 1)
    min_contexts : int
        Minimum history contexts required for filtering Ei (default 300)
    min_history_length : int
        Minimum combined dual-history length L (default 100). Paper §3.3.
    short_term_l : int
        Short-term chain length l (default 20). Paper §6.1.
    dual_history_target_L : int
        Target total history length L for dual extraction (default 100).
    dtf_alpha : float
        Dynamic threshold factor α (default 2.75). Paper §6.1.
    model : SentenceTransformer | None
        Embedding model for similarity computation (default: BERT NLI, AnRe §3.1).
    
    Returns
    -------
    list[SimilarEventCandidate]
        Top 'a' similar event candidates with dual-history chains Hai.
    """
    from history import get_entity_history
    from long_term import combine_dual_history, extract_dual_history

    sq, rq, _, tq = event_fields(query_event)
    sq = sq.strip()
    rq = rq.strip().lower()

    if model is None:
        model = SentenceTransformer(DEFAULT_EMBED_MODEL)

    all_candidates: list[tuple[Any, float, str, list[Any]]] = []

    for si in cluster_entities:
        si = si.strip()
        if si == sq:
            continue

        similar_events = get_similar_events_for_entity(si, rq, all_data, tq)

        filtered_events = filter_events_by_history_requirement(
            similar_events, all_data, min_contexts
        )

        if not filtered_events:
            continue

        ranked = rank_events_by_similarity(filtered_events, query_event, model)

        chosen: tuple[Any, float, str, list[Any]] | None = None
        for ev, sim_score in ranked:
            raw_history = get_entity_history(si, all_data)
            ev_t = event_fields(ev)[3]
            ev_dt = _parse_timestamp(ev_t)
            if ev_dt is not None:
                raw_history = [
                    h
                    for h in raw_history
                    if (_parse_timestamp(event_fields(h)[3]) or _datetime_min) < ev_dt
                ]

            masked_i = (si, rq, "?", ev_t)
            short_s, long_s = extract_dual_history(
                raw_history,
                masked_i,
                l=short_term_l,
                L=dual_history_target_L,
                alpha=dtf_alpha,
            )
            hai = combine_dual_history(short_s, long_s)

            if len(hai) < min_history_length:
                continue

            chosen = (ev, sim_score, si, hai)
            break

        if chosen is not None:
            all_candidates.append(chosen)

    all_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = all_candidates[:top_a]

    result = []
    for ev, sim_score, entity, history in top_candidates:
        result.append(
            SimilarEventCandidate(
                entity=entity,
                event=ev,
                history=history,
                similarity_score=sim_score,
            )
        )

    return result


def build_candidate_set(
    entity_history: Sequence[Any],
    query_subject: str,
) -> list[str]:
    """Build candidate answer entity set Oq from history.
    
    Paper §3.1: "Oq = {o | (s, r, o, t) ∈ Hq, s = sq} ∪
                       {s | (s, r, o, t) ∈ Hq, o = sq}"
    
    This includes ALL entities that have interacted with sq,
    not just objects from same-relation events.
    """
    sq = query_subject.strip()
    candidates: set[str] = set()
    
    for ev in entity_history:
        s, r, o, t = event_fields(ev)
        s = s.strip()
        o = o.strip()
        
        if s == sq and o not in {"?", "None", "null", ""}:
            candidates.add(o)
        if o == sq and s not in {"?", "None", "null", ""}:
            candidates.add(s)
    
    return sorted(candidates)


def build_candidate_set_second_order(
    entity_history: Sequence[Any],
    query_subject: str,
    all_data: Sequence[Any],
) -> list[str]:
    """Build second-order candidate set O²q.
    
    Paper Table 2: "O²q denotes using the set of historical
    second-order neighbor entities of sq as the candidate set."
    
    This expands candidates to include entities that interacted
    with first-order neighbors of sq.
    """
    sq = query_subject.strip()
    
    first_order = build_candidate_set(entity_history, sq)
    
    second_order: set[str] = set(first_order)
    
    from history import get_entity_history
    
    for neighbor in first_order:
        neighbor_history = get_entity_history(neighbor, all_data)
        for ev in neighbor_history:
            s, r, o, t = event_fields(ev)
            s = s.strip()
            o = o.strip()
            if s != sq and s not in {"?", "None", "null", ""}:
                second_order.add(s)
            if o != sq and o not in {"?", "None", "null", ""}:
                second_order.add(o)
    
    return sorted(second_order)


if __name__ == "__main__":
    import os

    os.environ.setdefault("LLM_SCORER", "long_term.dummy_scorer:score_fn")

    demo_data = [
        ("USA", "meet", "China", "2014-01-01"),
        ("USA", "meet", "Mexico", "2013-12-01"),
        ("Russia", "meet", "Belarus", "2013-06-01"),
        ("Russia", "visit", "China", "2013-05-01"),
        ("France", "meet", "Germany", "2013-07-01"),
    ]
    
    query = ("USA", "meet", "?", "2014-02-01")
    cluster = ["USA", "Russia", "France"]
    
    candidates = find_similar_events_from_cluster(
        query,
        cluster,
        demo_data,
        top_a=2,
        min_contexts=0,
        min_history_length=1,
        short_term_l=2,
        dual_history_target_L=10,
    )
    
    print("Similar event candidates:")
    for c in candidates:
        print(f"  Entity: {c.entity}, Event: {c.event}, Score: {c.similarity_score:.4f}")
    
    from history import get_entity_history
    usa_history = get_entity_history("USA", demo_data)
    oq = build_candidate_set(usa_history, "USA")
    print(f"\nCandidate set Oq: {oq}")
