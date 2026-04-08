"""
Final prediction module for Temporal Knowledge Graph Forecasting.

Implements the complete AnRe pipeline (Algorithm 1, Tang et al., ACL 2025):

  Require: Entity Set V, History Events Hn, LLM, Query q = (sq, rq, ?, tn+1)
  Ensure: Object Entity Prediction oq
  
  1. V' ← Clustering(V)
  2. X ← ClusterRetriever(V', sq)
  3. A, P ← ∅, ∅
  4-16. For each si in X: build histories and find similar events
  17. For similar events: construct analogical examples
  18-21. Generate analysis processes
  22. oq ← Infer(LLM(P, Hq, q, Oq))

Key components:
  - Semantic-driven Historical Clustering (§3.1)
  - Candidate History Filter (§3.1)
  - Dual History Extraction (§3.2)
  - Analogical Replay (§3.3)

Paper §3.3 Prediction Method:
  "We map each candidate entity to a numerical token, obtain the
  corresponding logarithmic output La from the LLM, and convert it
  into a normalized probability using the softmax function, resulting
  in the probability distribution of each candidate answer. We sort
  the probability results and select the highest probability result
  as the final prediction."
"""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from common import event_fields, parse_timestamp


def _load_callable_from_env(var_name: str) -> Callable[[str], str]:
    """Load a callable from environment variable specification."""
    spec = os.environ.get(var_name, "").strip()
    if not spec:
        if var_name == "LLM_PREDICTOR":
            spec = "llm.cloud_adapter:predict_fn"
        elif var_name == "LLM_GENERATOR":
            spec = "llm.cloud_adapter:generate_fn"
        elif var_name == "LLM_PREDICTOR_LOGPROBS":
            spec = "llm.cloud_adapter:predict_with_logprobs_fn"
        else:
            raise EnvironmentError(
                f"{var_name} is not set and no default is configured."
            )
    if ":" not in spec:
        raise ValueError(f"{var_name} must be in format 'module_path:function_name'.")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"{var_name} resolved to non-callable: {spec}")
    return fn


def _env_truthy(name: str, default: bool = False) -> bool:
    """Check if environment variable is truthy."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _llm_provider() -> str:
    """Return normalized LLM provider name."""
    return (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()


def _should_use_logprob_prediction() -> bool:
    """Decide whether to use logprob-based prediction.

    - If `USE_LOGPROB_PREDICTION` is explicitly set, honor it.
    - Otherwise default to:
      - True for cloud APIs (OpenAI/Groq), matching paper §3.3
      - False for local HF on Colab to avoid very slow per-candidate forwards
    """
    raw = os.environ.get("USE_LOGPROB_PREDICTION", "").strip()
    if raw:
        return raw.lower() in ("1", "true", "yes", "on")
    return _llm_provider() in ("openai", "groq")


def _max_logprob_candidates_default() -> int:
    """Provider-aware default candidate cap for logprob path."""
    # Paper §3.3 expects probability ranking across Oq by candidate indices.
    # A too-small cap silently falls back to generation+parsing for large Oq,
    # which makes Hit@k less faithful to the paper. Keep a practical default
    # cap of 512 for all providers; override via MAX_LOGPROB_CANDIDATES if needed.
    return 512


def _max_logprob_candidates() -> int:
    raw = os.environ.get("MAX_LOGPROB_CANDIDATES", "").strip()
    if raw.isdigit():
        return int(raw)
    return _max_logprob_candidates_default()


def _default_train_data_dir() -> str | None:
    """If `data/ICEWS05-15` exists under the repo root, use it as a smoke-test default."""
    candidate = Path(__file__).resolve().parents[1] / "data" / "ICEWS05-15"
    return str(candidate) if candidate.is_dir() else None


def _load_history_data(query_event: Any) -> list[Any]:
    """Load TKG training data for history retrieval."""
    if hasattr(query_event, "data"):
        return list(query_event.data)
    if isinstance(query_event, dict) and "data" in query_event:
        return list(query_event["data"])

    data_dir = os.environ.get("TKG_DATA_DIR", "").strip()
    if not data_dir:
        data_dir = (_default_train_data_dir() or "").strip()
    if not data_dir:
        raise EnvironmentError(
            "No data provided. Pass query_event.data, set TKG_DATA_DIR, or place "
            "dataset at data/ICEWS05-15 under the project root."
        )

    from preprocessing import load_dataset
    return load_dataset(data_dir, splits=["train"])


def _verbalize_query_masked(s: str, r: str, t: str) -> str:
    """Create a masked query sentence for prediction."""
    from preprocessing import verbalize_event
    sentence = verbalize_event(s, r, "?", t)
    if sentence.endswith("."):
        sentence = sentence[:-1] + "?"
    return sentence


def _extract_predicted_object(llm_output: str, candidates: Sequence[str]) -> str:
    """Extract the predicted entity from LLM output.

    Paper §3.3: the model should output the **index** (1..|Oq|) of the answer.
    Prefer parsing that index before any substring heuristic — chatty models
    often mention several candidate names in the rationale; longest-substring
    then picks the wrong entity unless index lines are handled first.
    """
    out = llm_output.strip()
    if not candidates:
        return out

    cand_list = [c for c in candidates if c]
    n = len(cand_list)
    if n == 0:
        return out

    def _index_from_match(m: re.Match[str]) -> int | None:
        idx = int(m.group(1))
        if 1 <= idx <= n:
            return idx
        return None

    lines = [ln.strip() for ln in out.splitlines()]
    non_empty = [ln for ln in lines if ln]

    # 1) Bottom-up: standalone index (models often put the final answer on the last line)
    for line in reversed(non_empty):
        m = re.fullmatch(r"(\d{1,8})\s*\.?", line)
        if m:
            ix = _index_from_match(m)
            if ix is not None:
                return cand_list[ix - 1]
        m = re.search(
            r"(?:^|\b)(?:answer|choice|final|index|prediction)\s*[:=#]?\s*(\d{1,8})\s*$",
            line,
            re.IGNORECASE,
        )
        if m:
            ix = _index_from_match(m)
            if ix is not None:
                return cand_list[ix - 1]

    # 2) Tail of output: "... answer is 1234" / "[1234]"
    tail = out[-800:] if len(out) > 800 else out
    for m in re.finditer(
        r"(?:answer|choice|index|prediction)\s+(?:is|[:=#])\s*(\d{1,8})\b",
        tail,
        re.IGNORECASE,
    ):
        ix = _index_from_match(m)
        if ix is not None:
            return cand_list[ix - 1]
    m = re.search(r"\[(\d{1,8})\]", tail)
    if m:
        ix = _index_from_match(m)
        if ix is not None:
            return cand_list[ix - 1]

    # 3) Any line: explicit "Answer: 17" (not only last line)
    for line in non_empty:
        m2 = re.search(
            r"(?:choice|answer|number)\s*[:is.]+\s*(\d{1,8})\b",
            line,
            re.IGNORECASE,
        )
        if m2:
            ix = _index_from_match(m2)
            if ix is not None:
                return cand_list[ix - 1]

    # 4) Leading number on a line — skip numbered-list lines like "1. Some text"
    for line in non_empty:
        if re.match(r"^\d{1,3}\.\s+\S", line):
            continue
        m = re.match(r"^(\d{1,8})\b", line)
        if m:
            ix = _index_from_match(m)
            if ix is not None:
                return cand_list[ix - 1]

    # 5) Exact first line == entity name
    first_line = non_empty[0] if non_empty else ""
    for c in cand_list:
        if first_line == c:
            return c

    # 6) JSON
    try:
        maybe = json.loads(out)
        if isinstance(maybe, str) and maybe in cand_list:
            return maybe
        if isinstance(maybe, int) and 1 <= maybe <= n:
            return cand_list[maybe - 1]
    except Exception:
        pass

    # 7) Longest substring match (last resort — can be wrong if rationale cites many entities)
    best = ""
    for c in sorted(cand_list, key=len, reverse=True):
        if c in out and len(c) > len(best):
            best = c
    if best:
        return best

    # 8) Single-token junk (e.g. ")" from truncation) — do not return as entity
    fl = (first_line or out).strip()
    if len(fl) <= 2 and fl and not any(fl == c for c in cand_list):
        return ""

    return first_line or out


@dataclass
class _PredictionContext:
    """Internal context for prediction pipeline."""
    candidate_set: list[str]
    final_prompt: str
    used_second_order_neighbors: bool = False


def _prepare_prediction_context(
    query_event: Any,
    cluster_result,
    use_second_order_candidates: bool,
) -> _PredictionContext:
    """Prepare all components for prediction (shared logic).
    
    Implements Algorithm 1 steps 1-21 from the AnRe paper.
    """
    from preprocessing import verbalize_event
    from history import get_entity_history
    from long_term import extract_dual_history, combine_dual_history
    from clustering.entity_cluster import cluster_entities, extract_entities
    from clustering.candidate_filter import (
        find_similar_events_from_cluster,
        build_candidate_set,
        build_candidate_set_second_order,
        build_candidate_set_adaptive,
    )
    from analogical import (
        construct_analogical_examples_batch,
        format_analogical_examples_for_prompt,
    )

    s, r, o, t = event_fields(query_event)
    sq, rq = s.strip(), r.strip()

    if _env_truthy("USE_SECOND_ORDER_CANDIDATES", False):
        use_second_order_candidates = True

    data = _load_history_data(query_event)
    
    if cluster_result is None:
        entities = extract_entities(data)
        cluster_result = cluster_entities(entities)
    
    try:
        cluster_X = cluster_result.get_cluster_of(sq)
    except KeyError:
        cluster_X = [sq]
    
    entity_history_q = get_entity_history(sq, data)
    
    query_dt = parse_timestamp(t)
    if query_dt is not None:
        entity_history_q = [
            ev for ev in entity_history_q
            if (parse_timestamp(event_fields(ev)[3]) or datetime.min) < query_dt
        ]
    
    l = int(os.environ.get("SHORT_TERM_L", "20"))
    L = int(os.environ.get("HISTORY_LENGTH_L", "100"))
    a = int(os.environ.get("NUM_ANALOGICAL_EXAMPLES", "1"))
    dtf_alpha = float(os.environ.get("DTF_ALPHA", "2.75"))
    
    masked_query = (sq, rq, "?", t)
    short_term_q, long_term_q = extract_dual_history(
        full_history=entity_history_q,
        query_event=masked_query,
        l=l,
        L=L,
    )
    history_q = combine_dual_history(short_term_q, long_term_q)
    
    used_o2 = False
    if use_second_order_candidates:
        candidate_set = build_candidate_set_second_order(entity_history_q, sq, data)
        used_o2 = True
    elif _env_truthy("ADAPTIVE_CANDIDATES", default=False):
        min_c = int(os.environ.get("ADAPTIVE_MIN_CANDIDATES", "3"))
        candidate_set, used_o2 = build_candidate_set_adaptive(
            entity_history_q, sq, data, min_first_order=min_c
        )
    else:
        candidate_set = build_candidate_set(entity_history_q, sq)
    
    similar_candidates = find_similar_events_from_cluster(
        query_event=masked_query,
        cluster_entities=cluster_X,
        all_data=data,
        top_a=a,
        min_contexts=int(os.environ.get("MIN_HISTORY_CONTEXTS", "300")),
        min_history_length=L,
        short_term_l=l,
        dual_history_target_L=L,
        dtf_alpha=dtf_alpha,
    )
    
    if similar_candidates:
        analogical_examples = construct_analogical_examples_batch(similar_candidates)
        analogical_text = format_analogical_examples_for_prompt(analogical_examples)
    else:
        analogical_text = "- No analogical examples available -"
    
    history_lines = [
        verbalize_event(*event_fields(ev)[:4]) for ev in history_q
    ]
    history_text = "\n".join(history_lines) if history_lines else "- none -"
    
    query_sentence = _verbalize_query_masked(sq, rq, t)
    
    candidates_numbered = [f"{i}. {c}" for i, c in enumerate(candidate_set, start=1)]
    candidates_text = "\n".join(candidates_numbered) if candidates_numbered else "- none -"
    
    code_root = Path(__file__).resolve().parents[1]
    prompt_path = code_root / "prompts" / "prediction_prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing prediction prompt: {prompt_path}")
    prediction_template = prompt_path.read_text(encoding="utf-8")
    
    final_prompt = prediction_template.format(
        analogical_examples=analogical_text,
        history=history_text,
        query=query_sentence,
        candidates=candidates_text,
    )
    
    return _PredictionContext(
        candidate_set=candidate_set,
        final_prompt=final_prompt,
        used_second_order_neighbors=used_o2,
    )


def get_prediction_context(
    query_event: Any,
    cluster_result=None,
    use_second_order_candidates: bool = False,
) -> _PredictionContext:
    """Return prompts and candidate list without running the LLM (for debugging / metrics)."""
    return _prepare_prediction_context(
        query_event, cluster_result, use_second_order_candidates
    )


def predict_next_object(
    query_event: Any,
    cluster_result=None,
    use_second_order_candidates: bool = False,
) -> str:
    """Predict the missing object entity for a query (s, r, ?, t).
    
    Implements Algorithm 1 from the AnRe paper.
    
    Parameters
    ----------
    query_event : Any
        Query as (subject, relation, "?", timestamp)
    cluster_result : ClusterResult, optional
        Pre-computed clustering result. If None, clustering is performed.
    use_second_order_candidates : bool
        If True, use O²q (second-order neighbors) as candidate set.
        Default False uses Oq (first-order only).
    
    Returns
    -------
    str
        The predicted object entity
    """
    ctx = _prepare_prediction_context(
        query_event, cluster_result, use_second_order_candidates
    )
    
    use_logprobs = _should_use_logprob_prediction()
    max_logprob_candidates = _max_logprob_candidates()

    if (
        use_logprobs
        and ctx.candidate_set
        and len(ctx.candidate_set) <= max_logprob_candidates
    ):
        try:
            predictor_logprobs = _load_callable_from_env("LLM_PREDICTOR_LOGPROBS")
            predicted, _ = predictor_logprobs(ctx.final_prompt, ctx.candidate_set)
            return predicted
        except Exception:
            pass
    
    predictor_spec = os.environ.get("LLM_PREDICTOR", "").strip()
    if predictor_spec:
        predictor = _load_callable_from_env("LLM_PREDICTOR")
    else:
        predictor = _load_callable_from_env("LLM_GENERATOR")

    llm_output = predictor(ctx.final_prompt)
    return _extract_predicted_object(str(llm_output), ctx.candidate_set)


@dataclass
class PredictionResult:
    """Container for prediction results with probability distribution.
    
    Paper §3.3: "We sort the probability results and select the highest
    probability result as the final prediction."
    
    This class enables Hit@k evaluation by providing ranked candidates.
    """
    predicted: str
    candidates: list[str]
    probabilities: list[float]
    
    def get_ranked_candidates(self) -> list[tuple[str, float]]:
        """Return candidates sorted by probability (highest first)."""
        ranked = sorted(
            zip(self.candidates, self.probabilities),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked
    
    def hit_at_k(self, ground_truth: str, k: int) -> bool:
        """Check if ground_truth is in top-k predictions."""
        ranked = self.get_ranked_candidates()
        top_k = [c for c, _ in ranked[:k]]
        return ground_truth in top_k


def predict_next_object_with_probs(
    query_event: Any,
    cluster_result=None,
    use_second_order_candidates: bool = False,
) -> PredictionResult:
    """Predict with full probability distribution for Hit@k evaluation.
    
    This function implements the paper's complete prediction approach
    including probability distribution computation (Paper §3.3).
    
    Returns
    -------
    PredictionResult
        Contains predicted entity, all candidates, and their probabilities
    """
    ctx = _prepare_prediction_context(
        query_event, cluster_result, use_second_order_candidates
    )
    
    max_logprob_candidates = _max_logprob_candidates()

    if ctx.candidate_set and len(ctx.candidate_set) <= max_logprob_candidates:
        try:
            predictor_logprobs = _load_callable_from_env("LLM_PREDICTOR_LOGPROBS")
            predicted, probabilities = predictor_logprobs(
                ctx.final_prompt, ctx.candidate_set
            )
            return PredictionResult(
                predicted=predicted,
                candidates=ctx.candidate_set,
                probabilities=probabilities,
            )
        except Exception:
            pass
    
    predictor = _load_callable_from_env("LLM_GENERATOR")
    llm_output = predictor(ctx.final_prompt)
    predicted = _extract_predicted_object(str(llm_output), ctx.candidate_set)
    
    probabilities = [1.0 if c == predicted else 0.0 for c in ctx.candidate_set]
    
    return PredictionResult(
        predicted=predicted,
        candidates=ctx.candidate_set,
        probabilities=probabilities,
    )


def predict_batch(
    queries: Sequence[Any],
    data: Sequence[Any] | None = None,
    use_second_order_candidates: bool = False,
) -> list[str]:
    """Predict object entities for multiple queries efficiently.
    
    Pre-computes clustering once for all queries.
    
    Parameters
    ----------
    queries : Sequence[Any]
        List of query events (s, r, ?, t)
    data : Sequence[Any], optional
        TKG training data. If None, loaded from TKG_DATA_DIR.
    use_second_order_candidates : bool
        Use O²q candidate set (default False)
    
    Returns
    -------
    list[str]
        Predicted object entities for each query
    """
    from clustering.entity_cluster import cluster_entities, extract_entities
    
    if data is None:
        data = _load_history_data(queries[0])
    
    entities = extract_entities(data)
    cluster_result = cluster_entities(entities)
    
    predictions = []
    for q in queries:
        pred = predict_next_object(
            query_event=q,
            cluster_result=cluster_result,
            use_second_order_candidates=use_second_order_candidates,
        )
        predictions.append(pred)
    
    return predictions


def _apply_dummy_llm_env_if_no_api_keys() -> None:
    """Use dummy LLM callables when no API key is set."""
    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_groq = bool(os.environ.get("GROQ_API_KEY", "").strip())
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    uses_local_hf = provider in ("hf", "huggingface", "local", "transformers")
    
    if has_openai or has_groq or uses_local_hf:
        return
    
    if not os.environ.get("LLM_GENERATOR", "").strip():
        os.environ["LLM_GENERATOR"] = "analogical.dummy_generator:generate_fn"
    if not os.environ.get("LLM_SCORER", "").strip():
        os.environ["LLM_SCORER"] = "long_term.dummy_scorer:score_fn"
    if not os.environ.get("LLM_PREDICTOR", "").strip():
        os.environ["LLM_PREDICTOR"] = "inference.dummy_predictor:predict_fn"


if __name__ == "__main__":
    import sys
    
    code_root = Path(__file__).resolve().parents[1]
    if not os.environ.get("TKG_DATA_DIR"):
        os.environ["TKG_DATA_DIR"] = str(code_root / "data" / "ICEWS05-15")
    
    os.environ.setdefault("MIN_HISTORY_CONTEXTS", "0")
    
    _apply_dummy_llm_env_if_no_api_keys()
    
    from preprocessing import load_dataset
    
    data_dir = os.environ["TKG_DATA_DIR"]
    valid_quads = load_dataset(data_dir, splits=["valid"])
    test_quads = load_dataset(data_dir, splits=["test"])
    train_quads = load_dataset(data_dir, splits=["train"])
    
    query_quad = valid_quads[0] if valid_quads else (test_quads[0] if test_quads else train_quads[0])
    q = (query_quad.subject, query_quad.relation, "?", query_quad.timestamp)
    
    print(f"Query: {q}")
    print(f"Ground truth: {query_quad.object}")
    
    prediction = predict_next_object(q)
    print(f"Prediction: {prediction}")
