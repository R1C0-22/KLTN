"""
Runtime benchmark for LLM cache effectiveness.

Measures end-to-end prediction runtime with cache OFF vs ON and reports
relative speedup. This supports IMPROVE.MD requirement:
"report how much inference time is reduced by caching".

Usage:
  python -m evaluation.run_cache_benchmark
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from evaluation.runtime import ensure_eval_runtime, patched_env
from inference.final_prediction import predict_next_object
from preprocessing import load_dataset


def _ensure_query() -> tuple[str, str, str, str]:
    data_dir = os.environ.get("TKG_DATA_DIR", "data/ICEWS05-15")
    valid = load_dataset(data_dir, splits=["valid"])
    if not valid:
        raise RuntimeError(f"No valid split found in {data_dir}")
    q = valid[0]
    return (q.subject, q.relation, "?", q.timestamp)


def _run_once(query: tuple[str, str, str, str]) -> tuple[str, float]:
    t0 = time.perf_counter()
    pred = predict_next_object(query)
    dt = time.perf_counter() - t0
    return pred, dt


def main() -> None:
    ensure_eval_runtime()
    cache_dir = os.environ.get("LLM_CACHE_DIR", ".llm_cache")
    os.environ["LLM_CACHE_DIR"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    query = _ensure_query()

    with patched_env(
        {
            "LLM_CACHE_PREDICT": "0",
            "LLM_CACHE_LOGPROBS": "0",
            "LLM_SCORE_PARSE_FALLBACK": os.environ.get("LLM_SCORE_PARSE_FALLBACK", "1"),
        }
    ):
        pred_off, t_off = _run_once(query)

    with patched_env(
        {
            "LLM_CACHE_PREDICT": "1",
            "LLM_CACHE_LOGPROBS": "1",
        }
    ):
        # Warm up cache
        _run_once(query)
        # Timed cached run
        pred_on, t_on = _run_once(query)

    speedup = (t_off / t_on) if t_on > 0 else 0.0
    reduction = (1.0 - (t_on / t_off)) * 100.0 if t_off > 0 else 0.0

    print("mode,prediction,time_sec")
    print(f"cache_off,{pred_off},{t_off:.4f}")
    print(f"cache_on,{pred_on},{t_on:.4f}")
    print(f"speedup_x,{speedup:.4f}")
    print(f"time_reduction_percent,{reduction:.2f}")


if __name__ == "__main__":
    main()

