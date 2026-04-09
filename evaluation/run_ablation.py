"""
Ablation runner for AnRe-style pipeline.

Runs mini ablations required in IMPROVE.MD:
  - w/o long-term
  - w/o short-term
  - w/o analogical

Usage:
  python -m evaluation.run_ablation
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from colab_setup import test_prediction_metrics


@contextmanager
def _patched_env(patch: dict[str, str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in patch}
    try:
        for k, v in patch.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_one(name: str, env_patch: dict[str, str], n_queries: int, sample_size: int) -> dict:
    with _patched_env(env_patch):
        result = test_prediction_metrics(
            n_queries=n_queries,
            sample_size=sample_size,
            use_second_order=False,
            start_index=0,
        )
    row = {
        "setting": name,
        "hit@1": float(result["hit_at_1"]),
        "hit@10": float(result["hit_at_10"]),
        "evaluated": int(result["evaluated"]),
        "skipped": int(result["skipped_gt_not_in_oq"]),
    }
    return row


def main() -> None:
    n_queries = int(os.environ.get("ABLATION_N_QUERIES", "20"))
    sample_size = int(os.environ.get("ABLATION_CLUSTER_SAMPLE", "500"))

    # Baseline: paper-faithful features enabled.
    baseline = _run_one(
        "full",
        {
            "SHORT_TERM_L": os.environ.get("SHORT_TERM_L", "20"),
            "NUM_ANALOGICAL_EXAMPLES": os.environ.get("NUM_ANALOGICAL_EXAMPLES", "1"),
        },
        n_queries=n_queries,
        sample_size=sample_size,
    )

    wo_long_term = _run_one(
        "w/o long-term",
        {
            "HISTORY_LENGTH_L": os.environ.get("SHORT_TERM_L", "20"),
            "NUM_ANALOGICAL_EXAMPLES": os.environ.get("NUM_ANALOGICAL_EXAMPLES", "1"),
        },
        n_queries=n_queries,
        sample_size=sample_size,
    )

    wo_short_term = _run_one(
        "w/o short-term",
        {
            "SHORT_TERM_L": "0",
            "NUM_ANALOGICAL_EXAMPLES": os.environ.get("NUM_ANALOGICAL_EXAMPLES", "1"),
        },
        n_queries=n_queries,
        sample_size=sample_size,
    )

    wo_analogical = _run_one(
        "w/o analogical",
        {
            "NUM_ANALOGICAL_EXAMPLES": "0",
            "SHORT_TERM_L": os.environ.get("SHORT_TERM_L", "20"),
        },
        n_queries=n_queries,
        sample_size=sample_size,
    )

    rows = [baseline, wo_long_term, wo_short_term, wo_analogical]
    print("\nAblation results")
    print("setting,hit@1,hit@10,evaluated,skipped")
    for r in rows:
        print(f"{r['setting']},{r['hit@1']:.4f},{r['hit@10']:.4f},{r['evaluated']},{r['skipped']}")


if __name__ == "__main__":
    main()

