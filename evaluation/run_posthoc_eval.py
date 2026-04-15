"""
Posthoc metric evaluation from saved per-query jsonl records.

Usage:
  1) Generate records during inference:
       set EVAL_OUTPUT_PATH=outputs/predictions.jsonl
       python -c "from colab_setup import test_prediction_metrics; print(test_prediction_metrics())"
  2) Re-evaluate with a chosen filter:
       set EVAL_RECORDS_PATH=outputs/predictions.jsonl
       set EVAL_FILTER=time-aware
       python -m evaluation.run_posthoc_eval
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from evaluation.eval_filters import (
    build_filter_index,
    compute_rank,
    filter_ranked_predictions,
    normalize_filter_mode,
)
from preprocessing import load_dataset


def _load_records(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"EVAL_RECORDS_PATH not found: {p}")
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    records_path = os.environ.get("EVAL_RECORDS_PATH", "").strip()
    if not records_path:
        raise EnvironmentError("Set EVAL_RECORDS_PATH to a predictions jsonl file.")

    eval_filter = normalize_filter_mode(os.environ.get("EVAL_FILTER", "none"))
    data_dir = os.environ.get("TKG_DATA_DIR", "data/ICEWS05-15")
    train_data = load_dataset(data_dir, splits=["train"])
    valid_data = load_dataset(data_dir, splits=["valid"])
    test_data = load_dataset(data_dir, splits=["test"])
    filter_index = build_filter_index(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        mode=eval_filter,
    )

    rows = _load_records(records_path)
    evaluated = 0
    skipped = 0
    hits1 = 0
    hits10 = 0

    for row in rows:
        if not bool(row.get("in_oq", False)):
            skipped += 1
            continue

        q = row.get("query") or {}
        s = str(q.get("subject", "")).strip()
        r = str(q.get("relation", "")).strip()
        t = str(q.get("timestamp", "")).strip()
        gt = str(row.get("ground_truth", "")).strip()

        raw = row.get("raw") or {}
        ranked = raw.get("ranked_candidates") or raw.get("top10") or []
        ranked_pairs = [(str(c).strip(), 0.0) for c in ranked if str(c).strip()]
        filtered = filter_ranked_predictions(
            query=(s, r, "?", t),
            ground_truth=gt,
            ranked_candidates=ranked_pairs,
            index=filter_index,
            mode=eval_filter,
        )
        rank = compute_rank(filtered, gt)
        evaluated += 1
        if rank == 1:
            hits1 += 1
        if rank is not None and rank <= 10:
            hits10 += 1

    hit1 = (hits1 / evaluated) if evaluated else 0.0
    hit10 = (hits10 / evaluated) if evaluated else 0.0

    print("eval_filter,records,evaluated,skipped,hit@1,hit@10")
    print(f"{eval_filter},{len(rows)},{evaluated},{skipped},{hit1:.4f},{hit10:.4f}")


if __name__ == "__main__":
    main()

