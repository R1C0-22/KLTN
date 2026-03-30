"""
Quick smoke-test for the preprocessing + clustering pipeline.

Run from the project root:
    python -m Code.test_pipeline                       # full test on ICEWS05-15
    python -m Code.test_pipeline --small               # fast test with synthetic data
    python -m Code.test_pipeline --data Code/data/ICEWS05-15
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────────────

def _separator(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── test with real dataset ─────────────────────────────────────────────────

def test_real(data_dir: str, max_entities: int | None = None) -> None:
    from Code.preprocessing import load_dataset, verbalize_event, build_corpus
    from Code.clustering import (
        extract_entities,
        embed_entities,
        find_optimal_k,
        run_kmeans,
        cluster_entities,
    )

    # 1. Load
    _separator("1. load_dataset")
    t0 = time.perf_counter()
    quads = load_dataset(data_dir, splits=["train"])
    print(f"  Loaded {len(quads):,} quadruples in {time.perf_counter()-t0:.2f}s")
    print(f"  First quad: {quads[0]}")

    # 2. Verbalize
    _separator("2. verbalize_event  +  build_corpus")
    q = quads[0]
    sentence = verbalize_event(q.subject, q.relation, q.object, q.timestamp)
    print(f"  Single: {sentence}")

    sample_corpus = build_corpus(quads[:5])
    for s in sample_corpus:
        print(f"  {s}")

    # 3. Extract entities
    _separator("3. extract_entities")
    all_entities = extract_entities(quads)
    print(f"  Unique entities: {len(all_entities):,}")

    if max_entities and 0 < max_entities < len(all_entities):
        # Random sample instead of alphabetical slice to avoid bias
        rng = random.Random(42)
        entities = rng.sample(all_entities, max_entities)
        # Guarantee probe entities are included if they exist
        for probe in ("China", "Barack Obama", "Japan", "Russia", "United States"):
            if probe in all_entities and probe not in entities:
                entities.append(probe)
        entities.sort()
        print(f"  (randomly sampled {len(entities)} entities for speed)")
    else:
        entities = all_entities

    print(f"  Sample: {entities[:5]}")

    # 4. Embed
    _separator("4. embed_entities")
    t0 = time.perf_counter()
    embeddings = embed_entities(entities, show_progress=True)
    print(f"  Shape: {embeddings.shape}  dtype: {embeddings.dtype}")
    print(f"  Elapsed: {time.perf_counter()-t0:.2f}s")

    # 5. Find optimal K  (let find_optimal_k pick its own k_max default)
    _separator("5. find_optimal_k")
    t0 = time.perf_counter()
    best_k, scores = find_optimal_k(embeddings, k_min=2)
    print(f"  Best K: {best_k}")
    for k, sc in sorted(scores.items()):
        marker = " <-- best" if k == best_k else ""
        print(f"    K={k:>3d}  silhouette={sc:.4f}{marker}")
    print(f"  Elapsed: {time.perf_counter()-t0:.2f}s")

    # 6. Run KMeans
    _separator("6. run_kmeans")
    labels = run_kmeans(embeddings, best_k)
    print(f"  Labels shape: {labels.shape}  unique clusters: {len(set(labels))}")

    # 7. Full pipeline — reuses the embeddings from step 4
    _separator("7. cluster_entities  (full pipeline, reusing embeddings)")
    result = cluster_entities(entities, embeddings=embeddings)
    print(result.summary())

    # 8. Query a specific entity
    _separator("8. get_cluster_of (query)")
    for probe in ("China", "Barack Obama", "Japan"):
        try:
            peers = result.get_cluster_of(probe)
            preview = ", ".join(peers[:8])
            tail = f" ... +{len(peers)-8} more" if len(peers) > 8 else ""
            print(f"  '{probe}' cluster ({len(peers)} members): {preview}{tail}")
        except KeyError:
            print(f"  '{probe}' not found in entity list")

    print("\nAll tests passed.")


# ── test with synthetic data (no dataset required) ─────────────────────────

def test_small() -> None:
    from Code.clustering import embed_entities, find_optimal_k, run_kmeans

    _separator("Small synthetic test (no dataset needed)")
    entities = [
        "United States", "China", "Japan", "Germany", "France",
        "Nigeria", "South Africa", "Brazil", "India", "Russia",
        "Barack Obama", "Xi Jinping", "Angela Merkel", "Narendra Modi",
        "Vladimir Putin", "Shinzo Abe", "Emmanuel Macron",
        "United Nations", "African Union", "European Union",
        "NATO", "Association of Southeast Asian Nations", "World Bank",
        "Military (Russia)", "Police (Nigeria)", "Israeli Defense Forces",
        "Defense / Security Ministry (United States)",
    ]
    print(f"  Entities: {len(entities)}")

    embeddings = embed_entities(entities, show_progress=False)
    print(f"  Embeddings shape: {embeddings.shape}")

    best_k, scores = find_optimal_k(embeddings, k_min=2, k_max=10)
    print(f"  Best K: {best_k}")
    for k, sc in sorted(scores.items()):
        marker = " <-- best" if k == best_k else ""
        print(f"    K={k}  silhouette={sc:.4f}{marker}")

    labels = run_kmeans(embeddings, best_k)
    for cid in sorted(set(labels)):
        members = [e for e, l in zip(entities, labels) if l == cid]
        print(f"  Cluster {cid}: {members}")

    best_sil = scores[best_k]
    if best_sil < 0.25:
        print(f"\n  [NOTE] Silhouette={best_sil:.4f} is low — expected with only"
              f" {len(entities)} short entity names.")
        print("         Run the full ICEWS test for meaningful clusters:")
        print("           python -m Code.test_pipeline --max-entities 1000")

    print("\nSmall test passed.")


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test preprocessing + clustering")
    parser.add_argument("--small", action="store_true", help="Fast test with 20 synthetic entities")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset dir (e.g. Code/data/ICEWS05-15)")
    parser.add_argument("--max-entities", type=int, default=500, help="Cap entity count for speed")
    args = parser.parse_args()

    if args.small:
        test_small()
    else:
        data_dir = args.data or str(Path(__file__).resolve().parent / "data" / "ICEWS05-15")
        test_real(data_dir, max_entities=args.max_entities)
