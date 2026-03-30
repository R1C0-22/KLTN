"""
Semantic-driven entity clustering for Temporal Knowledge Graph Forecasting.

Implements §3.1 of the AnRe framework (Tang et al., ACL 2025):
  1. Encode entity names into dense vectors with a sentence-transformer.
  2. Select the optimal number of clusters K via silhouette analysis.
  3. Run KMeans and return per-entity cluster assignments.

Dependencies
------------
    pip install sentence-transformers scikit-learn numpy
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", message=".*position_ids.*")

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Prefix gives the sentence-transformer semantic context so that bare
# names like "Xi Jinping" get placed closer to "China" than to
# "Angela Merkel".  This dramatically improves clustering quality.
_ENTITY_PROMPT_PREFIX = (
    "This is a named entity from an international political event dataset: "
)

_SILHOUETTE_WEAK = 0.25


# ---------------------------------------------------------------------------
# Public dataclass returned by the pipeline
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Container for the output of the clustering pipeline."""

    entities: list[str]
    embeddings: NDArray[np.float32]
    k: int
    labels: NDArray[np.intp]
    silhouette: float
    silhouette_scores: dict[int, float] = field(default_factory=dict)

    def get_cluster(self, cluster_id: int) -> list[str]:
        """Return all entities assigned to *cluster_id*."""
        return [e for e, lbl in zip(self.entities, self.labels) if lbl == cluster_id]

    def get_cluster_of(self, entity: str) -> list[str]:
        """Return every entity in the same cluster as *entity*."""
        try:
            idx = self.entities.index(entity)
        except ValueError:
            raise KeyError(f"Entity '{entity}' not found in the clustering result")
        return self.get_cluster(int(self.labels[idx]))

    def summary(self) -> str:
        quality = "GOOD" if self.silhouette >= _SILHOUETTE_WEAK else "WEAK"
        lines = [
            f"Entities  : {len(self.entities):,}",
            f"Optimal K : {self.k}",
            f"Silhouette: {self.silhouette:.4f}  ({quality})",
            "",
            "Cluster sizes:",
        ]
        unique, counts = np.unique(self.labels, return_counts=True)
        for cid, cnt in zip(unique, counts):
            lines.append(f"  cluster {cid:>3d}: {cnt:>6,} entities")
        if self.silhouette < _SILHOUETTE_WEAK:
            lines.append("")
            lines.append(
                f"  [!] Silhouette < {_SILHOUETTE_WEAK} — cluster structure is weak."
            )
            lines.append(
                "      This is expected with few entities. "
                "Try the full dataset for meaningful clusters."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def embed_entities(
    entity_list: Sequence[str],
    *,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 256,
    device: str | None = None,
    show_progress: bool = True,
    prompt_prefix: str | None = _ENTITY_PROMPT_PREFIX,
) -> NDArray[np.float32]:
    """Encode entity names into dense embedding vectors.

    Parameters
    ----------
    entity_list : sequence of str
        The entity names to embed.
    model_name : str
        HuggingFace sentence-transformer model identifier.
    batch_size : int
        Encoding batch size (tune for GPU memory).
    device : str or None
        ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.  ``None`` = auto-detect.
    show_progress : bool
        Show a tqdm progress bar during encoding.
    prompt_prefix : str or None
        Prepended to each entity name before encoding to give the
        sentence-transformer more semantic context.  Set to ``None``
        to embed the raw names.

    Returns
    -------
    np.ndarray of shape ``(len(entity_list), dim)`` with dtype float32.
    """
    if prompt_prefix:
        texts = [f"{prompt_prefix}{e}" for e in entity_list]
    else:
        texts = list(entity_list)

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def _find_elbow(k_values: list[int], inertias: list[float]) -> int:
    """Find the elbow/knee point using the max-distance-from-line method.

    Draws a straight line from the first point (k_min, inertia_max) to
    the last point (k_max, inertia_min) and picks the K whose inertia
    is farthest *below* that line.
    """
    p1 = np.array([k_values[0], inertias[0]], dtype=np.float64)
    p2 = np.array([k_values[-1], inertias[-1]], dtype=np.float64)
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return k_values[len(k_values) // 2]

    line_unit = line_vec / line_len
    best_dist = -1.0
    best_k = k_values[0]

    for k, inertia in zip(k_values, inertias):
        pt = np.array([k, inertia], dtype=np.float64)
        proj = np.dot(pt - p1, line_unit)
        closest = p1 + proj * line_unit
        dist = np.linalg.norm(pt - closest)
        if dist > best_dist:
            best_dist = dist
            best_k = k

    return best_k


def find_optimal_k(
    embeddings: NDArray[np.float32],
    *,
    k_min: int = 2,
    k_max: int | None = None,
    step: int = 1,
    random_state: int = 42,
    sample_limit: int = 20_000,
) -> tuple[int, dict[int, float]]:
    """Select the optimal cluster count K using elbow method + silhouette.

    Following AnRe Section 3.1, this uses *both* the elbow method (on KMeans
    inertia) and the silhouette coefficient:

    1. Run KMeans for each candidate K and record inertia + silhouette.
    2. Find the **elbow point** on the inertia curve (the K where adding
       more clusters stops reducing inertia significantly).
    3. In a neighbourhood around the elbow, pick the K with the
       **highest silhouette score**.

    Parameters
    ----------
    embeddings : ndarray (N, D)
    k_min, k_max : int
        Range of K values to evaluate.  *k_max* defaults to
        ``min(int(sqrt(N)), 100)`` when not set.
    step : int
        Step size between evaluated K values.
    random_state : int
        Seed for reproducibility.
    sample_limit : int
        If N > sample_limit, KMeans and silhouette are computed on a
        random subsample for speed.

    Returns
    -------
    (best_k, scores) where *scores* maps each tested K to its
    silhouette score.
    """
    n = len(embeddings)
    if k_max is None:
        k_max = min(int(np.sqrt(n)), 100)
    k_max = min(k_max, n - 1)
    k_min = max(k_min, 2)

    if k_min > k_max:
        logger.warning("k_min=%d > k_max=%d — returning k_min", k_min, k_max)
        return k_min, {k_min: 0.0}

    k_range = k_max - k_min
    if step == 1 and k_range > 30:
        step = max(2, k_range // 25)
        logger.info("Auto step=%d for K range [%d, %d]", step, k_min, k_max)

    use_sample = n > sample_limit
    if use_sample:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_limit, replace=False)
        sample_emb = embeddings[idx]
    else:
        sample_emb = embeddings

    k_values: list[int] = []
    inertias: list[float] = []
    scores: dict[int, float] = {}

    for k in range(k_min, k_max + 1, step):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(sample_emb)
        inertias.append(float(km.inertia_))
        k_values.append(k)

        sil = silhouette_score(
            sample_emb, labels,
            sample_size=min(5_000, len(sample_emb)),
        )
        scores[k] = float(sil)
        logger.info("K=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil)

    # --- Elbow detection on the inertia curve ---
    elbow_k = _find_elbow(k_values, inertias)
    logger.info("Elbow detected at K=%d", elbow_k)

    # --- Refine: pick best silhouette in a window around the elbow ---
    window = max(step * 3, 6)
    candidates = {
        k: s for k, s in scores.items()
        if (elbow_k - window) <= k <= (elbow_k + window)
    }
    if not candidates:
        candidates = scores

    best_k = max(candidates, key=candidates.get)  # type: ignore[arg-type]
    logger.info(
        "Best K=%d (silhouette=%.4f, elbow neighbourhood [%d, %d])",
        best_k, scores[best_k], elbow_k - window, elbow_k + window,
    )
    return best_k, scores


def run_kmeans(
    embeddings: NDArray[np.float32],
    k: int,
    *,
    random_state: int = 42,
) -> NDArray[np.intp]:
    """Run KMeans on *embeddings* with *k* clusters.

    Returns
    -------
    np.ndarray of shape ``(N,)`` — cluster label for each entity.
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(embeddings)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def cluster_entities(
    entity_list: Sequence[str],
    *,
    embeddings: NDArray[np.float32] | None = None,
    model_name: str = _DEFAULT_MODEL,
    k: int | None = None,
    k_min: int = 2,
    k_max: int | None = None,
    batch_size: int = 256,
    device: str | None = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> ClusterResult:
    """End-to-end pipeline: embed -> find K -> cluster -> return result.

    Parameters
    ----------
    entity_list : sequence of str
        Unique entity names.
    embeddings : ndarray or None
        Pre-computed embeddings (skips the encoding step).  Must have
        the same length as *entity_list*.
    model_name : str
        Sentence-transformer model to use (ignored when *embeddings*
        is provided).
    k : int or None
        If given, skip the silhouette search and use this K directly.
    k_min, k_max : int
        Search range when *k* is None.
    batch_size : int
        Encoding batch size.
    device : str or None
        Torch device for the encoder.
    random_state : int
        Seed for KMeans reproducibility.
    show_progress : bool
        Show progress bars.

    Returns
    -------
    :class:`ClusterResult`
    """
    entities = list(entity_list)

    if embeddings is not None:
        if len(embeddings) != len(entities):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) != "
                f"entity_list length ({len(entities)})"
            )
        logger.info("Using %d pre-computed embeddings", len(embeddings))
    else:
        logger.info("Embedding %d entities with '%s' ...", len(entities), model_name)
        embeddings = embed_entities(
            entities,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )

    sil_scores: dict[int, float] = {}
    if k is None:
        k, sil_scores = find_optimal_k(
            embeddings,
            k_min=k_min,
            k_max=k_max,
            random_state=random_state,
        )

    logger.info("Running KMeans with K=%d ...", k)
    labels = run_kmeans(embeddings, k, random_state=random_state)
    sil = float(silhouette_score(
        embeddings, labels,
        sample_size=min(5_000, len(embeddings)),
    ))

    return ClusterResult(
        entities=entities,
        embeddings=embeddings,
        k=k,
        labels=labels,
        silhouette=sil,
        silhouette_scores=sil_scores,
    )


def extract_entities(quads: Sequence) -> list[str]:
    """Extract the sorted unique entity set from loaded quadruples.

    *quads* is a sequence of ``Quadruple`` objects (from
    ``Code.preprocessing``) or plain 4-tuples.
    """
    entities: set[str] = set()
    for q in quads:
        if hasattr(q, "subject"):
            entities.add(q.subject)
            entities.add(q.object)
        else:
            entities.add(q[0])
            entities.add(q[2])
    return sorted(entities)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from Code.preprocessing import load_dataset  # noqa: E402

    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "data" / "ICEWS05-15"
    )

    print(f"Loading dataset from: {data_dir}")
    quads = load_dataset(data_dir, splits=["train"])
    entities = extract_entities(quads)
    print(f"Unique entities: {len(entities):,}\n")

    result = cluster_entities(entities)
    print()
    print(result.summary())

    print("\n=== Sample: entities in same cluster as 'China' ===")
    try:
        peers = result.get_cluster_of("China")
        for e in peers[:20]:
            print(f"  {e}")
        if len(peers) > 20:
            print(f"  ... and {len(peers) - 20} more")
    except KeyError as exc:
        print(f"  {exc}")
