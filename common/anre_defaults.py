"""Hyperparameters and model IDs aligned with Tang et al., AnRe (ACL 2025)."""

import os

# §3.1 Entity Semantic Clustering: BERT vector representations (Devlin, 2018).
# Override for faster local runs, e.g. ANRE_EMBED_MODEL=all-MiniLM-L6-v2
DEFAULT_EMBED_MODEL = os.environ.get(
    "ANRE_EMBED_MODEL",
    "sentence-transformers/bert-base-nli-mean-tokens",
)
