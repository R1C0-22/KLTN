"""
Colab setup helper for AnRe TKG Forecasting.

Why shells sometimes break on Colab
────────────────────────────────────
If you delete the folder that was the notebook's current working directory, the
shell session keeps a "dead" cwd → ``getcwd: cannot access parent directories``.
Fix: always ``cd /content`` before ``git`` / ``pip`` (see ``ensure_content_cwd``).

QUICKSTART (paste cells in order)
────────────────────────────────────
# --- Cell 0 (only if you see getcwd / pip "folder no longer found") ---
import os
if os.path.isdir("/content"):
    os.chdir("/content")

# --- Cell 1: repo + pip (idempotent; never rm -rf KLTN blindly) ---
# Option A — one line per command (works in Colab):
# !cd /content && (test -d KLTN/.git && git -C KLTN pull --ff-only || test ! -d KLTN && git clone https://github.com/R1C0-22/KLTN.git || (echo 'Fix /content/KLTN manually' && exit 1))
# !cd /content && python -m pip install -q -U pip && python -m pip install -q transformers accelerate bitsandbytes sentence-transformers scikit-learn numpy
# Option B — use ``colab_setup.ensure_colab_repo()`` after the first clone exists (Cell 3).

# --- Cell 2: Drive + HF token (optional) ---
# from google.colab import drive, userdata
# drive.mount("/content/drive")
# import os
# os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
# os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# --- Cell 3: copy data from Drive (if you store datasets on Drive) ---
# !test -d /content/drive/MyDrive/data && cp -r /content/drive/MyDrive/data /content/KLTN/ || true

# --- Cell 4: run tests ---
# import sys, os
# sys.path.insert(0, "/content/KLTN")
# os.chdir("/content/KLTN")
# from colab_setup import setup, test_quick
# setup("llama")   # Meta-Llama-3-8B-Instruct; L4/A100: 4-bit default
# test_quick()

Extras:
  - Disk cache: os.environ["LLM_CACHE_DIR"] = "/content/drive/MyDrive/llm_cache"
  - Strict paper Oq: setup(..., adaptive_candidates=False)
  - Fast smoke: setup(..., short_term_l=5, history_length=30)
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = "/content/KLTN"
DEFAULT_DATA_DIR = "data/ICEWS05-15"
DEFAULT_REPO_URL = "https://github.com/R1C0-22/KLTN.git"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def ensure_content_cwd() -> None:
    """Reset process cwd to ``/content`` (fixes Colab shells after a deleted folder)."""
    if Path("/content").is_dir():
        os.chdir("/content")


def ensure_colab_repo(
    dest: str = REPO_ROOT,
    repo_url: str = DEFAULT_REPO_URL,
) -> None:
    """Clone ``repo_url`` into ``dest``, or run ``git pull --ff-only`` if already a repo."""
    ensure_content_cwd()
    path = Path(dest)
    if (path / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(path), "pull", "--ff-only"],
            check=True,
        )
        return
    if path.exists():
        raise RuntimeError(
            f"{path} exists but is not a git repository. "
            "Remove or rename it, then run again."
        )
    subprocess.run(["git", "clone", repo_url, str(path)], check=True)


def pip_install_colab_deps(extra: list[str] | None = None) -> None:
    """Install runtime deps used by this project (Colab / GPU)."""
    base = [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
    ]
    if extra:
        base.extend(extra)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "pip", *base],
        check=True,
    )


def _log(msg: str) -> None:
    """Print with flush for real-time output in Colab."""
    print(msg, flush=True)


def clear_gpu_memory() -> None:
    """Clear GPU memory cache to prevent OOM errors."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def setup(
    model: str = "qwen",
    load_4bit: bool = True,
    max_tokens: int = 200,
    data_dir: str = DEFAULT_DATA_DIR,
    short_term_l: int = 20,
    history_length: int = 100,
    adaptive_candidates: bool = True,
    verbose: bool = True,
) -> None:
    """Configure environment for HF local LLM inference (Colab GPU, e.g. L4 / A100).
    
    Args:
        model: "qwen", "llama", or full HF model ID
        load_4bit: Use 4-bit quantization (default True to prevent OOM)
        max_tokens: Max new tokens for generation (reduced from 256 to prevent OOM)
        data_dir: Dataset directory relative to repo root
        short_term_l: Short-term chain length l (paper §6.1 default 20)
        history_length: Dual-history target length L (paper §6.1 default 100)
        adaptive_candidates: If True, expand to O²q when |Oq| is small (thesis
            improvement; paper Table 2). If False, strict Oq only.
        verbose: Enable real-time logging
    """
    if Path(REPO_ROOT).is_dir():
        try:
            os.chdir(REPO_ROOT)
        except OSError:
            pass

    model_id = MODELS.get(model.lower(), model)

    os.environ["LLM_PROVIDER"] = "hf"
    os.environ["HF_MODEL_ID"] = model_id
    os.environ["HF_LOAD_IN_4BIT"] = "1" if load_4bit else "0"
    os.environ["HF_MAX_NEW_TOKENS"] = str(max_tokens)
    # Final prediction: allow a bit more than generic max_tokens so the model can
    # print a short rationale + the index (paper §3.3). Overrides HF_MAX_NEW_TOKENS
    # only inside ``predict_fn`` (see llm/cloud_adapter.py).
    os.environ.setdefault("HF_PREDICT_MAX_NEW_TOKENS", str(max(128, min(512, max_tokens * 2))))
    os.environ["TKG_DATA_DIR"] = os.path.join(REPO_ROOT, data_dir)
    os.environ["LLM_SCORE_PARSE_FALLBACK"] = "1"
    os.environ.setdefault("HF_SCORE_MAX_NEW_TOKENS", "256")
    # Larger chunks => fewer HF forward passes per timestep (faster; slightly longer prompts).
    os.environ.setdefault("LLM_SCORE_CHUNK_SIZE", "48")
    # Cap events per calendar day before PDC — ICEWS days can have 1000+ events.
    os.environ.setdefault("LLM_SCORE_MAX_EVENTS_PER_TIMESTEP", "64")
    # empty_cache() after every generate() is very slow on Colab; enable only if OOM.
    os.environ.setdefault("HF_CLEAR_GPU_CACHE", "0")
    os.environ["LLM_VERBOSE"] = "1" if verbose else "0"

    # Single place to bind callables (Scout rule: explicit beats implicit).
    # Override in the notebook if you use a custom adapter.
    os.environ.setdefault("LLM_SCORER", "llm.cloud_adapter:score_fn")
    os.environ.setdefault("LLM_GENERATOR", "llm.cloud_adapter:generate_fn")
    os.environ.setdefault("LLM_PREDICTOR", "llm.cloud_adapter:predict_fn")
    os.environ.setdefault(
        "LLM_PREDICTOR_LOGPROBS", "llm.cloud_adapter:predict_with_logprobs_fn"
    )
    
    os.environ["SHORT_TERM_L"] = str(short_term_l)
    os.environ["HISTORY_LENGTH_L"] = str(history_length)
    os.environ["NUM_ANALOGICAL_EXAMPLES"] = "1"
    # Paper §3.1: ≥300 historical contexts before similar-event timestamp.
    os.environ.setdefault("MIN_HISTORY_CONTEXTS", "300")
    os.environ["ADAPTIVE_CANDIDATES"] = "1" if adaptive_candidates else "0"
    os.environ.setdefault("ADAPTIVE_MIN_CANDIDATES", "3")
    os.environ.setdefault("DTF_ALPHA", "2.75")

    clear_gpu_memory()
    
    _log(f"[setup] model={model_id}")
    _log(f"[setup] 4bit={load_4bit}, max_tokens={max_tokens}")
    _log(
        f"[setup] history: short_term={short_term_l}, target_L={history_length} "
        f"(paper §6.1); adaptive_O2={adaptive_candidates}"
    )
    _log(f"[setup] data={data_dir}")


def _timer(name: str):
    """Simple context manager for timing."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            _log(f"[{name}] completed in {elapsed:.1f}s")
    return Timer()


def test_llm() -> str:
    """Test 1: Basic LLM call."""
    from llm.unified import call_llm
    
    _log("[test_llm] calling model...")
    with _timer("test_llm"):
        result = call_llm("Say hello in one sentence.")
    _log(f"[test_llm] output: {result}")
    return result


def test_analogical(max_chars: int = 0) -> str:
    """Test 2: Analogical reasoning generation (paper §3.3).
    
    Args:
        max_chars: Max chars to display (0 = show all)
    """
    from analogical import generate_analogical_reasoning
    
    event = ("China", "meet", "?", "2014-01-01")
    similar = [
        ("Russia", "consult", "Belarus", "2012-06-01"),
        ("Russia", "meet", "Ukraine", "2012-12-01"),
        ("Russia", "meet", "Belarus", "2013-01-01"),
    ]
    
    _log(f"[test_analogical] event={event}")
    _log(f"[test_analogical] similar_events={len(similar)}")
    
    with _timer("test_analogical"):
        result = generate_analogical_reasoning(event, similar)
    
    _log(f"[test_analogical] output ({len(result)} chars):")
    _log("-" * 40)
    if max_chars > 0 and len(result) > max_chars:
        _log(result[:max_chars] + "...")
    else:
        _log(result)
    _log("-" * 40)
    return result


def test_scoring(n: int = 5) -> list[float]:
    """Test 3: LLM scoring for long-term filtering (paper §3.2 PDC)."""
    from preprocessing import load_dataset
    from long_term.long_term_filter import compute_scores_with_llm
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    hist = load_dataset(data_dir, splits=["train"])[:n]
    query = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
    
    _log(f"[test_scoring] query={query}")
    _log(f"[test_scoring] n_events={n}")
    
    with _timer("test_scoring"):
        scores = compute_scores_with_llm(hist, query)
    
    _log(f"[test_scoring] scores={scores}")
    
    if scores and all(s == 0.0 for s in scores):
        _log("[test_scoring] WARNING: all scores are 0.0 - LLM may not output proper logits")
    elif scores and any(s != 0.0 for s in scores):
        _log("[test_scoring] OK: scores have variance (LLM scoring works)")
    
    return scores


def test_prediction_quick() -> str:
    """Test 4a: Quick prediction using synthetic data (no clustering).
    
    This test bypasses the slow clustering step by providing
    pre-defined history data directly in the query.
    """
    from inference.final_prediction import predict_next_object
    
    synthetic_data = [
        ("China", "meet", "Japan", "2013-01-01"),
        ("China", "meet", "Russia", "2013-06-01"),
        ("China", "consult", "Japan", "2013-09-01"),
        ("China", "meet", "India", "2013-12-01"),
        ("Russia", "meet", "Belarus", "2013-01-01"),
        ("Japan", "visit", "China", "2013-03-01"),
    ]
    
    query = {
        "subject": "China",
        "relation": "meet",
        "object": "?",
        "timestamp": "2014-01-01",
        "data": synthetic_data,
    }
    
    _log(f"[test_prediction_quick] query=(China, meet, ?, 2014-01-01)")
    _log(f"[test_prediction_quick] history_size={len(synthetic_data)}")
    
    with _timer("test_prediction_quick"):
        pred = predict_next_object(query)
    
    _log(f"[test_prediction_quick] predicted={pred}")
    return pred


def test_prediction(sample_size: int = 500, use_second_order: bool = False) -> str:
    """Test 4b: Full prediction with real data (includes clustering).

    WARNING: Runtime is dominated by many LLM calls (PDC per timestep + analogical + predict).
    On A100 prefer ``setup(..., load_4bit=False)`` and ``verbose=False`` for speed.
    
    Args:
        sample_size: Max entities for clustering (reduces time and memory)
        use_second_order: If True, use O²q candidate set (paper Table 2). Helps when GT ∉ Oq.
    """
    from preprocessing import load_dataset
    from inference.final_prediction import predict_next_object, get_prediction_context
    from clustering.entity_cluster import cluster_entities, extract_entities
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    
    _log(f"[test_prediction] Loading data from {data_dir}...")
    valid_data = load_dataset(data_dir, splits=["valid"])
    train_data = load_dataset(data_dir, splits=["train"])
    
    e = valid_data[0]
    query = (e.subject, e.relation, "?", e.timestamp)
    
    _log(f"[test_prediction] query={query}")
    _log(f"[test_prediction] ground_truth={e.object}")
    
    _log(f"[test_prediction] Extracting entities...")
    entities = extract_entities(train_data)
    total_entities = len(entities)
    
    if sample_size and len(entities) > sample_size:
        import random
        random.seed(42)
        sampled = random.sample(entities, sample_size)
        if e.subject not in sampled:
            sampled.append(e.subject)
        if e.object not in sampled:
            sampled.append(e.object)
        entities = sorted(sampled)
        _log(f"[test_prediction] Sampled {len(entities)} entities (from {total_entities})")
    
    _log(f"[test_prediction] Clustering {len(entities)} entities...")
    with _timer("clustering"):
        cluster_result = cluster_entities(entities)
    
    clear_gpu_memory()

    ctx = get_prediction_context(query, cluster_result, use_second_order)
    gt_norm = e.object.strip()
    in_oq = gt_norm in {c.strip() for c in ctx.candidate_set}
    _log(
        f"[test_prediction] |Oq|={len(ctx.candidate_set)} "
        f"used_second_order_neighbors={getattr(ctx, 'used_second_order_neighbors', False)} "
        f"ground_truth_in_candidate_set={in_oq}"
    )
    if not in_oq:
        _log(
            "[test_prediction] NOTE: ground truth not in Oq — Hit@1 impossible; "
            "try test_prediction(..., use_second_order=True) or lower MIN_HISTORY_CONTEXTS."
        )
    
    _log(f"[test_prediction] Running prediction...")
    with _timer("prediction"):
        pred = predict_next_object(query, cluster_result, use_second_order)
    
    _log(f"[test_prediction] predicted={pred}")
    _log(f"[test_prediction] ground_truth={e.object}")
    _log(f"[test_prediction] correct={pred == e.object}")
    
    clear_gpu_memory()
    
    return pred


def test_quick() -> None:
    """Run quick tests (1-3 + 4a). Total time: ~30-60s."""
    _log("\n" + "=" * 50)
    _log("TEST 1: Basic LLM call")
    _log("=" * 50)
    test_llm()
    clear_gpu_memory()
    
    _log("\n" + "=" * 50)
    _log("TEST 2: Analogical reasoning (paper §3.3)")
    _log("=" * 50)
    test_analogical(max_chars=3000)
    clear_gpu_memory()
    
    _log("\n" + "=" * 50)
    _log("TEST 3: LLM scoring (paper §3.2 PDC)")
    _log("=" * 50)
    test_scoring(n=5)
    clear_gpu_memory()
    
    _log("\n" + "=" * 50)
    _log("TEST 4: Quick prediction (synthetic data)")
    _log("=" * 50)
    test_prediction_quick()
    clear_gpu_memory()
    
    _log("\n" + "=" * 50)
    _log("QUICK TESTS COMPLETED")
    _log("=" * 50)
    _log("\nTo run full prediction with clustering, use: test_prediction()")


def test_all() -> None:
    """Run all tests including full prediction. Total time: ~3-5 min."""
    test_quick()
    
    clear_gpu_memory()
    
    _log("\n" + "=" * 50)
    _log("TEST 5: Full prediction (real data + clustering)")
    _log("=" * 50)
    test_prediction()
    
    clear_gpu_memory()
    
    _log("\n" + "=" * 50)
    _log("ALL TESTS COMPLETED")
    _log("=" * 50)


def debug_scoring_raw(n: int = 3) -> str:
    """Debug: Print raw LLM output for scoring prompt."""
    from preprocessing import load_dataset
    from llm.unified import call_llm
    import long_term.long_term_filter as ltf
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    hist = load_dataset(data_dir, splits=["train"])[:n]
    query = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
    
    template = ltf._load_prompt_template()
    labeled = [
        f"{i}. ({e.subject}, {e.relation}, {e.object}, {e.timestamp})"
        for i, e in enumerate(hist, 1)
    ]
    query_text = ltf._make_question_from_query_event(query)
    
    prompt = template.format(
        history="\n".join(labeled),
        events="\n".join(labeled),
        query=query_text,
        n=len(hist),
    )
    
    _log("=== PROMPT (first 500 chars) ===")
    _log(prompt[:500])
    _log("\n=== RAW LLM OUTPUT ===")
    raw = call_llm(prompt)
    _log(raw)
    return raw
