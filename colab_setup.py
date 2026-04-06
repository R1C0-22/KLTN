"""
Colab setup helper for AnRe TKG Forecasting.

QUICKSTART (paste into Colab cell):
────────────────────────────────────────────────────────────────────
# Cell 1: Clone + install (run once, restart runtime after)
!cd /content && rm -rf KLTN && git clone https://github.com/R1C0-22/KLTN.git
!pip install -q transformers accelerate bitsandbytes sentence-transformers scikit-learn

# Cell 2: Mount drive for HF cache (optional but saves time)
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# Cell 3: Setup and quick test
import sys, os
sys.path.insert(0, "/content/KLTN")
os.chdir("/content/KLTN")

from colab_setup import setup, test_quick
setup("llama")
test_quick()

# Cell 4: Full test (slower, ~2-5 min)
from colab_setup import test_prediction
test_prediction()
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time

REPO_ROOT = "/content/KLTN"
DEFAULT_DATA_DIR = "data/ICEWS05-15"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def setup(
    model: str = "qwen",
    load_4bit: bool = False,
    max_tokens: int = 512,
    data_dir: str = DEFAULT_DATA_DIR,
) -> None:
    """Configure environment for HF local LLM inference.
    
    Args:
        model: "qwen", "llama", or full HF model ID
        load_4bit: Use 4-bit quantization (for T4). A100 can use False.
        max_tokens: Max new tokens for generation
        data_dir: Dataset directory relative to repo root
    """
    model_id = MODELS.get(model.lower(), model)
    
    os.environ["LLM_PROVIDER"] = "hf"
    os.environ["HF_MODEL_ID"] = model_id
    os.environ["HF_LOAD_IN_4BIT"] = "1" if load_4bit else "0"
    os.environ["HF_MAX_NEW_TOKENS"] = str(max_tokens)
    os.environ["TKG_DATA_DIR"] = os.path.join(REPO_ROOT, data_dir)
    os.environ["LLM_SCORE_PARSE_FALLBACK"] = "1"
    
    print(f"[setup] model={model_id}")
    print(f"[setup] 4bit={load_4bit}, max_tokens={max_tokens}")
    print(f"[setup] data={data_dir}")


def _timer(name: str):
    """Simple context manager for timing."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"[{name}] completed in {elapsed:.1f}s")
    return Timer()


def test_llm() -> str:
    """Test 1: Basic LLM call."""
    from llm.unified import call_llm
    
    print("[test_llm] calling model...")
    with _timer("test_llm"):
        result = call_llm("Say hello in one sentence.")
    print(f"[test_llm] output: {result}")
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
    
    print(f"[test_analogical] event={event}")
    print(f"[test_analogical] similar_events={len(similar)}")
    
    with _timer("test_analogical"):
        result = generate_analogical_reasoning(event, similar)
    
    print(f"[test_analogical] output ({len(result)} chars):")
    print("-" * 40)
    if max_chars > 0 and len(result) > max_chars:
        print(result[:max_chars] + "...")
    else:
        print(result)
    print("-" * 40)
    return result


def test_scoring(n: int = 5) -> list[float]:
    """Test 3: LLM scoring for long-term filtering (paper §3.2 PDC)."""
    from preprocessing import load_dataset
    from long_term.long_term_filter import compute_scores_with_llm
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    hist = load_dataset(data_dir, splits=["train"])[:n]
    query = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
    
    print(f"[test_scoring] query={query}")
    print(f"[test_scoring] n_events={n}")
    
    with _timer("test_scoring"):
        scores = compute_scores_with_llm(hist, query)
    
    print(f"[test_scoring] scores={scores}")
    
    if scores and all(s == 0.0 for s in scores):
        print("[test_scoring] WARNING: all scores are 0.0 - LLM may not output proper logits")
    elif scores and any(s != 0.0 for s in scores):
        print("[test_scoring] OK: scores have variance (LLM scoring works)")
    
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
    
    print(f"[test_prediction_quick] query=(China, meet, ?, 2014-01-01)")
    print(f"[test_prediction_quick] history_size={len(synthetic_data)}")
    
    with _timer("test_prediction_quick"):
        pred = predict_next_object(query)
    
    print(f"[test_prediction_quick] predicted={pred}")
    return pred


def test_prediction(sample_size: int = 1000) -> str:
    """Test 4b: Full prediction with real data (includes clustering).
    
    WARNING: This is slow (~2-5 min) because it:
    - Loads full training data
    - Embeds and clusters all entities
    - Runs the complete AnRe pipeline
    
    Args:
        sample_size: Max entities for clustering (reduces time)
    """
    from preprocessing import load_dataset
    from inference.final_prediction import predict_next_object
    from clustering.entity_cluster import cluster_entities, extract_entities
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    
    print(f"[test_prediction] Loading data from {data_dir}...")
    valid_data = load_dataset(data_dir, splits=["valid"])
    train_data = load_dataset(data_dir, splits=["train"])
    
    e = valid_data[0]
    query = (e.subject, e.relation, "?", e.timestamp)
    
    print(f"[test_prediction] query={query}")
    print(f"[test_prediction] ground_truth={e.object}")
    
    print(f"[test_prediction] Extracting entities...")
    entities = extract_entities(train_data)
    
    if sample_size and len(entities) > sample_size:
        import random
        random.seed(42)
        sampled = random.sample(entities, sample_size)
        if e.subject not in sampled:
            sampled.append(e.subject)
        if e.object not in sampled:
            sampled.append(e.object)
        entities = sorted(sampled)
        print(f"[test_prediction] Sampled {len(entities)} entities (from {len(extract_entities(train_data))})")
    
    print(f"[test_prediction] Clustering {len(entities)} entities...")
    with _timer("clustering"):
        cluster_result = cluster_entities(entities)
    
    print(f"[test_prediction] Running prediction...")
    with _timer("prediction"):
        pred = predict_next_object(query, cluster_result=cluster_result)
    
    print(f"[test_prediction] predicted={pred}")
    print(f"[test_prediction] ground_truth={e.object}")
    print(f"[test_prediction] correct={pred == e.object}")
    
    return pred


def test_quick() -> None:
    """Run quick tests (1-3 + 4a). Total time: ~30-60s."""
    print("\n" + "=" * 50)
    print("TEST 1: Basic LLM call")
    print("=" * 50)
    test_llm()
    
    print("\n" + "=" * 50)
    print("TEST 2: Analogical reasoning (paper §3.3)")
    print("=" * 50)
    test_analogical(max_chars=800)
    
    print("\n" + "=" * 50)
    print("TEST 3: LLM scoring (paper §3.2 PDC)")
    print("=" * 50)
    test_scoring(n=5)
    
    print("\n" + "=" * 50)
    print("TEST 4: Quick prediction (synthetic data)")
    print("=" * 50)
    test_prediction_quick()
    
    print("\n" + "=" * 50)
    print("QUICK TESTS COMPLETED")
    print("=" * 50)
    print("\nTo run full prediction with clustering, use: test_prediction()")


def test_all() -> None:
    """Run all tests including full prediction. Total time: ~3-5 min."""
    test_quick()
    
    print("\n" + "=" * 50)
    print("TEST 5: Full prediction (real data + clustering)")
    print("=" * 50)
    test_prediction()
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)


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
    
    print("=== PROMPT (first 500 chars) ===")
    print(prompt[:500])
    print("\n=== RAW LLM OUTPUT ===")
    raw = call_llm(prompt)
    print(raw)
    return raw
