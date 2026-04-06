"""
Colab setup helper for AnRe TKG Forecasting.

QUICKSTART (paste into Colab):
─────────────────────────────────────────────────────────────
# Cell 1: Clone repo + install deps (run once, then restart runtime)
!cd /content && rm -rf KLTN && git clone https://github.com/R1C0-22/KLTN.git
!pip install -q transformers accelerate bitsandbytes sentence-transformers scikit-learn

# Cell 2: Mount drive for HF cache (optional but recommended)
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # for gated Llama

# Cell 3: Setup and test
import sys
sys.path.insert(0, "/content/KLTN")
os.chdir("/content/KLTN")

from colab_setup import setup, test_all
setup("llama")  # or "qwen"
test_all()
─────────────────────────────────────────────────────────────

NOTES:
- Do NOT pin numpy version. Colab's packages require numpy>=2.0.
- A100: use load_4bit=False (enough VRAM). T4: use load_4bit=True.
"""

from __future__ import annotations

import os

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
    """Configure environment for HF local LLM inference."""
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


def test_llm() -> str:
    """Test basic LLM call."""
    from llm.unified import call_llm
    
    print("[test_llm] calling model...")
    result = call_llm("Say hello in one sentence.")
    print(f"[test_llm] output: {result}")
    return result


def test_analogical() -> str:
    """Test analogical reasoning generation (paper §3.3).
    
    Requires at least 2 similar events: first n-1 become history,
    last one is the similar event with known answer.
    """
    from analogical import generate_analogical_reasoning
    
    event = ("China", "meet", "?", "2014-01-01")
    # Need 2+ events: [:-1] = history, [-1] = similar event with answer
    similar = [
        ("Russia", "consult", "Belarus", "2012-06-01"),
        ("Russia", "meet", "Ukraine", "2012-12-01"),
        ("Russia", "meet", "Belarus", "2013-01-01"),  # answer = Belarus
    ]
    
    print(f"[test_analogical] event={event}")
    print(f"[test_analogical] similar_events={len(similar)} (history={len(similar)-1})")
    result = generate_analogical_reasoning(event, similar)
    print(f"[test_analogical] output[:300]: {result[:300]}")
    return result


def test_scoring(n: int = 5) -> list[float]:
    """Test LLM scoring for long-term filtering (paper §3.2 PDC).
    
    Note: scores [1.0, 0.0, ...] often means model picked "event #1" 
    instead of outputting proper logits. This is acceptable with fallback.
    """
    from preprocessing import load_dataset
    from long_term.long_term_filter import compute_scores_with_llm
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    hist = load_dataset(data_dir, splits=["train"])[:n]
    query = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
    
    print(f"[test_scoring] query={query}")
    print(f"[test_scoring] n_events={n}")
    scores = compute_scores_with_llm(hist, query)
    print(f"[test_scoring] scores={scores}")
    
    # Warn if scores look like index selection instead of logits
    if scores and scores.count(0.0) >= len(scores) - 1:
        print("[test_scoring] WARNING: scores look uniform - model may not output proper logits")
        print("[test_scoring] This is handled by fallback, but quality may vary")
    
    return scores


def test_prediction() -> str:
    """Test end-to-end prediction."""
    from preprocessing import load_dataset
    from inference.final_prediction import predict_next_object
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    data = load_dataset(data_dir, splits=["valid"])
    e = data[0]
    query = (e.subject, e.relation, "?", e.timestamp)
    
    print(f"[test_prediction] query={query}")
    print(f"[test_prediction] ground_truth={e.object}")
    
    pred = predict_next_object(query)
    print(f"[test_prediction] predicted={pred}")
    print(f"[test_prediction] correct={pred == e.object}")
    return pred


def test_all() -> None:
    """Run all tests in sequence."""
    print("\n" + "="*50)
    print("TEST 1: Basic LLM call")
    print("="*50)
    test_llm()
    
    print("\n" + "="*50)
    print("TEST 2: Analogical reasoning (paper §3.3)")
    print("="*50)
    test_analogical()
    
    print("\n" + "="*50)
    print("TEST 3: LLM scoring (paper §3.2 PDC)")
    print("="*50)
    test_scoring(n=5)
    
    print("\n" + "="*50)
    print("TEST 4: End-to-end prediction")
    print("="*50)
    test_prediction()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)


def debug_scoring_raw(n: int = 3) -> str:
    """Print raw LLM output for scoring prompt (debug only)."""
    from preprocessing import load_dataset
    from llm.unified import call_llm
    import long_term.long_term_filter as ltf
    
    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    hist = load_dataset(data_dir, splits=["train"])[:n]
    query = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
    
    template = ltf._load_prompt_template()
    labeled = [f"{i}. ({e.subject}, {e.relation}, {e.object}, {e.timestamp})"
               for i, e in enumerate(hist, 1)]
    query_text = ltf._make_question_from_query_event(query)
    
    prompt = template.format(
        history="\n".join(labeled),
        events="\n".join(labeled),
        query=query_text,
        n=len(hist),
    )
    
    print("=== PROMPT (first 400 chars) ===")
    print(prompt[:400])
    print("\n=== RAW LLM OUTPUT ===")
    raw = call_llm(prompt)
    print(raw)
    return raw
