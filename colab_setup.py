"""
Colab setup helper for AnRe TKG Forecasting.

Usage (in Colab):
    # Cell 1: Clone and install
    !cd /content && rm -rf KLTN && git clone https://github.com/R1C0-22/KLTN.git
    !pip install -q transformers>=4.41.0 accelerate bitsandbytes sentence-transformers scikit-learn
    # Then: Runtime -> Restart session
    
    # Cell 2: Setup and test
    import os, sys
    os.chdir("/content/KLTN")
    sys.path.insert(0, "/content/KLTN")
    
    from colab_setup import setup_env, test_llm
    setup_env(model="qwen")  # or "llama"
    test_llm()

IMPORTANT: Do NOT pin numpy version. Colab's preinstalled packages require numpy>=2.0.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = "/content/KLTN"


def _check_numpy_compat() -> None:
    """Warn if numpy version might cause binary incompatibility."""
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major < 2:
            print(f"[WARNING] numpy {np.__version__} detected. Colab packages need numpy>=2.0.")
            print("[WARNING] Run: pip install -U numpy && restart runtime")
    except Exception:
        pass


def setup_env(
    model: str = "qwen",
    load_4bit: bool = False,
    max_tokens: int = 200,
    hf_token: str | None = None,
) -> None:
    """Configure environment for HF local LLM inference.
    
    Args:
        model: "qwen" or "llama" (shortcuts), or full HF model ID
        load_4bit: Use 4-bit quantization (needs bitsandbytes)
        max_tokens: Max new tokens for generation
        hf_token: HF token for gated models (Llama)
    """
    _check_numpy_compat()
    
    model_map = {
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    model_id = model_map.get(model.lower(), model)
    
    os.environ["LLM_PROVIDER"] = "hf"
    os.environ["HF_MODEL_ID"] = model_id
    os.environ["HF_LOAD_IN_4BIT"] = "1" if load_4bit else "0"
    os.environ["HF_MAX_NEW_TOKENS"] = str(max_tokens)
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    os.environ["TKG_DATA_DIR"] = os.path.join(_REPO_ROOT, "data", "ICEWS05-15")
    
    print(f"[setup] LLM_PROVIDER=hf")
    print(f"[setup] HF_MODEL_ID={model_id}")
    print(f"[setup] HF_LOAD_IN_4BIT={os.environ['HF_LOAD_IN_4BIT']}")
    print(f"[setup] TKG_DATA_DIR={os.environ['TKG_DATA_DIR']}")


def test_llm(prompt: str = "Say hello in one sentence.") -> str:
    """Quick smoke test for LLM call."""
    from llm.unified import call_llm
    
    print(f"[test_llm] prompt: {prompt!r}")
    result = call_llm(prompt)
    print(f"[test_llm] output: {result}")
    return result


def test_scoring(n_events: int = 5) -> list[float]:
    """Test LLM scoring for long-term history filtering."""
    from preprocessing import load_dataset
    from long_term.long_term_filter import compute_scores_with_llm
    
    data_dir = os.environ.get("TKG_DATA_DIR", "data/ICEWS05-15")
    hist = load_dataset(data_dir, splits=["train"])[:n_events]
    
    q = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
    print(f"[test_scoring] query: {q}")
    print(f"[test_scoring] history events: {n_events}")
    
    scores = compute_scores_with_llm(hist, q)
    print(f"[test_scoring] scores: {scores}")
    return scores


def test_analogical() -> str:
    """Test analogical reasoning generation."""
    from analogical import generate_analogical_reasoning
    
    event = ("China", "meet", "?", "2014-01-01")
    similar = [
        ("Russia", "meet", "Belarus", "2013-01-01"),
        ("Iran", "meet", "Turkey", "2013-02-01"),
    ]
    
    print(f"[test_analogical] event: {event}")
    result = generate_analogical_reasoning(event, similar)
    print(f"[test_analogical] output (first 300 chars):\n{result[:300]}")
    return result


def test_prediction(split: str = "valid", idx: int = 0) -> str:
    """Test end-to-end prediction on a real query."""
    from preprocessing import load_dataset
    from inference.final_prediction import predict_next_object
    
    data_dir = os.environ.get("TKG_DATA_DIR", "data/ICEWS05-15")
    data = load_dataset(data_dir, splits=[split])
    
    e = data[idx]
    q = (e.subject, e.relation, "?", e.timestamp)
    
    print(f"[test_prediction] query: {q}")
    print(f"[test_prediction] ground truth: {e.object}")
    
    pred = predict_next_object(q)
    print(f"[test_prediction] predicted: {pred}")
    print(f"[test_prediction] correct: {pred == e.object}")
    return pred


def debug_scoring_prompt(n_events: int = 5) -> str:
    """Print the raw LLM output for scoring prompt (debug)."""
    from preprocessing import load_dataset
    from llm.unified import call_llm
    import long_term.long_term_filter as ltf
    
    data_dir = os.environ.get("TKG_DATA_DIR", "data/ICEWS05-15")
    hist = load_dataset(data_dir, splits=["train"])[:n_events]
    q = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
    
    prompt_template = ltf._load_prompt_template()
    labeled = [
        f"{i}. ({e.subject}, {e.relation}, {e.object}, {e.timestamp})"
        for i, e in enumerate(hist, 1)
    ]
    query_text = ltf._make_question_from_query_event(q)
    
    prompt = prompt_template.format(
        history="\n".join(labeled),
        events="\n".join(labeled),
        query=query_text,
        n=len(hist),
    )
    
    print("=== SCORING PROMPT ===")
    print(prompt[:500])
    print("\n=== RAW LLM OUTPUT ===")
    raw = call_llm(prompt)
    print(raw)
    return raw
