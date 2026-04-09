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

from common import env_truthy as _env_truthy, log as _log

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
    """Install runtime deps used by this project (Colab / GPU).

    Colab pitfall: ``pip install -U numpy`` can pull numpy 2.4+, which breaks
    the scipy/sklearn stack preinstalled on the image; then ``import transformers``
    fails with confusing errors (e.g. missing ``GenerationMixin`` / ``AutoModelForCausalLM``).
    Pin numpy first, then install the rest.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "pip", "numpy>=1.26,<2.1"],
        check=True,
    )
    base = [
        "transformers",
        "accelerate",
        "bitsandbytes>=0.46.1",
        "sentence-transformers",
        "scikit-learn",
    ]
    if extra:
        base.extend(extra)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", *base],
        check=True,
    )


def bitsandbytes_available() -> bool:
    """True if bitsandbytes can be imported and initialized (required for HF 4-bit load).

    On some Colab images, ``import bitsandbytes`` raises ``RuntimeError`` (e.g. duplicate
    kernel registration with a mismatched torch) — that must be treated as *unavailable*,
    not as a hard crash inside ``setup()``.
    """
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_load_4bit(requested_4bit: bool) -> bool:
    """Return whether we actually use 4-bit quantization.

    If ``requested_4bit`` but bitsandbytes is missing or broken, log once and fall back
    to FP16/BF16 (L4/A100 usually have enough VRAM for 7B/8B instruct models).
    """
    if not requested_4bit:
        return False
    if bitsandbytes_available():
        return True
    _log(
        "[setup] bitsandbytes missing or broken (ImportError or RuntimeError on import). "
        "Common on Colab after partial torch/bnb upgrades. Falling back to FP16/BF16 "
        "(set load_4bit=False explicitly to silence this). "
        "To fix 4-bit: pip install a matching bitsandbytes, then Runtime → Restart session."
    )
    return False


def clear_gpu_memory() -> None:
    """Clear GPU memory cache to prevent OOM errors.

    Never raises: on some Colab / PyTorch builds ``torch.cuda.synchronize()`` can
    throw ``AttributeError: module 'torch' has no attribute 'device'`` inside CUDA
    internals even when ``import torch`` looks fine. ``empty_cache()`` is enough
    for our use; synchronize is best-effort only.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not hasattr(torch, "device"):
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        pass


def verify_torch_install() -> None:
    """Fail fast if PyTorch is broken or shadowed (common after bad pip upgrades on Colab).

    Symptom: ``AttributeError: module 'torch' has no attribute 'device'`` inside
    ``transformers``/``accelerate`` when loading models — usually means ``torch`` was
    upgraded inconsistently with torchvision/torchaudio, or the runtime needs a restart.
    """
    import importlib

    t = importlib.import_module("torch")
    tf = getattr(t, "__file__", "") or ""

    # Real wheels load from site-packages/dist-packages; a local ``torch.py`` shadows PyTorch.
    if tf and "site-packages" not in tf and "dist-packages" not in tf:
        raise RuntimeError(
            f"``torch`` imported from unexpected path: {tf}\n"
            "Remove/rename any file named ``torch.py`` in the project cwd or sys.path."
        )

    if not hasattr(t, "device"):
        raise RuntimeError(
            "PyTorch looks broken: ``torch`` has no attribute ``device``.\n\n"
            "Typical Colab fixes (in order):\n"
            "  1) Runtime → Restart session (pip upgrades often need this).\n"
            "  2) Do **not** run ``pip install -U torch`` alone — it can mismatch torchvision/torchaudio.\n"
            "     Prefer Colab's preinstalled torch, or install a **matching trio** from pytorch.org "
            "(same CUDA build).\n"
            "  3) Re-run deps **without** upgrading torch:\n"
            "     ``pip install -q 'numpy>=1.26,<2.1' bitsandbytes transformers accelerate ...``\n"
            "See ``COLAB_SETUP.md`` Cell 1 for the recommended one-liners."
        )

    if not hasattr(t, "cuda"):
        raise RuntimeError("PyTorch install incomplete: ``torch`` has no attribute ``cuda``.")

    _log(f"[torch] version={getattr(t, '__version__', '?')} file={tf or '?'}")


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

    verify_torch_install()

    model_id = MODELS.get(model.lower(), model)

    effective_4bit = _resolve_load_4bit(load_4bit)

    os.environ["LLM_PROVIDER"] = "hf"
    os.environ["HF_MODEL_ID"] = model_id
    os.environ["HF_LOAD_IN_4BIT"] = "1" if effective_4bit else "0"
    os.environ["HF_MAX_NEW_TOKENS"] = str(max_tokens)
    # Final prediction: allow a bit more than generic max_tokens so the model can
    # print a short rationale + the index (paper §3.3). Overrides HF_MAX_NEW_TOKENS
    # only inside ``predict_fn`` (see llm/cloud_adapter.py).
    os.environ.setdefault("HF_PREDICT_MAX_NEW_TOKENS", str(max(128, min(512, max_tokens * 2))))
    os.environ["TKG_DATA_DIR"] = os.path.join(REPO_ROOT, data_dir)
    os.environ["LLM_SCORE_PARSE_FALLBACK"] = "1"
    os.environ.setdefault("HF_SCORE_MAX_NEW_TOKENS", "160")
    # Larger chunks => fewer HF forward passes per timestep (faster; slightly longer prompts).
    os.environ.setdefault("LLM_SCORE_CHUNK_SIZE", "24")
    # Cap events per calendar day before PDC — ICEWS days can have 1000+ events.
    os.environ.setdefault("LLM_SCORE_MAX_EVENTS_PER_TIMESTEP", "64")
    # Enable local on-disk cache by default on Colab so repeated notebook runs
    # (same prompts/candidates) do not recompute expensive HF forward passes.
    # This keeps algorithm logic unchanged and only avoids redundant calls.
    os.environ.setdefault("LLM_CACHE_DIR", "/content/llm_cache")
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
    # Optional second adaptive gate (disabled by default):
    # if top-1 probability < threshold, retry with O²q. Set >0 to enable.
    os.environ.setdefault("ADAPTIVE_CONFIDENCE_THRESHOLD", "0")
    os.environ.setdefault("DTF_ALPHA", "2.75")
    # Dual history (§3.2): one LLM PDC call per calendar day processed until L is filled.
    # Dense subjects can require 100+ days → 30+ minutes per query on T4. Cap timesteps for
    # Colab notebooks; set to "0" for full paper-faithful DTF (slow overnight runs).
    os.environ.setdefault("MAX_DTF_TIMESTEP_ITERATIONS", "40")

    clear_gpu_memory()
    
    _log(f"[setup] model={model_id}")
    _log(
        f"[setup] 4bit_requested={load_4bit} 4bit_effective={effective_4bit}, max_tokens={max_tokens}"
    )
    _log(
        f"[setup] history: short_term={short_term_l}, target_L={history_length} "
        f"(paper §6.1); adaptive_O2={adaptive_candidates}"
    )
    _log(
        f"[setup] MAX_DTF_TIMESTEP_ITERATIONS={os.environ.get('MAX_DTF_TIMESTEP_ITERATIONS', '0')} "
        f"(PDC calls per query ≈ min(days, cap); set 0 for no cap)"
    )
    _log(
        f"[setup] cache_dir={os.environ.get('LLM_CACHE_DIR', '(disabled)')} "
        "(set empty to disable cache)"
    )
    _log(f"[setup] data={data_dir}")

    try:
        import torch

        if torch.cuda.is_available():
            _log(f"[setup] cuda_available=True device={torch.cuda.get_device_name(0)}")
        else:
            _log("[setup] cuda_available=False (CPU-only — LLM + embeds will be very slow)")
    except Exception as exc:
        _log(f"[setup] could not probe torch/CUDA: {exc}")


def _log_quick_test_footer() -> None:
    """Short interpretation for Colab logs (paper §§3.2–3.3, IMPROVE.MD prediction parsing)."""
    _log("")
    _log("--- How to read this run (see COLAB_SETUP.md) ---")
    _log("• GPU line (T4/L4/A100): Colab gave you a GPU. There is no 'CPU L4' — L4 is always GPU.")
    _log("• TEST 2 in ~1s after TEST 1: normal — model already in VRAM; not a failed LLM call.")
    _log("• TEST 3 scores with variance: PDC path OK (§3.2). All zeros → run debug_scoring_raw().")
    _log("• TEST 4 synthetic: no gold label; predicted entity only proves pipeline runs.")
    _log("• MIN_HISTORY_CONTEXTS=0: smoke only. Paper §3.1 filtering uses 300 — set before setup().")
    _log("• Paper-faithful eval: test_prediction() on real valid + compare pred to e.object.")
    _log("• BERT UNEXPECTED position_ids: harmless for bert-base-nli-mean-tokens cross-load.")


def _timer(name: str):
    """Simple context manager for timing."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            _log(f"[{name}] starting...")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            _log(f"[{name}] completed in {elapsed:.1f}s")
    return Timer()


def test_llm() -> str:
    """Test 1: Basic LLM call."""
    from llm.unified import call_llm

    _log(
        "[test_llm] calling model... "
        "(first call in a fresh runtime includes HF download + load + first forward; "
        "often minutes on T4; later calls are much faster)"
    )
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
    
    # Same subject/relation family as the query so the LLM does not drift to unrelated entities.
    event = ("China", "meet", "?", "2014-01-01")
    similar = [
        ("China", "meet", "Japan", "2013-01-01"),
        ("China", "consult", "Russia", "2013-06-01"),
        ("China", "meet", "India", "2013-12-01"),
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


def test_prediction_metrics(
    n_queries: int = 20,
    sample_size: int = 500,
    use_second_order: bool = False,
    start_index: int = 0,
) -> dict[str, float | int | str]:
    """Evaluate Hit@1 / Hit@10 over multiple validation queries (paper-style aggregate metrics).

    A single query where ``predicted != ground_truth`` is **not** a pipeline bug by itself;
    TKGF is evaluated with MRR / Hits@k over many queries (see ``IMPROVE.MD``).

    Only queries whose ground truth appears in the candidate set :math:`O_q` are counted
    toward Hit@k (fair evaluation); others are logged as skipped.

    Parameters
    ----------
    n_queries
        Number of validation samples starting at ``start_index``.
    sample_size
        Max entities for clustering (same as ``test_prediction``).
    use_second_order
        If True, use second-order candidate set when implemented by ``get_prediction_context``.
    start_index
        Offset into the validation split (default 0 = same first query as legacy ``test_prediction``).
    """
    # If the notebook calls this function directly (without `setup()`),
    # we still want sane defaults that match `IMPROVE.MD` guidance.
    #
    # IMPORTANT: On Colab T4, paper-faithful DTF can take tens of minutes per query
    # because each timestep triggers multiple sequential LLM forwards.
    os.environ.setdefault("LLM_PROVIDER", "hf")
    os.environ.setdefault("LLM_VERBOSE", "1")

    # Pick conservative defaults for T4 unless the user already set env vars.
    gpu_name = ""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0) or ""
    except Exception:
        gpu_name = ""

    is_t4 = "t4" in gpu_name.lower()
    if "LLM_SCORE_CHUNK_SIZE" not in os.environ:
        os.environ["LLM_SCORE_CHUNK_SIZE"] = "16" if is_t4 else "24"
    if "LLM_SCORE_MAX_EVENTS_PER_TIMESTEP" not in os.environ:
        os.environ["LLM_SCORE_MAX_EVENTS_PER_TIMESTEP"] = "32" if is_t4 else "64"
    if "MAX_DTF_TIMESTEP_ITERATIONS" not in os.environ:
        os.environ["MAX_DTF_TIMESTEP_ITERATIONS"] = "10" if is_t4 else "40"

    if gpu_name:
        use_logprob = _env_truthy("USE_LOGPROB_PREDICTION", default=False)
        _log(
            f"[test_prediction_metrics] GPU={gpu_name} "
            f"(chunk={os.environ.get('LLM_SCORE_CHUNK_SIZE')}, "
            f"cap_per_day={os.environ.get('LLM_SCORE_MAX_EVENTS_PER_TIMESTEP')}, "
            f"max_dtf_days={os.environ.get('MAX_DTF_TIMESTEP_ITERATIONS')}, "
            f"use_logprob={use_logprob})"
        )

    from preprocessing import load_dataset
    from inference.final_prediction import get_prediction_context, predict_next_object_with_probs
    from clustering.entity_cluster import cluster_entities, extract_entities

    data_dir = os.environ.get("TKG_DATA_DIR", DEFAULT_DATA_DIR)
    valid_data = load_dataset(data_dir, splits=["valid"])
    train_data = load_dataset(data_dir, splits=["train"])
    n_valid = len(valid_data)
    if n_valid == 0:
        raise RuntimeError(f"No validation quadruples in {data_dir}")

    # n_queries <= 0 means "run all remaining validation queries".
    if n_queries <= 0:
        end = n_valid
    else:
        end = min(start_index + n_queries, n_valid)
    if start_index < 0 or start_index >= n_valid:
        raise ValueError(f"start_index must be in [0, {n_valid - 1}], got {start_index}")

    anchor = valid_data[start_index]
    _log(f"[test_prediction_metrics] Loading data from {data_dir}...")
    _log(f"[test_prediction_metrics] valid queries [{start_index}, {end}) (n_valid={n_valid})")

    _log(f"[test_prediction_metrics] Extracting entities...")
    entities = extract_entities(train_data)
    total_entities = len(entities)
    if sample_size and len(entities) > sample_size:
        import random

        random.seed(42)
        sampled = random.sample(entities, sample_size)
        # Keep all entities that appear in the evaluation window so clustering
        # quality is not biased toward only the first query.
        required_entities = {anchor.subject, anchor.object}
        for ev in valid_data[start_index:end]:
            required_entities.add(ev.subject)
            required_entities.add(ev.object)

        for probe in required_entities:
            if probe not in sampled:
                sampled.append(probe)
        entities = sorted(sampled)
        _log(f"[test_prediction_metrics] Sampled {len(entities)} entities (from {total_entities})")

    _log(f"[test_prediction_metrics] Clustering {len(entities)} entities...")
    with _timer("clustering"):
        cluster_result = cluster_entities(entities)
    clear_gpu_memory()

    hits1 = 0
    hits10 = 0
    evaluated = 0
    skipped = 0
    last_prediction = ""

    for idx in range(start_index, end):
        e = valid_data[idx]
        query = (e.subject, e.relation, "?", e.timestamp)
        gt = e.object.strip()
        ctx = get_prediction_context(query, cluster_result, use_second_order)
        cand_norm = {c.strip() for c in ctx.candidate_set}
        in_oq = gt in cand_norm
        if not in_oq:
            skipped += 1
            _log(
                f"[test_prediction_metrics] i={idx} SKIP gt not in Oq | gt={gt!r} |Oq|={len(ctx.candidate_set)}"
            )
            continue

        _log(
            f"[test_prediction_metrics] i={idx} query={query} gt={gt!r} "
            f"|Oq|={len(ctx.candidate_set)} "
            f"used_second_order={getattr(ctx, 'used_second_order_neighbors', False)}"
        )
        with _timer(f"prediction[{idx}]"):
            res = predict_next_object_with_probs(query, cluster_result, use_second_order)
        pred = res.predicted.strip()
        last_prediction = pred
        evaluated += 1
        ok1 = pred == gt
        ok10 = res.hit_at_k(gt, 10)
        if ok1:
            hits1 += 1
        if ok10:
            hits10 += 1
        _log(
            f"[test_prediction_metrics] i={idx} predicted={pred!r} "
            f"Hit@1={ok1} Hit@10={ok10}"
        )

    hit1_rate = (hits1 / evaluated) if evaluated else 0.0
    hit10_rate = (hits10 / evaluated) if evaluated else 0.0
    _log(
        f"[test_prediction_metrics] SUMMARY evaluated={evaluated} "
        f"skipped_gt_not_in_Oq={skipped} "
        f"Hit@1={hit1_rate:.4f} Hit@10={hit10_rate:.4f}"
    )
    if end - start_index == 1 and evaluated == 1 and hits1 == 0:
        _log(
            "[test_prediction_metrics] NOTE: Hit@1 miss on one query is normal for TKGF; "
            "use test_prediction_metrics(n_queries=20, ...) for a stable aggregate. "
            "If Hit@1 stays low: run test_scoring(), set HF_LOGPROB_FAST=0, or try use_second_order=True."
        )

    clear_gpu_memory()
    return {
        "last_prediction": last_prediction,
        "hit_at_1": hit1_rate,
        "hit_at_10": hit10_rate,
        "evaluated": evaluated,
        "skipped_gt_not_in_oq": skipped,
        "n_queries_run": end - start_index,
    }


def test_prediction(sample_size: int = 500, use_second_order: bool = False) -> str:
    """Test 4b: Full prediction with real data (includes clustering).

    WARNING: Runtime is dominated by many LLM calls (PDC per timestep + analogical + predict).
    On A100 prefer ``setup(..., load_4bit=False)`` and ``verbose=False`` for speed.

    This is a thin wrapper around :func:`test_prediction_metrics` with ``n_queries=1``.
    For paper-faithful numbers, call ``test_prediction_metrics(n_queries=20, ...)`` instead.
    """
    stats = test_prediction_metrics(
        n_queries=1,
        sample_size=sample_size,
        use_second_order=use_second_order,
        start_index=0,
    )
    return str(stats["last_prediction"])


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
    _log_quick_test_footer()
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
