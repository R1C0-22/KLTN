# Colab Setup for AnRe TKG Forecasting

## Quick Start (GPU: T4 / L4 / A100)

Colab **L4** is an NVIDIA **GPU**, not a CPU. Free-tier runtimes often assign **Tesla T4** (~16GB VRAM); Pro / higher tiers may give **L4** or **A100**. If the runtime shows CPU-only, embedding + LLM will be very slow. Check: `import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))`.

### Reading a saved log (e.g. `output.txt`)

| Line / block | Meaning |
|--------------|--------|
| `device: Tesla T4` (or L4 / A100) | GPU name — **not** “CPU L4”; L4 is always a GPU. |
| `Loading weights: 100%` then long `[test_llm] completed in XXXs` | First `call_llm` pays **one-time** model load; XXXs is normal (often 5–10+ min on first run). |
| `temperature` / `top_p` ignored | Fixed in `llm/unified.py` by greedy `GenerationConfig` without sampling fields. `git pull` + restart runtime if you still see it; harmless if generation still works. |
| `[test_analogical] completed in 0.4s` right after TEST 1 | **Expected**: model already loaded; second forward is fast. Not dummy/cached unless output says `dummy`. |
| TEST 2 long creative text | Analogical text is **not** a numeric metric; LLMs may hallucinate dates — tune `prompts/reasoning_prompt.txt` for stricter paper-style output. |
| TEST 3 `scores=[..., ...]` with variance | PDC path OK (JSON logits parsed). All zeros ⇒ check raw output via `debug_scoring_raw()`. |
| TEST 4 `predicted=India` on synthetic | **No ground-truth label** in toy history — success = pipeline finished; compare to `e.object` only in `test_prediction()` on real `valid`. |
| BERT `UNEXPECTED position_ids` | Harmless for `bert-base-nli-mean-tokens` cross-load (see note below). |
| `test_prediction` `correct=False` | **Not a pipeline bug by itself.** TKGF is evaluated with **MRR / Hits@k** over many test queries; a single miss (pred ≠ `e.object`) is expected. If *most* queries miss, check PDC scores, candidate set (`ground_truth_in_candidate_set`), and HF logprob scoring (`HF_LOGPROB_FAST` — default full-label sum in `llm/unified.py`). |

### How to read `test_quick()` output

| Block | Meaning |
|-------|--------|
| TEST 1 | `call_llm` works (Llama/Qwen loaded). |
| TEST 2 | Analogical text (§3.3). The prompt now includes the **masked target query** plus the similar-event chain (last event = known answer). If the narrative still drifts, check `prompts/reasoning_prompt.txt` and `generate_analogical_reasoning`. |
| TEST 3 | PDC scores as JSON array (§3.2). Non-zero variance ⇒ scorer OK. |
| TEST 4 | End-to-end on **synthetic** history; loads the §3.1 embedder once (shared cache). `predicted=…` is not necessarily “ground truth” — synthetic data has no label. |
| BERT `UNEXPECTED position_ids` | Harmless when loading `bert-base-nli-mean-tokens` cross-task. |

**Smoke test with tiny history:** set `MIN_HISTORY_CONTEXTS=0` before `setup()` so similar-event filtering (§3.1, paper ≥300) does not empty candidates on toy data:

```python
# Full cell (matches typical notebook flow): verify GPU + smoke tests
import os, sys

import torch
print("torch:", getattr(torch, "__version__", "?"), getattr(torch, "__file__", ""))
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    # Colab assigns T4 / L4 / A100 — all are GPUs. "L4" is never a CPU.
    print("device:", torch.cuda.get_device_name(0))

os.chdir("/content/KLTN")
sys.path.insert(0, "/content/KLTN")

os.environ["MIN_HISTORY_CONTEXTS"] = "0"  # smoke only; paper §3.1: use "300" for real ICEWS runs

from colab_setup import setup, test_quick

setup(
    "llama",
    load_4bit=True,
    max_tokens=128,
    short_term_l=5,
    history_length=30,
)
test_quick()
```

**Is the run “correct”?** For smoke: yes if TEST 1 prints text, TEST 3 scores are not all zeros, TEST 4 finishes. `MIN_HISTORY_CONTEXTS=0` is **not** paper-faithful for §3.1 filtering; use `300` when comparing to AnRe. TEST 4 synthetic `predicted=India` has **no gold label** — only proves the pipeline runs; use `test_prediction()` on real `valid` for Hit@1-style checks.

### Cell 1: Clone and Install
`load_4bit=True` (default in `setup()`) **requires a working `bitsandbytes`**. If install is missing you get `ImportError`. If PyTorch and bitsandbytes versions conflict, `import bitsandbytes` can raise `RuntimeError` (e.g. duplicate kernel registration). In that case `setup()` **falls back to FP16/BF16** automatically (fine on **L4 / A100**; on **T4** you may need 4-bit or a smaller model). To force FP16: `setup("llama", load_4bit=False)`.

**Do not** run `pip install -U torch` by itself. Colab ships PyTorch + CUDA builds; upgrading only `torch` often breaks `torchvision` / `torchaudio` and can leave `torch` in a half-upgraded state (`AttributeError: module 'torch' has no attribute 'device'`). Prefer keeping Colab’s torch unless you install a **matching** trio from [pytorch.org](https://pytorch.org/get-started/locally/).

If `setup()` used to crash inside `clear_gpu_memory` → `torch.cuda.synchronize()`, `colab_setup.py` now catches that and continues (empty cache still runs). **Runtime → Restart session** is still the right fix if `import torch` itself is broken.

```python
# Mount Drive (optional - cache HF models)
from google.colab import drive, userdata
drive.mount("/content/drive")

import os
os.environ["LLM_CACHE_DIR"] = "/content/drive/MyDrive/llm_cache"
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # gated Llama only

# Repo + data (idempotent: pull if repo exists, else clone)
!cd /content && (test -d KLTN/.git && git -C KLTN pull --ff-only || test ! -d KLTN && git clone https://github.com/R1C0-22/KLTN.git || echo "Fix /content/KLTN manually")

# Copy dataset from Drive if present
!test -d /content/drive/MyDrive/data && cp -r /content/drive/MyDrive/data /content/KLTN/ || true

# Dependencies: numpy pin first, then ML stack — **omit torch** (use Colab default)
!python -m pip install -q -U pip
!python -m pip install -q -U "numpy>=1.26,<2.1"
!python -m pip install -q -U "bitsandbytes>=0.46.1" transformers accelerate sentence-transformers scikit-learn
```

Pip may print **dependency conflict warnings** (e.g. RAPIDS `libcuvs` vs CUDA 13). Those are often harmless if your imports work; if `import torch` fails, **restart runtime** and avoid upgrading unrelated CUDA stacks in the same session.

**After this cell: Runtime → Restart session**, then run Cell 2.

**No restart + no bitsandbytes:** use `setup("llama", load_4bit=False)` (FP16; needs enough VRAM, e.g. L4/A100).

### Cell 2: Quick Test (~30-60s)
```python
import os, sys
os.chdir("/content/KLTN")
sys.path.insert(0, "/content/KLTN")

from colab_setup import setup, test_quick
setup("llama")  # or "qwen"
test_quick()
```

### Cell 3: Full Test (slower; logs whether ground truth is in candidate set Oq)
```python
from colab_setup import test_prediction
# If you see ground_truth_in_candidate_set=False, Hit@1 cannot match; try:
# test_prediction(use_second_order=True)
# or: os.environ["MIN_HISTORY_CONTEXTS"] = "0" before setup()
test_prediction()
```

---

## Test Functions

| Function | Time | Description |
|----------|------|-------------|
| `test_quick()` | ~30-60s | Tests 1-4a (no clustering) |
| `test_all()` | ~3-5 min | All tests including full prediction |
| `test_llm()` | ~5s | Basic LLM call |
| `test_analogical()` | ~10s | Analogical reasoning (§3.3) |
| `test_scoring(n=5)` | ~10s | Long-term scoring (§3.2) |
| `test_prediction_quick()` | ~15s | Quick prediction (synthetic data) |
| `test_prediction()` | ~2-4 min | Full prediction (real data + clustering) |

---

## Debug

```python
from colab_setup import debug_scoring_raw
debug_scoring_raw(n=3)  # See raw LLM output for scoring
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `Unsupported LLM_PROVIDER='hf'` | `git pull` + restart runtime |
| `ImportError: ... bitsandbytes ...` (4-bit) | Run `!pip install -U 'bitsandbytes>=0.46.1'` then **Restart session**, or `setup(..., load_4bit=False)` |
| `OPENAI_API_KEY is not set` | Set `LLM_PROVIDER=hf` |
| `Could not infer dtype` | `git pull` + restart runtime |
| Test 4 hangs at "Batches" | Use `test_prediction_quick()` or wait for clustering |
| `CUDA out of memory` | Use `setup("llama", load_4bit=True)` (default now) |
| Prediction ≠ ground truth but `ground_truth_in_candidate_set=False` | Expected: expand candidates with `test_prediction(use_second_order=True)` or lower `MIN_HISTORY_CONTEXTS` |
| `temperature` / `top_p` ignored warnings | Fixed in `llm/unified.py` via explicit `GenerationConfig`; `git pull` |
| `numpy._core... _blas_supports_fpe` / `ModuleNotFoundError: GenerationMixin` / `AutoModelForCausalLM` | Caused by numpy 2.4+ breaking scipy/sklearn. Fix: `pip install "numpy>=1.26,<2.1"` then **Restart runtime**, re-run pip + `test_quick()` |
| `AttributeError: module 'torch' has no attribute 'device'` | Broken/partial torch upgrade. **Restart runtime**. Remove `pip install -U torch` from Cell 1; use Colab’s torch or install matching `torch`+`torchvision`+`torchaudio` from pytorch.org. `colab_setup.setup()` now calls `verify_torch_install()` with a clearer message. |
| Very slow PDC scoring | Defaults: `HF_SCORE_MAX_NEW_TOKENS=256`, `LLM_SCORE_CHUNK_SIZE=24` (set in `setup()`) |

### Cell: paper-style Hit@1 on first *N* valid queries (not one-shot)

`test_prediction()` compares **one** `valid[0]` query. For a defensible thesis metric, average over many queries:

```python
import os, sys
os.chdir("/content/KLTN")
sys.path.insert(0, "/content/KLTN")

from colab_setup import setup
from preprocessing import load_dataset
from inference.final_prediction import predict_next_object, get_prediction_context
from clustering.entity_cluster import cluster_entities, extract_entities

setup("llama", load_4bit=True, max_tokens=256, short_term_l=20, history_length=100, adaptive_candidates=True)
os.environ.setdefault("MIN_HISTORY_CONTEXTS", "300")

data_dir = os.environ["TKG_DATA_DIR"]
train = load_dataset(data_dir, splits=["train"])
valid = load_dataset(data_dir, splits=["valid"])
entities = extract_entities(train)
# cap clustering for speed (same idea as test_prediction sample_size)
cluster_result = cluster_entities(entities[:500] if len(entities) > 500 else entities)

N = 20
hits = 0
for i in range(min(N, len(valid))):
    e = valid[i]
    q = (e.subject, e.relation, "?", e.timestamp)
    ctx = get_prediction_context(q, cluster_result, use_second_order=False)
    if e.object.strip() not in {c.strip() for c in ctx.candidate_set}:
        continue  # skip impossible Hit@1
    pred = predict_next_object(q, cluster_result, use_second_order=False)
    if pred.strip() == e.object.strip():
        hits += 1
print(f"Hit@1 (approx, {N} tries): {hits}/{min(N, len(valid))}")
```

---

## Model Options

| Alias | Model ID | Notes |
|-------|----------|-------|
| `llama` | `meta-llama/Meta-Llama-3-8B-Instruct` | Needs `HF_TOKEN` |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | No token needed |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend | `hf` |
| `HF_MODEL_ID` | HuggingFace model | - |
| `HF_LOAD_IN_4BIT` | 4-bit quantization | `1` (on) |
| `HF_MAX_NEW_TOKENS` | Max generation length | `256` |
| `HF_LOGPROB_FAST` | HF only: `1` = first-subword logprob (fast, inaccurate for `1` vs `10`); default full label | `0` |
| `TKG_DATA_DIR` | Dataset directory | `data/ICEWS05-15` |
| `SHORT_TERM_L` | Short-term history limit | `10` |
| `HISTORY_LENGTH_L` | Total history limit | `50` |
