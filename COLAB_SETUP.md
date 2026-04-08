# Colab Setup for AnRe TKG Forecasting

## Quick Start (GPU: L4 / A100)

Colab **L4** is an NVIDIA **GPU**, not a CPU. If the runtime shows CPU-only, embedding + LLM will be very slow. Check: `import torch; print(torch.cuda.is_available())`.

### How to read `test_quick()` output

| Block | Meaning |
|-------|--------|
| TEST 1 | `call_llm` works (Llama/Qwen loaded). |
| TEST 2 | Analogical text (§3.3). If the model talks about the wrong country, the **similar_events** in the test were mismatched; use coherent China-centric examples (see `colab_setup.test_analogical`). |
| TEST 3 | PDC scores as JSON array (§3.2). Non-zero variance ⇒ scorer OK. |
| TEST 4 | End-to-end on **synthetic** history; loads the §3.1 embedder once (shared cache). `predicted=…` is not necessarily “ground truth” — synthetic data has no label. |
| BERT `UNEXPECTED position_ids` | Harmless when loading `bert-base-nli-mean-tokens` cross-task. |

**Smoke test with tiny history:** set `MIN_HISTORY_CONTEXTS=0` before `setup()` so similar-event filtering (§3.1, paper ≥300) does not empty candidates on toy data:

```python
import os, sys
os.chdir("/content/KLTN")
sys.path.insert(0, "/content/KLTN")
os.environ["MIN_HISTORY_CONTEXTS"] = "0"  # smoke only; use 300 for real ICEWS runs

from colab_setup import setup, test_quick
setup("llama", load_4bit=True, max_tokens=128, short_term_l=5, history_length=30)
test_quick()
```

### Cell 1: Clone and Install
`load_4bit=True` (default in `setup()`) **requires `bitsandbytes`**. If you skip install or run `setup()` before install, you get `ImportError: ... bitsandbytes ...`. Install first, then restart.

```python
# Mount Drive (optional - cache HF models)
from google.colab import drive, userdata
drive.mount('/content/drive')

import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # for gated Llama

# Clone repo + copy data
!cd /content && rm -rf KLTN && git clone https://github.com/R1C0-22/KLTN.git
!cp -r /content/drive/MyDrive/data /content/KLTN/  # if data on Drive

# Install dependencies — bitsandbytes MUST be present before setup(load_4bit=True)
!python -m pip install -q -U pip
!python -m pip install -q -U "bitsandbytes>=0.46.1" transformers accelerate sentence-transformers scikit-learn numpy
```

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
| Very slow PDC scoring | Defaults: `HF_SCORE_MAX_NEW_TOKENS=256`, `LLM_SCORE_CHUNK_SIZE=24` (set in `setup()`) |

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
| `TKG_DATA_DIR` | Dataset directory | `data/ICEWS05-15` |
| `SHORT_TERM_L` | Short-term history limit | `10` |
| `HISTORY_LENGTH_L` | Total history limit | `50` |
