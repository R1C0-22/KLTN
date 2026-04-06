# Colab Setup for AnRe TKG Forecasting

## Quick Start (A100 GPU)

### Cell 1: Clone and Install
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

# Install dependencies (DO NOT pin numpy!)
!pip install -q transformers accelerate bitsandbytes sentence-transformers scikit-learn
```

**After this cell: Runtime → Restart session**

### Cell 2: Quick Test (~30-60s)
```python
import os, sys
os.chdir("/content/KLTN")
sys.path.insert(0, "/content/KLTN")

from colab_setup import setup, test_quick
setup("llama")  # or "qwen"
test_quick()
```

### Cell 3: Full Test (~3-5 min, optional)
```python
from colab_setup import test_prediction
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
| `OPENAI_API_KEY is not set` | Set `LLM_PROVIDER=hf` |
| `Could not infer dtype` | `git pull` + restart runtime |
| Test 4 hangs at "Batches" | Use `test_prediction_quick()` or wait for clustering |
| `CUDA out of memory` | Use `setup("llama", load_4bit=True)` (default now) |

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
