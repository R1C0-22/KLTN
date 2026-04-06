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

### Cell 2: Setup and Test
```python
import os, sys
os.chdir("/content/KLTN")
sys.path.insert(0, "/content/KLTN")

from colab_setup import setup, test_all

setup("llama")  # or "qwen"
test_all()
```

---

## Individual Tests

```python
from colab_setup import test_llm, test_analogical, test_scoring, test_prediction

test_llm()          # Basic LLM call
test_analogical()   # Analogical reasoning (paper §3.3)
test_scoring(n=5)   # Long-term scoring (paper §3.2)
test_prediction()   # End-to-end prediction
```

---

## Debug Scoring
```python
from colab_setup import debug_scoring_raw
debug_scoring_raw(n=3)  # See raw LLM output for scoring prompt
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `numpy.dtype size changed` | Pinned numpy<2.0 | Don't pin numpy, restart runtime |
| `Unsupported LLM_PROVIDER='hf'` | Old code | `git pull` + restart runtime |
| `OPENAI_API_KEY is not set` | Wrong provider | Set `LLM_PROVIDER=hf` |
| `Could not infer dtype` | Old tokenizer code | `git pull` + restart runtime |

---

## Model Options

| Model | ID | Notes |
|-------|-----|-------|
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | No token needed |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | Needs HF token |

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | Backend | `hf` |
| `HF_MODEL_ID` | Model ID | `Qwen/Qwen2.5-7B-Instruct` |
| `HF_LOAD_IN_4BIT` | Quantization | `0` (A100) or `1` (T4) |
| `HF_MAX_NEW_TOKENS` | Max output | `512` |
| `TKG_DATA_DIR` | Dataset path | `data/ICEWS05-15` |
| `LLM_SCORE_PARSE_FALLBACK` | Fallback scoring | `1` |
