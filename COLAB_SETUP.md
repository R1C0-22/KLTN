# Colab Setup for AnRe TKG Forecasting

## Quick Start (A100 GPU)

### Cell 1: Clone and Install
```python
# Mount Drive (optional - cache HF models)
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"

# Clone repo
!cd /content && rm -rf KLTN && git clone https://github.com/R1C0-22/KLTN.git

# Install dependencies (DO NOT pin numpy version!)
!pip install -q transformers>=4.41.0 accelerate bitsandbytes sentence-transformers scikit-learn
```

**After this cell: Runtime → Restart session**

### Cell 2: Setup Environment
```python
import os, sys
os.chdir("/content/KLTN")
sys.path.insert(0, "/content/KLTN")

from colab_setup import setup_env, test_llm

# Choose model: "qwen" or "llama"
setup_env(model="qwen", load_4bit=False, max_tokens=200)

# Test LLM
test_llm("Say hello in one sentence.")
```

### Cell 3: Run Paper Tests
```python
from colab_setup import test_analogical, test_scoring, test_prediction

# Analogical reasoning (paper §3.3)
test_analogical()

# Long-term scoring (paper §3.2)
test_scoring(n_events=5)

# End-to-end prediction
test_prediction(split="valid", idx=0)
```

---

## Common Errors

### `numpy.dtype size changed, may indicate binary incompatibility`
**Cause**: You pinned `numpy<2.0` but Colab packages need `numpy>=2.0`.

**Fix**: 
1. Do NOT install with `numpy>=1.26,<2.0`
2. Just run: `pip install -q transformers accelerate bitsandbytes sentence-transformers scikit-learn`
3. **Runtime → Restart session**

### `Unsupported LLM_PROVIDER='hf'`
**Cause**: Old code version on Colab.

**Fix**: `git pull` and restart runtime.

### `OPENAI_API_KEY is not set`
**Cause**: Using cloud backend but no API key.

**Fix**: Set `os.environ["LLM_PROVIDER"] = "hf"` for local HF models.

---

## Model Options

| Model | HF ID | Notes |
|-------|-------|-------|
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | No token needed |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | Needs HF token (gated) |

For Llama, set token:
```python
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

---

## Environment Variables

```python
os.environ["LLM_PROVIDER"] = "hf"                    # Use HuggingFace local
os.environ["HF_MODEL_ID"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["HF_LOAD_IN_4BIT"] = "0"                  # "1" for 4-bit quantization
os.environ["HF_MAX_NEW_TOKENS"] = "200"
os.environ["TKG_DATA_DIR"] = "data/ICEWS05-15"
```
