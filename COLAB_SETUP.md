# Colab Setup (HF Llama/Qwen) - stable for AnRe

## Problem
`StrictDataclassDefinitionError: Class 'BloomConfig' must be a dataclass before applying @strict.`

This error typically happens when `huggingface_hub` and `transformers` versions are not compatible
on Colab. In your run, `call_llm` + analogical generation + long-term scoring succeed,
but `predict_next_object()` crashes right after, due to this dependency mismatch.

## Fix: pin HF deps (recommended)
Run the following shell block **in your Colab notebook** (once per runtime, before running the pipeline).

```bash
!pip -q install -U \
  "torch" \
  "transformers" \
  "accelerate" \
  "bitsandbytes" \
  "sentence-transformers" \
  "scikit-learn" \
  "numpy" \
  "huggingface_hub<0.32"
```

Notes:
- We intentionally pin `huggingface_hub<0.32` to avoid the strict dataclass validation that
  triggers the `BloomConfig` crash.
- After this install, **Restart runtime** (recommended), then rerun your cells.

## Env variables for this repo
Example for **Llama 3**:

```python
import os
os.environ["LLM_PROVIDER"] = "hf"
os.environ["HF_MODEL_ID"] = "meta-llama/Meta-Llama-3-8B-Instruct"
os.environ["HF_LOAD_IN_4BIT"] = "0"        # A100: safe; set "1" if you want 4-bit
os.environ["HF_MAX_NEW_TOKENS"] = "200"
os.environ["HF_DO_SAMPLE"] = "0"
# For gated models (Llama): set HF_TOKEN
# os.environ["HF_TOKEN"] = "hf_xxx"
os.environ["TKG_DATA_DIR"] = "data/ICEWS05-15"
```

## Paper-aligned smoke tests (run after env set)
1) `call_llm` smoke:
```python
from llm.unified import call_llm
print(call_llm("Say hello in one short sentence."))
```
2) Analogical generation:
```python
from analogical import generate_analogical_reasoning
event = ("China", "meet", "?", "2014-01-01")
similar_events = [("Russia", "meet", "Belarus", "2013-01-01"), ("Iran", "meet", "Turkey", "2013-02-01")]
print(generate_analogical_reasoning(event, similar_events)[:500])
```
3) Long-term scoring:
```python
from preprocessing import load_dataset
from long_term.long_term_filter import compute_scores_with_llm
hist = load_dataset("data/ICEWS05-15", splits=["train"])[0:5]
q = (hist[0].subject, hist[0].relation, "?", hist[0].timestamp)
print(compute_scores_with_llm(hist, q))
```
4) End-to-end one-query prediction:
```python
from inference.final_prediction import predict_next_object
v = load_dataset("data/ICEWS05-15", splits=["valid"])
e = v[0]
q = (e.subject, e.relation, "?", e.timestamp)
print(predict_next_object(q))
```

