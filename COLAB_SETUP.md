# Colab Setup (HF Llama/Qwen) - stable for AnRe (A100)

## Common Colab errors
- `PyTorch and torchvision were compiled with different CUDA major versions`
- `Could not import module 'Gemma3nConfig'`
- `StrictDataclassDefinitionError ... BloomConfig ... @strict`

These are environment/package mismatch issues (not AnRe logic bugs).

## Fix: install / upgrade (without breaking Colab)

**Do not** run `pip uninstall -y torch transformers huggingface_hub tokenizers ...` on Colab first.  
That removes packages that **other preinstalled libraries** (`torchtune`, `peft`, `timm`, `sentence-transformers`) depend on. Pip will often print long red lines like **“requires huggingface-hub … which is not installed”** during the *same* install — that is confusing but usually means “resolver saw a transient state”. If imports still work after the cell finishes, you can ignore those lines; if imports break, use **Runtime → Restart session** and re-run the **additive** install below (never bulk-uninstall first).

### Recommended (additive — KISS)
Run **once per runtime**, then **Restart session** if pip upgraded something major:

```bash
!pip -q install -U \
  "transformers>=4.41.0,<7.0.0" \
  accelerate \
  bitsandbytes \
  huggingface_hub \
  tokenizers \
  sentence-transformers \
  scikit-learn \
  numpy
```

### Full “whole cell” shell (copy-paste) — A100 / typical Colab GPU
Use this **as the only pip cell** when you start a fresh runtime (KISS):

```bash
%%bash
set -e
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

pip -q install -U \
  "transformers>=4.41.0,<7.0.0" \
  accelerate \
  bitsandbytes \
  huggingface_hub \
  tokenizers \
  sentence-transformers \
  scikit-learn \
  numpy

python -c "import transformers, huggingface_hub, tokenizers; print('HF stack OK', transformers.__version__)"
```

### Only if you truly have a CUDA / torch mismatch
Try this **instead** of uninstall-everything (still order matters — torch first, then HF stack):

```bash
!pip -q install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
!pip -q install -U \
  "transformers>=4.41.0,<7.0.0" \
  accelerate bitsandbytes huggingface_hub tokenizers \
  sentence-transformers scikit-learn numpy
```

### If you already bulk-uninstalled and Colab is “broken”
1. **Runtime → Restart session** (clears half-installed state).
2. Re-run **only** the “torch first (optional) + additive HF stack” block above — no `pip uninstall`.
3. If `import transformers` still fails: factory-reset runtime or start a **new notebook**; avoid uninstall storms.

Notes:
- After install, **Runtime → Restart session** if you still see import errors.
- Re-run setup cells after restart (no stale imports).

### Long prompts / PDC scoring (HF 8k models)
If `predict_next_object` fails in `cloud_adapter.py` with **“Could not find '[' in model output”**, the long-term scorer prompt was too long for the context window. Fix:
- Chunk scoring (default **32** events per LLM call): set `LLM_SCORE_CHUNK_SIZE=32` (or `24` on tighter GPUs).
- Optional: `HF_MAX_INPUT_TOKENS=6000`. Do **not** set `HF_SCORE_MAX_NEW_TOKENS` too low (e.g. 128 truncates the JSON score array). If unset, the repo picks a safe minimum from chunk size; you may set a **floor** only (e.g. `512`).
- Last resort for debugging: `LLM_SCORE_PARSE_FALLBACK=1` (deterministic pseudo-scores if JSON parse fails).

## Env variables for this repo (A100)
Example for **Llama 3**:

```python
import os
os.environ["LLM_PROVIDER"] = "hf"
os.environ["HF_MODEL_ID"] = "meta-llama/Meta-Llama-3-8B-Instruct"
os.environ["HF_LOAD_IN_4BIT"] = "0"        # A100: safe; set "1" if you want 4-bit
os.environ["HF_MAX_NEW_TOKENS"] = "200"
os.environ["LLM_SCORE_CHUNK_SIZE"] = "32"
os.environ["HF_DO_SAMPLE"] = "0"
# Optional PDC JSON floor (auto minimum is derived from chunk size if unset):
# os.environ["HF_SCORE_MAX_NEW_TOKENS"] = "512"
# For gated models (Llama): set HF_TOKEN via Colab Secrets (do not hardcode in repo)
# from google.colab import userdata
# os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["TKG_DATA_DIR"] = "data/ICEWS05-15"
```

Example for **Qwen2.5-7B**:
```python
import os
os.environ["LLM_PROVIDER"] = "hf"
os.environ["HF_MODEL_ID"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["HF_LOAD_IN_4BIT"] = "0"
os.environ["HF_MAX_NEW_TOKENS"] = "200"
os.environ["LLM_SCORE_CHUNK_SIZE"] = "32"
os.environ["HF_DO_SAMPLE"] = "0"
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
import os
from preprocessing import load_dataset
from inference.final_prediction import predict_next_object

os.environ["TKG_DATA_DIR"] = "data/ICEWS05-15"  # required when query is a plain tuple
v = load_dataset("data/ICEWS05-15", splits=["valid"])
e = v[0]
q = (e.subject, e.relation, "?", e.timestamp)
print(predict_next_object(q))
```

If `data/ICEWS05-15` exists under the repo root, you may omit `TKG_DATA_DIR` — the inference module will default to that path.

