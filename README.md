# AnRe — Analogical Replay for Temporal Knowledge Graph Forecasting

Implementation of the AnRe framework (Tang et al., ACL 2025) with the following
thesis improvements:

| Improvement | Description |
|---|---|
| Adaptive O_q / O²_q | Dynamic 1-hop → 2-hop candidate expansion when `\|O_q\|` is small |
| LLM response caching | Disk cache keyed by `(provider, model, prompt)` to avoid redundant calls |
| Hyperparameter sweep | Configurable sweep over `L`, `l`, `α` (paper §6.1) |
| Ablation study | w/o long-term, w/o short-term, w/o analogical reasoning |

## Project structure

```
Code/
├── analogical/          # §3.3  Analogical replay reasoning
├── clustering/          # §3.1  Entity clustering + candidate filter
├── common/              # Shared utilities (event parsing, env helpers)
├── evaluation/          # Ablation, hyperparameter sweep, cache benchmark
├── history/             # Entity history retrieval
├── inference/           # Final prediction pipeline (Algorithm 1)
├── llm/                 # Unified LLM interface (HF local / OpenAI / Groq)
├── long_term/           # §3.2  Dual history extraction (PDC + DTF)
├── preprocessing/       # Dataset loading + verbalization
├── prompts/             # Prompt templates (filter, reasoning, prediction)
├── short_term/          # §3.2  Short-term history selection
├── colab_setup.py       # One-call Colab configuration + test suite
├── test_pipeline.py     # Offline smoke test (clustering only)
└── requirements.txt
```

## Quickstart (Google Colab with T4 GPU)

```python
# Cell 1 — clone + install
!cd /content && git clone https://github.com/R1C0-22/KLTN.git
!pip install -q -U pip 'numpy>=1.26,<2.1' transformers accelerate bitsandbytes sentence-transformers scikit-learn

# Cell 2 — setup
import sys, os
sys.path.insert(0, "/content/KLTN")
os.chdir("/content/KLTN")

from colab_setup import setup, test_quick
setup("llama")       # or "qwen"
test_quick()

# Cell 3 — evaluation
from colab_setup import test_prediction_metrics
stats = test_prediction_metrics(n_queries=20, sample_size=500)
print(stats)
```

## Models tested

| Model | HF ID |
|---|---|
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` |

## Datasets

- **ICEWS05-15** — placed at `data/ICEWS05-15/` with `train`, `valid`, `test` splits.
