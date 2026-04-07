"""
Unified LLM interface for this project.

Supports multiple providers behind a single function:

    from llm import call_llm
    text = call_llm("Your prompt here")

Currently supported providers
-----------------------------
- OpenAI (chat/completions-style HTTP API)
- Groq (Groq cloud chat API)
- Hugging Face local (Transformers on GPU — Colab-friendly)

Provider & model selection
--------------------------
Configuration is via environment variables:

    LLM_PROVIDER=openai|groq|hf      # which backend to use

For OpenAI:
    OPENAI_API_KEY=...               # required
    OPENAI_BASE_URL=...              # optional, default: https://api.openai.com/v1
    OPENAI_MODEL=...                 # optional, default: gpt-4o-mini

For Groq:
    GROQ_API_KEY=...                 # required
    GROQ_BASE_URL=...                # optional, default: https://api.groq.com/openai/v1
    GROQ_MODEL=...                   # optional, default: llama-3.1-8b-instant

For Hugging Face local (Colab GPU — download weights, matches paper model families):
    LLM_PROVIDER=hf
    HF_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct   # or Qwen/Qwen2.5-7B-Instruct
    HF_TOKEN=...                     # Hugging Face token (required for gated Llama)
    HF_LOAD_IN_4BIT=1                # default on; use 0 for full FP16 if VRAM allows
    HF_MAX_NEW_TOKENS=512
    HF_TRUST_REMOTE_CODE=0           # set 1 only if the model card asks for it
    HF_MAX_INPUT_TOKENS=...          # optional cap on *prompt* tokens (truncate); avoids HF crashes on long PDC prompts
    HF_CLEAR_GPU_CACHE=0             # default off — empty_cache every gen is slow; set 1 only if OOM

    pip install torch transformers accelerate bitsandbytes

`call_llm(prompt)` always returns a plain string with the model’s text output.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _hf_drop_fixed_max_length(model: Any) -> None:
    """HF checkpoints often set generation_config.max_length (e.g. 4096). Passing
    max_new_tokens in generate() then triggers: "Both max_new_tokens and max_length".
    Unset max_length once after load so only max_new_tokens controls decode length."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    try:
        setattr(gc, "max_length", None)
    except (TypeError, AttributeError):
        try:
            object.__setattr__(gc, "max_length", None)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Hugging Face local cache (lazy-loaded)
# ---------------------------------------------------------------------------

_hf_model: Any = None
_hf_tokenizer: Any = None


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Call the configured LLM provider with a simple text prompt.

    Returns the model's textual response (stripped).
    """
    provider = (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider == "openai":
        return _call_openai(prompt).strip()
    if provider == "groq":
        return _call_groq(prompt).strip()
    if provider in ("hf", "huggingface", "local", "transformers"):
        return _call_huggingface(prompt).strip()
    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. Use 'openai', 'groq', or 'hf'."
    )


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _call_openai(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"OpenAI API error {e.code}: {body[:500]}") from e

    data = json.loads(raw)
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response format: {raw[:500]}") from exc


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------

def _call_groq(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.2")),
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"Groq API error {e.code}: {body[:500]}") from e

    data = json.loads(raw)
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:
        raise RuntimeError(f"Unexpected Groq response format: {raw[:500]}") from exc


# ---------------------------------------------------------------------------
# Hugging Face local backend (Colab / GPU)
# ---------------------------------------------------------------------------

def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _log(msg: str) -> None:
    """Print with flush for real-time output in Colab."""
    print(msg, flush=True)


def _load_huggingface_model(model_id: str) -> None:
    """Load model + tokenizer once (4-bit by default for Colab T4)."""
    global _hf_model, _hf_tokenizer

    import torch
    
    _log(f"[llm] Loading model: {model_id}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        msg = str(exc)
        if "PyTorch and torchvision were compiled with different CUDA major versions" in msg:
            raise RuntimeError(
                "Detected mismatched torch/torchvision CUDA builds in Colab. "
                "Reinstall matching wheels (e.g. torch==2.5.1, torchvision==0.20.1 "
                "from cu124 index), then restart runtime."
            ) from exc
        if "Gemma3nConfig" in msg:
            raise RuntimeError(
                "Transformers installation is inconsistent (missing Gemma3nConfig). "
                "Reinstall transformers + huggingface_hub with compatible versions, "
                "then restart runtime."
            ) from exc
        raise

    token = os.environ.get("HF_TOKEN", "").strip() or None
    trust_remote = _env_truthy("HF_TRUST_REMOTE_CODE", False)
    load_4bit = _env_truthy("HF_LOAD_IN_4BIT", True)

    _log(f"[llm] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    _log(f"[llm] Loading weights (4bit={load_4bit})...")
    model_kwargs: dict[str, Any] = dict(
        device_map="auto",
        token=token,
        trust_remote_code=trust_remote,
    )
    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16,
            )
        except Exception:
            model_kwargs.pop("quantization_config", None)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    _hf_drop_fixed_max_length(model)
    _hf_model = model
    _hf_tokenizer = tokenizer
    _log(f"[llm] Model loaded successfully")


def _clear_gpu_cache() -> None:
    """Clear GPU cache to prevent OOM during repeated generations."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _call_huggingface(prompt: str) -> str:
    """Generate text using local HuggingFace model with proper memory management."""
    global _hf_model, _hf_tokenizer
    import gc

    model_id = os.environ.get("HF_MODEL_ID", "").strip()
    if not model_id:
        raise EnvironmentError(
            "HF_MODEL_ID is not set. Examples: "
            "meta-llama/Meta-Llama-3-8B-Instruct, Qwen/Qwen2.5-7B-Instruct"
        )

    if _hf_model is None or _hf_tokenizer is None:
        _load_huggingface_model(model_id)

    import torch

    model = _hf_model
    tokenizer = _hf_tokenizer
    verbose = _env_truthy("LLM_VERBOSE", False)

    # Config
    max_new = int(os.environ.get("HF_MAX_NEW_TOKENS", "256"))
    cfg = model.config
    model_ctx = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 8192)
    input_cap_override = os.environ.get("HF_MAX_INPUT_TOKENS", "").strip()
    input_cap = int(input_cap_override) if input_cap_override else max(256, model_ctx - max_new - 64)

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt_text = prompt

    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=input_cap)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if verbose:
        _log(f"[llm] Generating (input={input_ids.shape[1]} tokens, max_new={max_new})...")

    # Explicit GenerationConfig avoids warnings from stale model.generation_config
    # (temperature/top_p) when do_sample=False.
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )

    out = None
    new_tokens = None
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
                use_cache=True,
            )

        new_tokens = out[0, input_ids.shape[1] :]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if verbose:
            _log(f"[llm] Generated {len(new_tokens)} tokens")
    finally:
        del input_ids, attention_mask, enc
        if out is not None:
            del out
        if new_tokens is not None:
            del new_tokens
        gc.collect()
        # Clearing CUDA cache every call is very slow on Colab; set HF_CLEAR_GPU_CACHE=1 if OOM.
        if _env_truthy("HF_CLEAR_GPU_CACHE", False):
            _clear_gpu_cache()

    return result


# ---------------------------------------------------------------------------
# Logprob-based scoring for candidates (Paper §3.3, §2.2)
# ---------------------------------------------------------------------------

def call_llm_logprobs(prompt: str, candidate_labels: list[str]) -> list[float]:
    """Get log-probabilities for candidate labels following the paper's approach.
    
    Paper §3.3 and §2.2: "We map each candidate entity to a numerical token,
    obtain the corresponding logarithmic output La from the LLM, and convert
    it into a normalized probability using the softmax function."
    
    Parameters
    ----------
    prompt : str
        The complete prompt ending with a request for the model to output
        a numerical label (e.g., "Your choice is:")
    candidate_labels : list[str]
        List of label strings (e.g., ["1", "2", "3", ...]) to score
    
    Returns
    -------
    list[float]
        Log-probabilities (logits) for each candidate label
    """
    provider = (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider == "openai":
        return _logprobs_openai(prompt, candidate_labels)
    if provider == "groq":
        return _logprobs_groq(prompt, candidate_labels)
    if provider in ("hf", "huggingface", "local", "transformers"):
        return _logprobs_huggingface(prompt, candidate_labels)
    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}' for logprobs. Use 'openai', 'groq', or 'hf'."
    )


def _logprobs_openai(prompt: str, candidate_labels: list[str]) -> list[float]:
    """Get logprobs from OpenAI API using logprobs parameter."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
        "logprobs": True,
        "top_logprobs": min(20, len(candidate_labels)),
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"OpenAI API error {e.code}: {body[:500]}") from e

    data = json.loads(raw)
    
    logprob_scores = [-100.0] * len(candidate_labels)
    
    try:
        logprobs_content = data["choices"][0].get("logprobs", {})
        if logprobs_content and "content" in logprobs_content:
            first_token_logprobs = logprobs_content["content"][0].get("top_logprobs", [])
            for lp_entry in first_token_logprobs:
                token = lp_entry.get("token", "").strip()
                logprob = lp_entry.get("logprob", -100.0)
                for i, label in enumerate(candidate_labels):
                    if token == label or token == label.strip():
                        logprob_scores[i] = max(logprob_scores[i], logprob)
    except (KeyError, IndexError, TypeError):
        pass
    
    return logprob_scores


def _logprobs_groq(prompt: str, candidate_labels: list[str]) -> list[float]:
    """Get logprobs from Groq API (fallback to generation if not supported)."""
    # Groq doesn't fully support logprobs yet, use generation fallback
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"Groq API error {e.code}: {body[:500]}") from e

    data = json.loads(raw)
    
    # Fallback: assign high score to the generated token if it matches a label
    logprob_scores = [-100.0] * len(candidate_labels)
    
    try:
        generated = str(data["choices"][0]["message"]["content"]).strip()
        for i, label in enumerate(candidate_labels):
            if generated.startswith(label) or label in generated[:10]:
                logprob_scores[i] = 0.0
    except Exception:
        pass
    
    return logprob_scores


def _logprobs_huggingface(prompt: str, candidate_labels: list[str]) -> list[float]:
    """Get logprobs from HuggingFace model using forward pass (paper §3.3, §2.2)."""
    global _hf_model, _hf_tokenizer
    import gc

    model_id = os.environ.get("HF_MODEL_ID", "").strip()
    if not model_id:
        raise EnvironmentError("HF_MODEL_ID is not set.")

    if _hf_model is None or _hf_tokenizer is None:
        _load_huggingface_model(model_id)

    import torch

    model = _hf_model
    tokenizer = _hf_tokenizer
    
    # Config
    cfg = model.config
    model_ctx = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 8192)
    input_cap = max(256, model_ctx - 64)

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt_text = prompt

    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=input_cap)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1, :].cpu()
        
        # Extract logprobs for each candidate label
        logprob_scores = []
        for label in candidate_labels:
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            if label_tokens:
                logprob_scores.append(next_token_logits[label_tokens[0]].item())
            else:
                logprob_scores.append(-100.0)
    finally:
        del input_ids, attention_mask, enc, outputs, next_token_logits
        gc.collect()
        if _env_truthy("HF_CLEAR_GPU_CACHE", False):
            _clear_gpu_cache()

    return logprob_scores


if __name__ == "__main__":
    # Simple manual test:
    #   set LLM_PROVIDER and OPENAI_* / GROQ_* / HF_* env vars, then run:
    #   python -m llm.unified
    test_prompt = "Say 'hello' and nothing else."
    print(call_llm(test_prompt))

