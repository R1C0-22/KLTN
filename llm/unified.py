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


def _load_huggingface_model(model_id: str) -> None:
    """Load model + tokenizer once (4-bit by default for Colab T4)."""
    global _hf_model, _hf_tokenizer

    import torch
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

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


def _call_huggingface(prompt: str) -> str:
    global _hf_model, _hf_tokenizer

    model_id = os.environ.get("HF_MODEL_ID", "").strip()
    if not model_id:
        raise EnvironmentError(
            "HF_MODEL_ID is not set. Examples (AnRe-style families): "
            "meta-llama/Meta-Llama-3-8B-Instruct, Qwen/Qwen2.5-7B-Instruct"
        )

    if _hf_model is None or _hf_tokenizer is None:
        _load_huggingface_model(model_id)

    import torch

    model = _hf_model
    tokenizer = _hf_tokenizer
    messages = [{"role": "user", "content": prompt}]

    max_new = int(os.environ.get("HF_MAX_NEW_TOKENS", "512"))
    cfg = model.config
    model_ctx = getattr(cfg, "max_position_embeddings", None) or getattr(
        cfg, "n_positions", None
    )
    if model_ctx is None:
        model_ctx = int(os.environ.get("HF_DEFAULT_MODEL_CONTEXT", "8192"))
    input_cap_override = os.environ.get("HF_MAX_INPUT_TOKENS", "").strip()
    if input_cap_override:
        input_cap = max(64, int(input_cap_override))
    else:
        input_cap = max(256, int(model_ctx) - max_new - 64)

    attention_mask: Any | None = None
    if getattr(tokenizer, "chat_template", None):
        # Some transformer/tokenizer versions return non-tensor objects from
        # `apply_chat_template(..., return_tensors="pt")`. To stay robust,
        # first materialize the prompt text, then re-tokenize with
        # `tokenizer(prompt_text, return_tensors="pt")`.
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=input_cap,
        )
    else:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=input_cap,
        )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    # Defensive fallback: enforce a torch tensor of token ids.
    if not torch.is_tensor(input_ids):
        # tokenizers.Encoding / BatchEncoding might sneak in on edge versions.
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = _env_truthy("HF_DO_SAMPLE", False)
    if attention_mask is not None:
        gen_in = dict(input_ids=input_ids, attention_mask=attention_mask)
    else:
        gen_in = dict(input_ids=input_ids)

    # max_new_tokens is required (open-ended decode). max_length on the model was
    # cleared in _hf_drop_fixed_max_length to avoid HF warnings.
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(os.environ.get("HF_TEMPERATURE", "0.2"))
    else:
        # Greedy: neutral sampling kwargs silence "temperature/top_p not valid" on some HF versions.
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0

    with torch.no_grad():
        out = model.generate(**gen_in, **gen_kwargs)

    new_tokens = out[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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
    """Get logprobs from HuggingFace local model using forward pass.
    
    This is the most accurate implementation matching the paper's methodology.
    """
    global _hf_model, _hf_tokenizer

    model_id = os.environ.get("HF_MODEL_ID", "").strip()
    if not model_id:
        raise EnvironmentError("HF_MODEL_ID is not set.")

    if _hf_model is None or _hf_tokenizer is None:
        _load_huggingface_model(model_id)

    import torch

    model = _hf_model
    tokenizer = _hf_tokenizer
    
    messages = [{"role": "user", "content": prompt}]
    
    cfg = model.config
    model_ctx = getattr(cfg, "max_position_embeddings", None) or getattr(
        cfg, "n_positions", None
    )
    if model_ctx is None:
        model_ctx = int(os.environ.get("HF_DEFAULT_MODEL_CONTEXT", "8192"))
    input_cap = max(256, int(model_ctx) - 64)

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=input_cap,
        )
    else:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=input_cap,
        )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    if not torch.is_tensor(input_ids):
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Get logits for the next token position
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[0, -1, :]
    
    # Extract logprobs for each candidate label token
    logprob_scores = []
    for label in candidate_labels:
        label_tokens = tokenizer.encode(label, add_special_tokens=False)
        if label_tokens:
            first_token_id = label_tokens[0]
            logprob = next_token_logits[first_token_id].item()
        else:
            logprob = -100.0
        logprob_scores.append(logprob)
    
    return logprob_scores


if __name__ == "__main__":
    # Simple manual test:
    #   set LLM_PROVIDER and OPENAI_* / GROQ_* / HF_* env vars, then run:
    #   python -m llm.unified
    test_prompt = "Say 'hello' and nothing else."
    print(call_llm(test_prompt))

