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

    pip install torch transformers accelerate bitsandbytes

`call_llm(prompt)` always returns a plain string with the model’s text output.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

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
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        enc = tokenizer(prompt_text, return_tensors="pt", padding=False)
    else:
        enc = tokenizer(prompt, return_tensors="pt", padding=False)

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

    max_new = int(os.environ.get("HF_MAX_NEW_TOKENS", "512"))
    do_sample = _env_truthy("HF_DO_SAMPLE", False)
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(os.environ.get("HF_TEMPERATURE", "0.2"))
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    new_tokens = out[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    # Simple manual test:
    #   set LLM_PROVIDER and OPENAI_* / GROQ_* / HF_* env vars, then run:
    #   python -m llm.unified
    test_prompt = "Say 'hello' and nothing else."
    print(call_llm(test_prompt))

