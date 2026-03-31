"""
Analogical reasoning generation for Temporal Knowledge Graphs.

Implements:
  - generate_analogical_reasoning(event, similar_events)

This function uses an LLM generation callable provided by environment:
  LLM_GENERATOR="some_module:some_function"

The callable must accept a single argument `prompt` and return generated text.
If your callable takes positional/keyword variants, you can adapt the adapter
inside this module.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Sequence


def _event_to_text(ev: Any) -> str:
    """Convert an event/quadruple-like object into a compact string."""
    if (
        hasattr(ev, "subject")
        and hasattr(ev, "relation")
        and hasattr(ev, "object")
        and hasattr(ev, "timestamp")
    ):
        # Prefer natural-language verbalization if available.
        try:
            from Code.preprocessing import verbalize_event  # local import

            return verbalize_event(
                str(ev.subject),
                str(ev.relation),
                str(ev.object),
                str(ev.timestamp),
            )
        except Exception:
            return f"({ev.subject}, {ev.relation}, {ev.object}, {ev.timestamp})"
    if isinstance(ev, (tuple, list)) and len(ev) >= 4:
        s, r, o, t = ev[0], ev[1], ev[2], ev[3]
        try:
            from Code.preprocessing import verbalize_event  # local import

            return verbalize_event(str(s), str(r), str(o), str(t))
        except Exception:
            return f"({s}, {r}, {o}, {t})"
    return str(ev)


def _load_prompt_template() -> str:
    repo_root = Path(__file__).resolve().parent.parent.parent
    prompt_path = repo_root / "prompts" / "reasoning_prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Missing prompt template: {prompt_path}. "
            "Please create `prompts/reasoning_prompt.txt`."
        )
    return prompt_path.read_text(encoding="utf-8")


def _load_llm_generator_from_env() -> Callable[[str], str]:
    spec = os.environ.get("LLM_GENERATOR", "").strip()
    if not spec:
        # Permanent default: use local Ollama adapter.
        # You can override by setting LLM_GENERATOR.
        spec = "Code.llm.ollama_adapter:generate_fn"
    if ":" not in spec:
        raise ValueError("LLM_GENERATOR must be in format 'module_path:function_name'.")

    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"LLM_GENERATOR resolved to non-callable: {spec}")

    return fn  # type: ignore[return-value]


def generate_analogical_reasoning(event: Any, similar_events: Sequence[Any]) -> str:
    """Generate analogical reasoning for `event` using `similar_events`.

    Parameters
    ----------
    event:
        The target event (quadruple-like or any object that can be stringified).
    similar_events:
        Sequence of analogous events (quadruple-like or any stringifiable objects).

    Returns
    -------
    str:
        The generated reasoning text from the LLM.
    """
    prompt_template = _load_prompt_template()
    generator = _load_llm_generator_from_env()

    event_text = _event_to_text(event)
    similar_lines = "\n".join(f"{i+1}. {_event_to_text(e)}" for i, e in enumerate(similar_events))

    prompt = prompt_template.format(event=event_text, similar_events=similar_lines)

    # Try common callable conventions.
    try:
        out = generator(prompt)  # type: ignore[misc]
    except TypeError:
        out = generator(prompt=prompt)  # type: ignore[misc]

    if not isinstance(out, str):
        out = str(out)
    return out.strip()


if __name__ == "__main__":
    # Local smoke test with dummy generator.
    os.environ.setdefault("LLM_GENERATOR", "Code.analogical.dummy_generator:generate_fn")
    event = ("USA", "meet", "China", "2014-01-01")
    similar = [
        ("Russia", "meet", "Belarus", "2013-01-01"),
        ("France", "visit", "Germany", "2013-02-01"),
    ]
    print(generate_analogical_reasoning(event, similar))

