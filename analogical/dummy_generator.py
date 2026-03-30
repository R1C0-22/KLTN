"""
Dummy LLM generator for local testing.

Sets:
  LLM_GENERATOR="Code.analogical.dummy_generator:generate_fn"
"""

from __future__ import annotations


def generate_fn(prompt: str) -> str:
    # Deterministic "reasoning" stub so tests are repeatable.
    # This is NOT real LLM output; it just helps validate integration.
    return (
        "Analogical reasoning (dummy):\n"
        f"{prompt[:250].strip()}\n"
        "...\n"
        "Conclusion: the target event likely follows the patterns "
        "observed in the similar events."
    )

