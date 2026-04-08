"""
Analogical Replay reasoning generation for Temporal Knowledge Graphs.

Implements §3.3 of the AnRe framework (Tang et al., ACL 2025):
  - Generate analysis process pai for similar events
  - Construct analogical examples (Hai, eai, pai)
  - Format examples for the final prediction prompt

Key Algorithm (Algorithm 1, lines 18-21):
  For each (Hai, eai) in A:
    exai ← Replay(LLM(Hai, eai))
    P ← P ∪ exai

The analogical example includes:
  1. Historical Events (Hai) - the history chain of similar entity
  2. Question - masked similar event (si, rq, ?, ti)
  3. Answer - ground truth object oai (known since event is from history)
  4. Analysis - LLM-generated reasoning process pai
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from common import event_fields


@dataclass
class AnalogicalExample:
    """Container for a single analogical reasoning example.
    
    Paper §3.3: "The analogical example exai is constructed as (Hai, eai, pai)"
    """
    history: list[Any]
    event: Any
    answer: str
    analysis: str
    
    def format_for_prompt(self, example_num: int = 1) -> str:
        """Format this example for inclusion in the final prediction prompt.
        
        Follows the format in Table 9 (OEP prompt template).
        """
        from preprocessing import verbalize_event
        
        history_lines = []
        for ev in self.history:
            s, r, o, t = event_fields(ev)
            history_lines.append(verbalize_event(s, r, o, t))
        
        es, er, eo, et = event_fields(self.event)
        question = verbalize_event(es, er, "?", et)
        if question.endswith("."):
            question = question[:-1] + "?"
        
        return f"""Analogical Example {example_num}:
Historical Events:
{chr(10).join(history_lines)}

Question: {question}
Answer: {self.answer}. {self.analysis}
"""


def _event_to_text(ev: Any) -> str:
    """Convert an event to a verbalized sentence."""
    try:
        from preprocessing import verbalize_event
        s, r, o, t = event_fields(ev)
        return verbalize_event(s, r, o, t)
    except Exception:
        s, r, o, t = event_fields(ev)
        return f"({s}, {r}, {o}, {t})"


def _load_prompt_template() -> str:
    """Load the reasoning prompt template (Table 8)."""
    code_root = Path(__file__).resolve().parents[1]
    prompt_path = code_root / "prompts" / "reasoning_prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Missing prompt template: {prompt_path}. "
            "Please create `prompts/reasoning_prompt.txt`."
        )
    return prompt_path.read_text(encoding="utf-8")


def _load_llm_generator_from_env() -> Callable[[str], str]:
    """Load the LLM generation callable."""
    spec = os.environ.get("LLM_GENERATOR", "").strip()
    if not spec:
        spec = "llm.cloud_adapter:generate_fn"
    if ":" not in spec:
        raise ValueError("LLM_GENERATOR must be in format 'module_path:function_name'.")

    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"LLM_GENERATOR resolved to non-callable: {spec}")

    return fn


def generate_analysis_process(
    history: Sequence[Any],
    similar_event: Any,
    ground_truth_answer: str,
    *,
    target_query: str | None = None,
) -> str:
    """Generate analysis process pai for a similar event using LLM.
    
    Paper §3.3: "pai = LLM(θ2(Hai, eai))"
    
    This uses the reasoning prompt (Table 8) which includes the answer
    so the LLM can explain HOW that answer was derived from history.
    
    Parameters
    ----------
    history : Sequence[Any]
        Historical events Hai leading up to similar_event
    similar_event : Any
        The similar event eai (si, rq, oai, ti)
    ground_truth_answer : str
        The known answer oai (object entity of the similar event)
    
    Returns
    -------
    str
        The analysis process text explaining how answer is derived
    """
    prompt_template = _load_prompt_template()
    generator = _load_llm_generator_from_env()

    tq = (target_query or "").strip()
    if not tq:
        tq = "— (not specified — internal call without a separate target query) —"

    history_lines = [_event_to_text(ev) for ev in history]
    history_text = "\n".join(history_lines) if history_lines else "- none -"

    s, r, o, t = event_fields(similar_event)
    from preprocessing import verbalize_event
    question_text = verbalize_event(s, r, "?", t)
    if question_text.endswith("."):
        question_text = question_text[:-1] + "?"

    prompt = prompt_template.format(
        target_query=tq,
        history=history_text,
        question=question_text,
        answer=ground_truth_answer,
    )
    
    try:
        analysis = generator(prompt)
    except TypeError:
        analysis = generator(prompt=prompt)
    
    if not isinstance(analysis, str):
        analysis = str(analysis)
    
    return analysis.strip()


def generate_analogical_reasoning(event: Any, similar_events: Sequence[Any]) -> str:
    """Backward-compatible hook for notebooks and smoke tests.

    Older Colab snippets use ``from analogical import generate_analogical_reasoning``.
    The paper-aligned API is :func:`generate_analysis_process` /
    :func:`construct_analogical_example`; this wrapper keeps those snippets working.

    Uses the **last** event in *similar_events* as the grounded similar quadruple
    (answer = its object) and all **preceding** events as *history*. The *event*
    argument (masked target query) is verbalized into the prompt as *target_query*
    so the LLM sees both the main prediction task and the analogical chain (§3.3).
    """
    if not similar_events:
        raise ValueError("similar_events must contain at least one event.")
    history = list(similar_events[:-1])
    similar_event = similar_events[-1]
    _s, _r, ans, _t = event_fields(similar_event)

    from preprocessing import verbalize_event

    ts, tr, to, tt = event_fields(event)
    mask = "?" if to is None or str(to).strip() in {"?", "None", "null"} else str(to)
    target_query = verbalize_event(ts, tr, mask, tt)

    return generate_analysis_process(
        history=history,
        similar_event=similar_event,
        ground_truth_answer=str(ans).strip(),
        target_query=target_query,
    )


def construct_analogical_example(
    history: Sequence[Any],
    similar_event: Any,
) -> AnalogicalExample:
    """Construct a complete analogical example.
    
    Paper §3.3: "The analogical example exai is constructed as (Hai, eai, pai)"
    
    The answer is extracted from the similar_event itself (known from history).
    """
    s, r, o, t = event_fields(similar_event)
    ground_truth = o.strip()
    
    analysis = generate_analysis_process(
        history=history,
        similar_event=similar_event,
        ground_truth_answer=ground_truth,
    )
    
    return AnalogicalExample(
        history=list(history),
        event=similar_event,
        answer=ground_truth,
        analysis=analysis,
    )


def construct_analogical_examples_batch(
    similar_candidates: Sequence,
) -> list[AnalogicalExample]:
    """Construct multiple analogical examples from SimilarEventCandidates.
    
    Paper Algorithm 1, lines 18-21:
      for (Hai, eai) in A:
        exai ← Replay(LLM(Hai, eai))
        P ← P ∪ exai
    
    Parameters
    ----------
    similar_candidates : Sequence[SimilarEventCandidate]
        Output from find_similar_events_from_cluster()
    
    Returns
    -------
    list[AnalogicalExample]
        Analogical examples ready for the final prediction prompt
    """
    examples = []
    for candidate in similar_candidates:
        example = construct_analogical_example(
            history=candidate.history,
            similar_event=candidate.event,
        )
        examples.append(example)
    
    return examples


def format_analogical_examples_for_prompt(
    examples: Sequence[AnalogicalExample],
) -> str:
    """Format all analogical examples for the final prediction prompt.
    
    Returns a string ready to be inserted into the OEP prompt (Table 9).
    """
    if not examples:
        return "- No analogical examples available -"
    
    formatted = []
    for i, ex in enumerate(examples, start=1):
        formatted.append(ex.format_for_prompt(example_num=i))
    
    return "\n".join(formatted)


if __name__ == "__main__":
    os.environ.setdefault("LLM_GENERATOR", "analogical.dummy_generator:generate_fn")
    
    demo_history = [
        ("Russia", "visit", "China", "2013-05-01"),
        ("Russia", "meet", "China", "2013-06-01"),
        ("Russia", "negotiate", "Belarus", "2013-07-01"),
    ]
    similar_event = ("Russia", "meet", "Belarus", "2013-08-01")
    
    example = construct_analogical_example(demo_history, similar_event)
    print("=== Analogical Example ===")
    print(example.format_for_prompt())
