"""
Smoke test for generate_analogical_reasoning using the dummy generator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this file directly from the project root.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from analogical import generate_analogical_reasoning  # noqa: E402


def main() -> None:
    os.environ.setdefault("LLM_GENERATOR", "analogical.dummy_generator:generate_fn")

    event = ("USA", "meet", "China", "2014-01-01")
    similar_events = [
        ("Russia", "meet", "Belarus", "2013-01-01"),
        ("France", "visit", "Germany", "2013-02-01"),
    ]

    out = generate_analogical_reasoning(event, similar_events)
    assert "Analogical reasoning" in out
    print(out)


if __name__ == "__main__":
    main()

