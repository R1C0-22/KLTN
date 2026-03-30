"""
Smoke test for generate_analogical_reasoning using the dummy generator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure `import Code.*` works when running this file directly:
#   python Code\analogical\test_analogical.py
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Code.analogical import generate_analogical_reasoning  # noqa: E402


def main() -> None:
    os.environ.setdefault("LLM_GENERATOR", "Code.analogical.dummy_generator:generate_fn")

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

