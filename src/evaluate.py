"""Shared evaluation entry point for the HTR project.

Evaluate CRNN and TrOCR on the same IAM-line test samples and persist the
results (metrics, predictions, figures) into ``experiments/``.

    python src/evaluate.py
    python src/evaluate.py --max-test 400
    python src/evaluate.py --beam-width 5 --trocr-beams 4

For the full list of options, run with --help.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    # Delegate to the existing, fully-featured evaluate script so all options,
    # checkpoint discovery and figure generation are preserved verbatim.
    from scripts.evaluate import main as evaluate_main

    evaluate_main()


if __name__ == "__main__":
    main()
