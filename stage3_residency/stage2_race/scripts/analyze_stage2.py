"""Stage 2 analysis, tables, figures and the frozen report."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import EVALUATION_DIR, FROZEN_CONFIG, PREREGISTRATION, ROOT, STAGE2

from race_stage2.analysis import analyze_and_report


def main() -> None:
    analysis = analyze_and_report(
        ROOT, PREREGISTRATION, FROZEN_CONFIG, EVALUATION_DIR, STAGE2
    )
    print(analysis["verdict"])
    print(analysis["decision"]["reason"])


if __name__ == "__main__":
    main()
