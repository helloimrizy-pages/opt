"""One frozen Stage 2 evaluation over the ten sealed Stage 0 workload paths."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from _bootstrap import EVALUATION_DIR, FROZEN_CONFIG, PREREGISTRATION, ROOT

from race_stage2.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    arguments = parser.parse_args()
    manifest = run_evaluation(
        ROOT,
        PREREGISTRATION,
        FROZEN_CONFIG,
        EVALUATION_DIR,
        workers=arguments.workers,
        labels=arguments.labels,
    )
    print(f"conditions        : {manifest['conditions']}")
    print(f"per-sequence rows : {manifest['per_sequence_rows']}")
    print(f"results sha256    : {manifest['results_sha256']}")


if __name__ == "__main__":
    main()
