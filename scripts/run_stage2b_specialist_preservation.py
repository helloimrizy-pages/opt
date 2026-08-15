#!/usr/bin/env python3
"""Stage 2B end-to-end orchestrator.

Stages:
  local        scores -> frozen allocations -> held-out splits -> audit
  development  development 20% evaluation -> analysis -> audit
  final        final frozen sweep (requires FULL_EVALUATION_GO) -> analysis -> audit

The local stage runs on any machine. The evaluation stages require the pinned
OLMoE checkpoint and, for production results, a CUDA device.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts"


def run(command: list[str]) -> None:
    print(f"\n=== {' '.join(command)} ===", flush=True)
    result = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("local", "development", "final"), default="local"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--determinism-warn-only", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    results_dir = str(args.results_dir)
    cache = ["--cache-dir", args.cache_dir] if args.cache_dir else []

    if args.stage == "local":
        run([python, str(SCRIPTS / "build_specialist_scores.py"),
             "--output-dir", results_dir])
        run([python, str(SCRIPTS / "solve_protection_allocations.py"),
             "--output-dir", results_dir])
        run([python, str(SCRIPTS / "build_heldout_splits.py"),
             "--output-dir", results_dir, *cache])
        run([python, str(SCRIPTS / "audit_specialist_preservation.py"),
             "--results-dir", results_dir])
        print(
            "\nLocal Stage 2B artifacts are frozen. Run --stage development on "
            "the CUDA machine next."
        )
        return 0

    evaluation = [
        python, str(SCRIPTS / "evaluate_protection_allocations.py"),
        "--phase", args.stage, "--results-dir", results_dir,
        "--device", args.device, "--batch-size", str(args.batch_size), *cache,
    ]
    if args.determinism_warn_only:
        evaluation.append("--determinism-warn-only")
    run(evaluation)
    run([python, str(SCRIPTS / "analyze_protection_results.py"),
         "--phase", args.stage, "--results-dir", results_dir])
    run([python, str(SCRIPTS / "audit_specialist_preservation.py"),
         "--results-dir", results_dir])
    if args.stage == "development":
        decision = json.loads(
            (args.results_dir / "stage2b_decision.json").read_text()
        )["decision"]
        print(f"\nDevelopment decision: {decision}")
        if decision == "FULL_EVALUATION_GO":
            print("Run --stage final to evaluate the frozen sweep.")
        else:
            print(
                "ROBUST_PRESERVATION_NO_GO: stop here. Preserve all results; do "
                "not alter the objective, budgets, or methods, and do not "
                "inspect the final evaluation set."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
