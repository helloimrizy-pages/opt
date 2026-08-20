"""Stage 2 calibration and configuration freeze."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from _bootstrap import CALIBRATION_DIR, PREREGISTRATION, ROOT

from race_stage2.calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    outcome = run_calibration(
        ROOT, PREREGISTRATION, CALIBRATION_DIR, workers=arguments.workers
    )
    frozen = outcome["frozen_config"]
    print("frozen Stage 2 configuration:")
    print(f"  file sha256          : {outcome['frozen_config_file_sha256']}")
    print(f"  selected eta         : {frozen['selected_eta']}")
    print(f"  initialization       : {frozen['selected_initialization']}")
    print(f"  primary loss         : {frozen['primary_loss']}")
    print(f"  primary variant      : {frozen['primary_variant_id']}")
    print(f"  transition models    : {frozen['transition_model_hash']}")
    print(f"  source bundle        : {frozen['stage2_source_bundle_hash']}")


if __name__ == "__main__":
    main()
