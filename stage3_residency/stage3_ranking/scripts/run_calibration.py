"""Stage 3 calibration and configuration freeze."""
from __future__ import annotations
import _bootstrap  # noqa: F401
from _bootstrap import CALIBRATION_DIR, PREREGISTRATION, ROOT
from race_stage3.calibration import run_calibration


def main() -> None:
    outcome = run_calibration(ROOT, PREREGISTRATION, CALIBRATION_DIR)
    frozen = outcome["frozen_config"]
    print("frozen Stage 3 configuration:")
    print(f"  file sha256    : {outcome['frozen_config_file_sha256']}")
    print(f"  primary variant: {frozen['primary_variant']}")
    print(f"  source bundle  : {frozen['stage3_source_bundle_hash']}")


if __name__ == "__main__":
    main()
