"""Fit the Stage 2 Markov transition models from calibration data only."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import CALIBRATION_DIR, PREREGISTRATION, ROOT

from race_stage2.calibration import ensure_transition_models
from race_stage2.frozen import load_and_verify_stage2_inputs


def main() -> None:
    inputs = load_and_verify_stage2_inputs(ROOT, PREREGISTRATION)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    _models, metadata, audit = ensure_transition_models(
        inputs, ROOT, CALIBRATION_DIR / "transition_models.npz"
    )
    print(f"transition models: {metadata['npz_sha256']}")
    print(f"stage1 horizon reuse audit: {audit}")


if __name__ == "__main__":
    main()
