"""Stage 1 phenomenon-validation utilities for optimizer-state memory under CTTA.

This package is a *diagnostic* toolkit.  It deliberately contains no new
optimizer, no gradient-similarity moment decay and no adaptive reset policy.
Every state intervention implemented here is an oracle-boundary diagnostic used
to measure whether carried Adam state causally changes early adaptation.
"""

__all__ = [
    "adam_state",
    "data",
    "diagnostics",
    "env",
    "metrics",
    "model",
    "stats",
    "tent_core",
]
