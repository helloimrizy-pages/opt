"""Preregistered outcome metrics.

Primary: ``early10_accuracy`` = mean online top-1 accuracy over the first 10
adaptation batches after a boundary, using pre-update predictions.  With batch
size 200 that is the first 2,000 post-boundary samples.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

WINDOWS: Sequence[int] = (1, 5, 10, 25, 50)
PRIMARY_WINDOW = 10

# Preregistered recovery-time rule (fixed before any curve was inspected):
#   plateau        = CARRY_ALL accuracy over the last 10 batches of the branch window
#   threshold      = plateau - RECOVERY_TOLERANCE
#   recovery_batch = smallest k >= 1 such that the mean accuracy of batches
#                    k .. k + RECOVERY_WINDOW - 1 is >= threshold; censored value
#                    RECOVERY_CENSORED if the condition never holds.
RECOVERY_TOLERANCE = 0.02
RECOVERY_WINDOW = 5
RECOVERY_CENSORED = 999


def window_accuracy(records: Sequence, k: int) -> Optional[float]:
    sel = records[:k]
    if len(sel) < k:
        return None
    n = sum(r.n for r in sel)
    c = sum(r.n_correct for r in sel)
    return c / n if n else None


def accuracy_curve(records: Sequence) -> List[float]:
    return [r.n_correct / r.n for r in records]


def window_metrics(records: Sequence, windows: Iterable[int] = WINDOWS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in windows:
        acc = window_accuracy(records, k)
        out[f"acc_first{k}"] = acc if acc is not None else float("nan")
        out[f"err_first{k}"] = (1.0 - acc) if acc is not None else float("nan")
    n = sum(r.n for r in records)
    c = sum(r.n_correct for r in records)
    out["acc_full"] = c / n if n else float("nan")
    out["cumulative_errors_first10"] = float(
        sum(r.n - r.n_correct for r in records[:10])
    )
    out["mean_entropy_loss_first10"] = (
        sum(r.entropy_loss for r in records[:10]) / max(1, len(records[:10]))
    )
    out["mean_pred_entropy_first10"] = (
        sum(r.mean_pred_entropy for r in records[:10]) / max(1, len(records[:10]))
    )
    out["mean_grad_norm_first10"] = (
        sum(r.grad_norm for r in records[:10]) / max(1, len(records[:10]))
    )
    return out


def plateau_accuracy(records: Sequence, last: int = 10) -> float:
    sel = records[-last:]
    n = sum(r.n for r in sel)
    c = sum(r.n_correct for r in sel)
    return c / n if n else float("nan")


def recovery_batch(records: Sequence, threshold: float,
                   window: int = RECOVERY_WINDOW) -> int:
    curve = accuracy_curve(records)
    for k in range(len(curve) - window + 1):
        if sum(curve[k:k + window]) / window >= threshold:
            return k + 1
    return RECOVERY_CENSORED


def collapse_indicator(records: Sequence, floor: float = 0.15) -> Dict[str, float]:
    """Crude model-collapse flags: accuracy at/near chance late in the window."""
    curve = accuracy_curve(records)
    tail = curve[-5:] if len(curve) >= 5 else curve
    return {
        "min_batch_accuracy": min(curve) if curve else float("nan"),
        "tail_accuracy": sum(tail) / len(tail) if tail else float("nan"),
        "collapsed": float(bool(tail and (sum(tail) / len(tail)) <= floor)),
    }
