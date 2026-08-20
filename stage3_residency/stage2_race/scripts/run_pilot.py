"""Stage 2 pilot audit on a prefix of the frozen calibration path."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401
from _bootstrap import CALIBRATION_DIR, PILOT_DIR, PREREGISTRATION, ROOT

from race_stage2.audit import run_pilot_audit


def main() -> None:
    report = run_pilot_audit(
        ROOT,
        PREREGISTRATION,
        CALIBRATION_DIR / "transition_models.npz",
        PILOT_DIR,
    )
    for check in report["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}")
    print(f"pilot events: {report['events']}")
    print(f"runtime: {report['runtime_seconds_with_diagnostics']:.1f}s "
          f"({report['microseconds_per_event']:.0f} us/event)")
    print(f"pilot costs: {json.dumps(report['pilot_costs'])}")
    if not report["passed"]:
        raise SystemExit("Stage 2 pilot audit failed")
    print("PILOT AUDIT PASSED")


if __name__ == "__main__":
    main()
