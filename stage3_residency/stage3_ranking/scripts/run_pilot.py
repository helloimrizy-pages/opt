"""Stage 3 pilot audit on a prefix of the frozen calibration path."""
from __future__ import annotations
import _bootstrap  # noqa: F401
from _bootstrap import FROZEN_CONFIG, PILOT_DIR, PREREGISTRATION, ROOT
from race_stage3.audit import run_pilot_audit


def main() -> None:
    report = run_pilot_audit(ROOT, PREREGISTRATION, PILOT_DIR,
                             frozen_config_path=FROZEN_CONFIG)
    for check in report["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}")
    print(f"events {report['events']:,}  {report['microseconds_per_event']:.0f} us/event")
    print("stage1 pilot:", report["stage1_reference_pilot_costs"])
    if report["frozen_primary_pilot_costs"]:
        print("stage3 pilot:", report["frozen_primary_pilot_costs"])
    if not report["passed"]:
        raise SystemExit("Stage 3 pilot audit failed")
    print("PILOT AUDIT PASSED")


if __name__ == "__main__":
    main()
