#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/stage1_prediction/src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/src"))

from race_stage1.archive import freeze_archive, verify_archive


def main() -> None:
    stage1 = REPOSITORY_ROOT / "stage3_residency/stage1_prediction"
    path = stage1 / "reports/final_archive_manifest.json"
    if path.exists():
        result = verify_archive(stage1)
        if not result["passed"]:
            raise SystemExit(f"sealed archive verification failed: {result['failures']}")
        print(f"archive already sealed and valid: {result['manifest_sha256']}")
        return
    result = freeze_archive(REPOSITORY_ROOT, stage1)
    verification = verify_archive(stage1)
    if not verification["passed"]:
        raise SystemExit(f"archive verification failed: {verification['failures']}")
    print(f"archive sealed: {result['manifest_sha256']}")


if __name__ == "__main__":
    main()
