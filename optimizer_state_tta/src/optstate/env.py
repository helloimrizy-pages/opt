"""Environment capture, seeding and determinism helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _pkg_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"MISSING ({type(exc).__name__})"
    v = getattr(module, "__version__", None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version as _v
        return str(_v(name))
    except Exception:  # pragma: no cover
        return "unknown"


def _robustbench_source() -> Dict[str, Any]:
    """RobustBench ships no ``__version__``; record the installed dist + VCS pin."""
    info: Dict[str, Any] = {}
    try:
        from importlib.metadata import distribution
        dist = distribution("robustbench")
        info["version"] = dist.version
        try:
            direct = dist.read_text("direct_url.json")
            if direct:
                info["direct_url"] = json.loads(direct)
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    return info


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        return True


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_description(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": device.type}
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["capability"] = list(torch.cuda.get_device_capability(device))
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
    elif device.type == "mps":
        info["name"] = platform.processor() or "apple-silicon"
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            )
            info["chip"] = out.stdout.strip()
        except Exception:  # pragma: no cover
            info["chip"] = "unknown"
        info["cuda_version"] = None
    else:
        info["name"] = platform.processor() or "cpu"
        info["cuda_version"] = None
    return info


def set_seed(seed: int) -> None:
    """Seed every RNG this study can touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:  # pragma: no cover
            pass


def enable_determinism(strict: bool = False) -> Dict[str, Any]:
    """Prefer deterministic kernels where the backend supports them.

    ``strict`` raises on non-deterministic ops.  On MPS PyTorch offers no
    deterministic-algorithm guarantee, so strict mode is only requested when
    explicitly asked for; the returned dictionary records what was actually
    achieved so the report can be honest about it.
    """
    achieved: Dict[str, Any] = {"strict_requested": strict}
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(strict, warn_only=not strict)
        achieved["use_deterministic_algorithms"] = strict
    except Exception as exc:  # pragma: no cover
        achieved["use_deterministic_algorithms"] = f"failed: {exc}"
    achieved["cudnn_deterministic"] = True
    achieved["note"] = (
        "PyTorch does not provide deterministic-algorithm guarantees on the MPS "
        "backend. Matched branches are made comparable by construction (identical "
        "weights, identical batch order) rather than by global determinism."
    )
    return achieved


def environment_record(seeds, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    device = select_device()
    record: Dict[str, Any] = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "torchvision_version": _pkg_version("torchvision"),
        "robustbench_version": _pkg_version("robustbench"),
        "robustbench_source": _robustbench_source(),
        "numpy_version": _pkg_version("numpy"),
        "scipy_version": _pkg_version("scipy"),
        "pandas_version": _pkg_version("pandas"),
        "matplotlib_version": _pkg_version("matplotlib"),
        "device": device_description(device),
        "seeds": list(seeds),
    }
    try:
        import robustbench  # noqa: F401
        rb_path = Path(robustbench.__file__).resolve().parents[1]
        record["robustbench_install_path"] = str(rb_path)
    except Exception:  # pragma: no cover
        pass
    if extra:
        record.update(extra)
    return record


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path
