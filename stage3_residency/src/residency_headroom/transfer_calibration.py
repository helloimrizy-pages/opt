from __future__ import annotations

import platform
import time
from typing import Any, Sequence

import numpy as np
import torch


def measure_host_device_expert_transfer(
    expert_bytes: int,
    *,
    repeats: int = 100,
    warmup: int = 20,
    modes: Sequence[str] = ("pinned", "pageable"),
) -> dict[str, Any]:
    """Measure isolated tensor-copy cost, excluding allocation and model execution."""

    if expert_bytes < 1 or repeats < 2 or warmup < 1:
        raise ValueError("Invalid transfer-calibration size/repeat settings")
    if not torch.cuda.is_available():
        return {
            "schema_version": "race_stage0_transfer_calibration_v1",
            "available": False,
            "reason": "CUDA is not available in the active PyTorch environment",
            "host": platform.platform(),
            "expert_bytes": expert_bytes,
            "label": "measured host-device expert transfer cost",
            "not_end_to_end_inference_latency": True,
        }
    device = torch.device("cuda")
    elements = int(np.ceil(expert_bytes / 2))
    actual_bytes = elements * 2
    output: list[dict[str, Any]] = []
    for mode in modes:
        if mode not in {"pinned", "pageable"}:
            raise ValueError(f"Unknown host memory mode {mode!r}")
        pinned = mode == "pinned"
        host_source = torch.empty(elements, dtype=torch.float16, pin_memory=pinned)
        host_destination = torch.empty(elements, dtype=torch.float16, pin_memory=pinned)
        device_tensor = torch.empty(elements, dtype=torch.float16, device=device)
        host_source.fill_(0.25)
        for direction in ("host_to_device", "device_to_host"):
            for _ in range(warmup):
                if direction == "host_to_device":
                    device_tensor.copy_(host_source, non_blocking=pinned)
                else:
                    host_destination.copy_(device_tensor, non_blocking=pinned)
                torch.cuda.synchronize(device)
            samples = np.empty(repeats, dtype=np.float64)
            for index in range(repeats):
                torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                if direction == "host_to_device":
                    device_tensor.copy_(host_source, non_blocking=pinned)
                else:
                    host_destination.copy_(device_tensor, non_blocking=pinned)
                torch.cuda.synchronize(device)
                samples[index] = (time.perf_counter_ns() - started) / 1e6
            median_ms = float(np.median(samples))
            output.append(
                {
                    "direction": direction,
                    "host_memory": mode,
                    "requested_bytes": expert_bytes,
                    "actual_tensor_bytes": actual_bytes,
                    "warmup": warmup,
                    "repeats": repeats,
                    "median_ms": median_ms,
                    "mean_ms": float(samples.mean()),
                    "std_ms": float(samples.std(ddof=1)),
                    "p05_ms": float(np.quantile(samples, 0.05)),
                    "p25_ms": float(np.quantile(samples, 0.25)),
                    "p75_ms": float(np.quantile(samples, 0.75)),
                    "p95_ms": float(np.quantile(samples, 0.95)),
                    "median_effective_gib_per_s": (
                        actual_bytes / (1024**3) / (median_ms / 1000.0)
                    ),
                    "samples_ms": samples.tolist(),
                }
            )
    return {
        "schema_version": "race_stage0_transfer_calibration_v1",
        "available": True,
        "label": "measured host-device expert transfer cost",
        "not_end_to_end_inference_latency": True,
        "allocation_overhead_included": False,
        "synchronization": "torch.cuda.synchronize before and after every timed copy",
        "device": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "expert_bytes": expert_bytes,
        "measurements": output,
    }
