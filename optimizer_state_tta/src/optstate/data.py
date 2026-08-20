"""CIFAR-10-C stream construction.

Loading semantics deliberately mirror ``robustbench.data.load_corruptions_cifar``:
severity ``s`` selects rows ``[(s-1)*10000 : s*10000]`` of the per-corruption
``.npy`` array, images are transposed to NCHW and scaled by ``1/255``.  The only
addition is an explicit, seeded sample permutation so that data order can be
varied and recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch

# The conventional corruption ordering used by the official Tent CIFAR-10-C
# example (``conf.py`` ``_C.CORRUPTION.TYPE``) and by the CoTTA/CTTA literature.
CONVENTIONAL_ORDER: Tuple[str, ...] = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)

N_PER_CORRUPTION = 10_000


class Cifar10CStore:
    """Lazy, memory-cached access to CIFAR-10-C arrays kept as ``uint8``."""

    def __init__(self, data_dir: str | Path, severity: int = 5):
        self.root = Path(data_dir) / "CIFAR-10-C"
        if not self.root.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C not found at {self.root}. Run the download step first."
            )
        self.severity = int(severity)
        self._labels = np.load(self.root / "labels.npy")[:N_PER_CORRUPTION]
        self._cache: dict[Tuple[str, int], np.ndarray] = {}

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def images(self, corruption: str, severity: int | None = None) -> np.ndarray:
        sev = self.severity if severity is None else int(severity)
        key = (corruption, sev)
        if key not in self._cache:
            path = self.root / f"{corruption}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"missing corruption file {path}")
            arr = np.load(path, mmap_mode="r")
            lo, hi = (sev - 1) * N_PER_CORRUPTION, sev * N_PER_CORRUPTION
            self._cache[key] = np.ascontiguousarray(arr[lo:hi])
        return self._cache[key]


def permutation_for(seed: int, corruption: str, severity: int,
                    n: int = N_PER_CORRUPTION) -> np.ndarray:
    """Deterministic per-(seed, domain) sample order.

    Seed 0 keeps the official unshuffled order so the reproduction protocol and
    the primary experiment share one stream at the reference seed; seeds >= 1
    use independent permutations.
    """
    if seed == 0:
        return np.arange(n)
    ss = np.random.SeedSequence(entropy=[seed, abs(hash((corruption, severity))) % (2**31)])
    return np.random.default_rng(ss).permutation(n)


@dataclass(frozen=True)
class DomainSpec:
    corruption: str
    severity: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.corruption}:s{self.severity}"


class DomainStream:
    """Fixed-order batch stream over one corruption domain."""

    def __init__(self, store: Cifar10CStore, spec: DomainSpec, batch_size: int,
                 device: torch.device):
        self.store = store
        self.spec = spec
        self.batch_size = int(batch_size)
        self.device = device
        self.order = permutation_for(spec.seed, spec.corruption, spec.severity)
        self._images = store.images(spec.corruption, spec.severity)
        self._labels = store.labels

    @property
    def n_batches(self) -> int:
        return len(self.order) // self.batch_size

    def batch(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.n_batches:
            raise IndexError(index)
        lo = index * self.batch_size
        idx = self.order[lo: lo + self.batch_size]
        imgs = self._images[idx]                       # (B, 32, 32, 3) uint8
        x = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
        y = self._labels[idx].astype(np.int64)
        return (
            torch.from_numpy(np.ascontiguousarray(x)).to(self.device),
            torch.from_numpy(y).to(self.device),
        )

    def batches(self, start: int, count: int) -> Iterator[Tuple[int, torch.Tensor, torch.Tensor]]:
        for k in range(start, min(start + count, self.n_batches)):
            x, y = self.batch(k)
            yield k, x, y


def corruption_orders(n_permutations: int = 3, base_seed: int = 20240,
                      order: Sequence[str] = CONVENTIONAL_ORDER) -> List[dict]:
    """Conventional order plus ``n_permutations`` fixed, seeded permutations."""
    out = [{"name": "conventional", "perm_seed": None, "order": list(order)}]
    for k in range(n_permutations):
        seed = base_seed + k
        rng = np.random.default_rng(seed)
        perm = list(np.array(order)[rng.permutation(len(order))])
        out.append({"name": f"perm{k + 1}", "perm_seed": seed, "order": perm})
    return out
