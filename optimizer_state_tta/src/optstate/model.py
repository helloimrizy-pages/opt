"""Source model loading and Tent model configuration.

``configure_tent_model`` and ``collect_bn_params`` reproduce the official Tent
reference implementation (DequanWang/tent ``tent.configure_model`` /
``tent.collect_params``) exactly: train mode, all gradients off except
BatchNorm2d affine scale/shift, and running statistics discarded so every batch
is normalised with its own statistics.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_ARCH = "Standard"          # RobustBench WideResNet-28-10
AUGMIX_ARCH = "Hendrycks2020AugMix_WRN"   # WRN-40-2, the second Tent example


def load_source_model(arch: str, ckpt_dir: str, device: torch.device) -> nn.Module:
    from robustbench.utils import load_model
    model = load_model(model_name=arch, model_dir=ckpt_dir,
                       dataset="cifar10", threat_model="corruptions")
    return model.to(device)


def configure_tent_model(model: nn.Module) -> nn.Module:
    """Exact port of ``tent.configure_model``."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_bn_params(model: nn.Module) -> Tuple[List[torch.nn.Parameter], List[str]]:
    """Exact port of ``tent.collect_params``."""
    params: List[torch.nn.Parameter] = []
    names: List[str] = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np_, p in m.named_parameters():
                if np_ in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def check_tent_config(model: nn.Module) -> dict:
    """Assertions from ``tent.check_model`` plus a few extras used in tests."""
    is_training = model.training
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any = any(param_grads)
    has_all = all(param_grads)
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    stats_off = all(
        m.running_mean is None and m.running_var is None
        for m in model.modules() if isinstance(m, nn.BatchNorm2d)
    )
    return {
        "training": is_training,
        "some_params_require_grad": has_any,
        "not_all_params_require_grad": not has_all,
        "has_batchnorm2d": has_bn,
        "bn_running_stats_disabled": stats_off,
        "n_trainable_tensors": sum(1 for p in model.parameters() if p.requires_grad),
        "n_trainable_scalars": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def make_adam(params: Sequence[torch.nn.Parameter], lr: float, beta1: float,
              beta2: float, weight_decay: float) -> torch.optim.Adam:
    """Tent's optimiser: ``Adam(params, lr, betas=(BETA, 0.999), weight_decay=WD)``."""
    return torch.optim.Adam(list(params), lr=lr, betas=(beta1, beta2),
                            weight_decay=weight_decay)
