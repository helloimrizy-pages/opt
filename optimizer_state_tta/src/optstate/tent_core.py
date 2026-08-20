"""Online Tent step with the official prediction-before-update semantics.

Sequence per batch (``tent.forward_and_adapt``):

    predict -> record prediction -> entropy -> backward -> optimizer step

The recorded logits are the ones produced *before* the update.  Target labels
are used for scoring only and never touch the loss or any diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@dataclass
class BatchRecord:
    batch_index: int
    n: int
    n_correct: int
    entropy_loss: float
    mean_pred_entropy: float
    grad_norm: float
    post_update_correct: Optional[int] = None
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n


@torch.enable_grad()
def tent_step(model: nn.Module, optimizer: torch.optim.Optimizer,
              x: torch.Tensor, y: torch.Tensor, batch_index: int,
              record_grad_norm: bool = True,
              record_post_update: bool = False,
              record_state: bool = False) -> BatchRecord:
    """One online Tent update; returns metrics for the *pre-update* prediction.

    ``record_state`` additionally captures the pre-step optimizer state and its
    alignment with the current gradient.  It is read-only instrumentation.
    """
    outputs = model(x)
    with torch.no_grad():
        pred = outputs.argmax(1)
        n_correct = int((pred == y).sum().item())
        ent = softmax_entropy(outputs)
        mean_ent = float(ent.mean().item())

    loss = softmax_entropy(outputs).mean(0)
    loss.backward()

    grad_norm = float("nan")
    if record_grad_norm:
        with torch.no_grad():
            sq = torch.zeros((), device=x.device)
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        sq = sq + (p.grad.detach() ** 2).sum()
            grad_norm = float(sq.sqrt().item())

    extra: Dict[str, float] = {}
    if record_state:
        from . import adam_state as _A
        extra = _A.live_state_stats(optimizer, _A.flat_grads(optimizer))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    post_correct = None
    if record_post_update:
        with torch.no_grad():
            post_correct = int((model(x).argmax(1) == y).sum().item())

    return BatchRecord(
        batch_index=batch_index,
        n=int(y.numel()),
        n_correct=n_correct,
        entropy_loss=float(loss.detach().item()),
        mean_pred_entropy=mean_ent,
        grad_norm=grad_norm,
        post_update_correct=post_correct,
        extra=extra,
    )


@torch.no_grad()
def source_eval_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     batch_index: int) -> BatchRecord:
    outputs = model(x)
    pred = outputs.argmax(1)
    ent = softmax_entropy(outputs)
    return BatchRecord(
        batch_index=batch_index,
        n=int(y.numel()),
        n_correct=int((pred == y).sum().item()),
        entropy_loss=float(ent.mean().item()),
        mean_pred_entropy=float(ent.mean().item()),
        grad_norm=float("nan"),
    )


def run_domain(model: nn.Module, optimizer: torch.optim.Optimizer, stream,
               start: int, count: int, record_post_update: bool = False,
               record_state_first_k: int = 0) -> List[BatchRecord]:
    out: List[BatchRecord] = []
    for j, (k, x, y) in enumerate(stream.batches(start, count)):
        out.append(tent_step(model, optimizer, x, y, k,
                             record_post_update=record_post_update,
                             record_state=(j < record_state_first_k)))
    return out
