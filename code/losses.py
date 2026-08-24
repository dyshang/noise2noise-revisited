"""Loss functions for the residual term, with the likelihood interpretation
spelled out in comments.

These are *pixel-domain* losses on the residual r = pred - target. They are
distinct from any weight regularisation on the network parameters; the latter
can be added separately via `weight_decay` (L2 / ridge) or via `lasso_penalty`
in the trainer (L1 / Lasso).
"""

from __future__ import annotations

import torch
from torch import nn


class L2Loss(nn.Module):
    """MSE; MLE under Gaussian residual; conditional-mean estimator."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean()


class L1Loss(nn.Module):
    """MAE; MLE under Laplace residual; conditional-median estimator.

    Heavy-tailed residual likelihood => bounded influence per pixel =>
    robustness to impulse-like / saturated outliers at test time.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()


class CharbonnierLoss(nn.Module):
    """sqrt(r^2 + eps^2). Smooth surrogate to L1 with everywhere-defined gradient."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class HuberLoss(nn.Module):
    """Quadratic for |r| <= delta, linear beyond. Interpolates MSE and MAE."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.smooth_l1_loss(pred, target, beta=self.delta)


def lasso_penalty(model: nn.Module) -> torch.Tensor:
    """Sum of |w| over all conv layer weights. Use to study the *separate*
    sparsity-inducing effect of L1 weight regularisation. This is what the
    Eqs. 2-7 in the original draft were actually describing.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total = total + m.weight.abs().sum()
    return total


LOSS_FACTORY = {
    "l2": L2Loss,
    "l1": L1Loss,
    "charbonnier": CharbonnierLoss,
    "huber": HuberLoss,
}


def make_loss(name: str, huber_delta: float = 1.0) -> nn.Module:
    if name not in LOSS_FACTORY:
        raise KeyError(f"unknown loss '{name}', expected one of {list(LOSS_FACTORY)}")
    if name == "huber":
        # On [0,1] images residuals never exceed 1, so the default delta=1.0
        # keeps smooth-L1 permanently in its quadratic regime (== 0.5*MSE).
        # delta must sit at the residual scale (~2 sigma) for the loss to
        # actually interpolate between the L2 and L1 regimes.
        return HuberLoss(delta=huber_delta)
    return LOSS_FACTORY[name]()
