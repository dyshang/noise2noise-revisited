"""U-Net backbone shared across all loss-function ablations.

The architecture is held identical across the L1/L2/Charbonnier/Huber and
Lasso-MSE configurations so that any difference in cross-noise generalisation
is attributable solely to the training objective.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(c_in: int, c_out: int, batchnorm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, 3, padding=1)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(c_out, c_out, 3, padding=1))
    if batchnorm:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """``residual=True`` predicts a correction added to the input (global skip):
    the identity mapping is then free, which direct image prediction through a
    bottleneck + BN struggles to learn at small step budgets (the v2 gate run
    plateaued at ~22 dB PSNR across all sigma). Old checkpoints predate these
    flags and load with residual=False, batchnorm=True.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        base: int = 48,
        depth: int = 4,
        residual: bool = False,
        batchnorm: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.residual = residual
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pools = nn.ModuleList()

        widths = [base * (2 ** i) for i in range(depth + 1)]

        prev = in_ch
        for w in widths[:-1]:
            self.downs.append(_conv_block(prev, w, batchnorm))
            self.pools.append(nn.MaxPool2d(2))
            prev = w

        self.bottleneck = _conv_block(prev, widths[-1], batchnorm)

        for i in range(depth - 1, -1, -1):
            self.ups.append(
                nn.ModuleDict(
                    {
                        "upconv": nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2),
                        "conv": _conv_block(widths[i + 1], widths[i], batchnorm),
                    }
                )
            )

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for block, pool in zip(self.downs, self.pools):
            h = block(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for block, skip in zip(self.ups, reversed(skips)):
            h = block["upconv"](h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block["conv"](h)
        out = self.head(h)
        return x + out if self.residual else out
