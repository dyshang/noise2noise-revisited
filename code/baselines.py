"""External baselines: BM3D and a minimal Noise2Void.

BM3D is a non-learning collaborative-filtering benchmark; it is given the test-
time sigma for Gaussian noise and a robust MAD-based sigma estimate otherwise.

Noise2Void is implemented as the standard blind-spot training scheme on a
single noisy image, using stratified random pixel substitution. The
implementation here is intentionally compact; it is sufficient to act as a
fair external reference, not state-of-the-art N2V.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from model import UNet


# ----------------------------------------------------------------------------
# BM3D wrapper
# ----------------------------------------------------------------------------


def _mad_sigma(img: np.ndarray) -> float:
    """Median-absolute-deviation noise sigma in the highest-frequency wavelet band.

    Donoho's wavelet-domain MAD estimator (sigma_hat = MAD/0.6745 on the
    diagonal detail coefficients) via ``skimage.restoration.estimate_sigma``,
    averaged over channels for colour input. Note: a naive row-difference MAD
    without the 1/sqrt(2) correction overestimates sigma by ~41% and picks up
    image gradients on top.
    """
    from skimage.restoration import estimate_sigma

    if img.ndim == 3:
        return float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))
    return float(estimate_sigma(img))


def bm3d_denoise(noisy: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Apply colour BM3D (CBM3D, luminance-chrominance transform) to RGB input,
    plain BM3D to single-channel input. Tensor in [0,1], shape (3,H,W) or (1,H,W)."""
    import bm3d

    arr = noisy.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = arr.transpose(1, 2, 0) if arr.shape[0] in (1, 3) else arr
    if sigma is None:
        sigma = _mad_sigma(arr) * 255.0
    if arr.shape[2] == 3:
        out = bm3d.bm3d_rgb(arr, sigma_psd=sigma / 255.0)
    else:
        out = bm3d.bm3d(arr[..., 0], sigma_psd=sigma / 255.0)[..., None]
    out = np.clip(out, 0.0, 1.0).astype(np.float32).transpose(2, 0, 1)
    return torch.from_numpy(out).to(noisy.device)


# ----------------------------------------------------------------------------
# Minimal Noise2Void
# ----------------------------------------------------------------------------


class _BlindSpotMasker:
    """Stratified random pixel substitution for Noise2Void training.

    For each patch we pick a fraction ``frac_masked`` of pixels; each is
    replaced by a random neighbour drawn uniformly from a
    ``(2*window+1)^2`` box centred on its location, and the loss is
    computed only at the masked positions. Krull et al. 2019 recommend
    0.5%--1.5% mask coverage; the previous default of 64 px in a
    128*128 patch (0.39%) was below that floor, so ~99.6% of training
    compute produced zero gradient -- the network was severely
    undersupervised.

    Implementation is fully vectorised via advanced indexing: a 32-batch
    with ~250 masks no longer round-trips 8000+ scalars to the host.
    """

    def __init__(self, frac_masked: float = 0.015, window: int = 5):
        if not 0.0 < frac_masked < 1.0:
            raise ValueError(f"frac_masked must be in (0,1), got {frac_masked}")
        self.frac_masked = frac_masked
        self.window = window

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        device = x.device
        n = max(1, int(round(self.frac_masked * h * w)))

        yy = torch.randint(0, h, (b, n), device=device)
        xx = torch.randint(0, w, (b, n), device=device)
        dy = torch.randint(-self.window, self.window + 1, (b, n), device=device)
        dx = torch.randint(-self.window, self.window + 1, (b, n), device=device)
        ny = (yy + dy).clamp_(0, h - 1)
        nx = (xx + dx).clamp_(0, w - 1)

        b_idx = torch.arange(b, device=device).unsqueeze(1).expand(-1, n)
        neighbour = x[b_idx, :, ny, nx]  # (b, n, c) via advanced indexing
        masked = x.clone()
        masked[b_idx, :, yy, xx] = neighbour
        loss_mask = torch.zeros((b, 1, h, w), device=device)
        loss_mask[b_idx, 0, yy, xx] = 1.0
        return masked, loss_mask


def n2v_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE evaluated only at the masked pixels."""
    diff = (pred - target) ** 2
    diff = diff.mean(dim=1, keepdim=True)
    denom = mask.sum().clamp(min=1.0)
    return (diff * mask).sum() / denom


def make_n2v_model() -> nn.Module:
    return UNet()
