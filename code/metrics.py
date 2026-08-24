"""PSNR / SSIM / LPIPS computed on raw network output.

No brightness, contrast, sharpness or colour adjustment is applied to `pred`
before metric computation. This is a deliberate departure from the prior
draft's PIL `ImageEnhance` post-processing, which inflated PSNR by ~3 dB and
made cross-method comparison ill-defined.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(_psnr(_to_np(target), _to_np(pred), data_range=1.0))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        _ssim(
            _to_np(target),
            _to_np(pred),
            channel_axis=2,
            data_range=1.0,
        )
    )


@lru_cache(maxsize=1)
def _lpips_model(device: str = "cpu"):
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def lpips_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    device = pred.device
    model = _lpips_model(str(device))
    p = (pred.unsqueeze(0).clamp(0.0, 1.0) * 2 - 1).to(device)
    t = (target.unsqueeze(0).clamp(0.0, 1.0) * 2 - 1).to(device)
    with torch.no_grad():
        d = model(p, t)
    return float(d.item())


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "lpips": lpips_score(pred, target),
    }
