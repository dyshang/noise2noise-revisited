"""Noise generators used for training and cross-noise evaluation.

Inputs are torch tensors in [0, 1]. Outputs are clamped back to [0, 1] so the
metrics operate on a consistent range. Each generator takes a `generator`
argument so train/test draws are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _sample(fn, *args, generator: torch.Generator | None, device, **kwargs) -> torch.Tensor:
    """Sample on the generator's device (default: CPU) then move to ``device``.

    PyTorch's MPS RNG is partial (no native poisson, partial randint), so we
    keep all randomness on CPU and transfer once. This is also a portability
    benefit: same numerics across CPU / MPS / CUDA.
    """
    gen_device = "cpu" if generator is None else str(generator.device)
    out = fn(*args, generator=generator, device=gen_device, **kwargs)
    return out.to(device) if str(out.device) != str(device) else out


def gaussian(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    noise = _sample(torch.randn, x.shape, generator=generator, device=x.device) * (sigma / 255.0)
    return (x + noise).clamp(0.0, 1.0)


def gaussian_range(
    x: torch.Tensor, sigma_lo: float, sigma_hi: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    gen_device = "cpu" if generator is None else str(generator.device)
    sigma = torch.empty(1, device=gen_device).uniform_(sigma_lo, sigma_hi, generator=generator).item()
    return gaussian(x, sigma, generator=generator)


def random_impulse(x: torch.Tensor, p: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Replace a fraction p of pixels with U(0,1) noise (per-channel-independent)."""
    mask = _sample(torch.rand, x.shape, generator=generator, device=x.device) < p
    rand_pix = _sample(torch.rand, x.shape, generator=generator, device=x.device)
    return torch.where(mask, rand_pix, x)


def salt_pepper(x: torch.Tensor, p: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Replace fraction p of pixels with 0 or 1 (50/50)."""
    u = _sample(torch.rand, x.shape, generator=generator, device=x.device)
    salt = u < (p / 2)
    pepper = (u >= (p / 2)) & (u < p)
    out = x.clone()
    out[salt] = 1.0
    out[pepper] = 0.0
    return out


def poisson(x: torch.Tensor, lam: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Shot-noise Poisson with mean rate `lam` * x. Larger lam = less noise."""
    scaled = (x * lam).cpu()
    noisy = torch.poisson(scaled, generator=generator).to(x.device)
    return (noisy / lam).clamp(0.0, 1.0)


def speckle(x: torch.Tensor, var: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Multiplicative noise: y = x + x*n, n ~ N(0, var)."""
    n = _sample(torch.randn, x.shape, generator=generator, device=x.device) * (var ** 0.5)
    return (x + x * n).clamp(0.0, 1.0)


@dataclass
class NoiseSpec:
    name: str
    fn: callable
    kwargs: dict

    def apply(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.fn(x, **self.kwargs, generator=generator)


def standard_eval_grid() -> list[NoiseSpec]:
    """The cross-noise evaluation grid used in the paper."""
    return [
        NoiseSpec("gauss15", gaussian, {"sigma": 15}),
        NoiseSpec("gauss25", gaussian, {"sigma": 25}),
        NoiseSpec("gauss50", gaussian, {"sigma": 50}),
        NoiseSpec("impulse10", random_impulse, {"p": 0.1}),
        NoiseSpec("impulse30", random_impulse, {"p": 0.3}),
        NoiseSpec("impulse60", random_impulse, {"p": 0.6}),
        NoiseSpec("sp05", salt_pepper, {"p": 0.05}),
        NoiseSpec("sp10", salt_pepper, {"p": 0.10}),
        NoiseSpec("poisson30", poisson, {"lam": 30.0}),
        NoiseSpec("poisson60", poisson, {"lam": 60.0}),
        NoiseSpec("speckle05", speckle, {"var": 0.05}),
        NoiseSpec("speckle10", speckle, {"var": 0.10}),
        NoiseSpec("speckle20", speckle, {"var": 0.20}),
        NoiseSpec("speckle50", speckle, {"var": 0.50}),
    ]
