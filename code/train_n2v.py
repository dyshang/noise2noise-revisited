"""Train a Noise2Void baseline on a single set of noisy images.

This is included so the cross-noise generalisation table can include a fair
external self-supervised reference. The implementation is intentionally
compact: standard U-Net + blind-spot masking + MSE loss on the masked
positions only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torchvision.transforms.functional as TF
from PIL import Image

from baselines import _BlindSpotMasker, make_n2v_model, n2v_loss
from data import RealN2NPairDataset, _list_images
from noise import gaussian_range


class _N2VDataset(Dataset):
    """N2V training pairs.

    ``length`` sets the virtual-epoch size; ``__getitem__(i)`` cycles
    through the images via ``i % len(paths)``. Each call returns a
    fresh random crop with optional 8x augmentation (h/v flip + 90 deg
    rotation), matching Krull et al. 2019. The previous version drew
    only ``len(paths)`` crops per epoch with no augmentation, which
    -- combined with the under-coverage masker -- left N2V badly
    undertrained.
    """

    def __init__(
        self,
        root: str,
        patch: int = 128,
        sigma_lo: float = 5.0,
        sigma_hi: float = 50.0,
        length: int | None = None,
        augment: bool = True,
    ):
        self.paths = _list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")
        self.patch = patch
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi
        self.length = length if length is not None else len(self.paths)
        self.augment = augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> torch.Tensor:
        i = i % len(self.paths)
        img = Image.open(self.paths[i]).convert("RGB")
        clean = TF.to_tensor(img)
        _, h, w = clean.shape
        if h < self.patch or w < self.patch:
            clean = TF.resize(clean, [max(self.patch, h), max(self.patch, w)], antialias=True)
            _, h, w = clean.shape
        i0 = torch.randint(0, h - self.patch + 1, (1,)).item()
        j0 = torch.randint(0, w - self.patch + 1, (1,)).item()
        crop = clean[:, i0 : i0 + self.patch, j0 : j0 + self.patch]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                crop = torch.flip(crop, dims=[2])
            if torch.rand(1).item() < 0.5:
                crop = torch.flip(crop, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                crop = torch.rot90(crop, k=k, dims=[1, 2])
        return gaussian_range(crop, self.sigma_lo, self.sigma_hi)


class _N2VRealDataset(Dataset):
    """N2V training crops from real SIDD noisy shots (no synthetic noise,
    GT never read). Wraps ``RealN2NPairDataset`` for the scene split and
    lazy-mmap shot access and keeps only one of the two crops it yields.
    Same 8x augmentation as the synthetic ``_N2VDataset`` so the two N2V
    arms differ only in their noise source.
    """

    def __init__(
        self,
        root: str,
        patch: int = 128,
        length: int = 5000,
        npy_root: str | None = None,
        augment: bool = True,
    ):
        self.inner = RealN2NPairDataset(
            root, patch_size=patch, length=length, split="train", npy_root=npy_root
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> torch.Tensor:
        crop, _ = self.inner[i]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                crop = torch.flip(crop, dims=[2])
            if torch.rand(1).item() < 0.5:
                crop = torch.flip(crop, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                crop = torch.rot90(crop, k=k, dims=[1, 2])
        return crop.contiguous()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default="../data/CBSD400",
        help="folder of *colour* training images, relative to code/ "
             "(grayscale sets silently reproduce the channel-statistics "
             "confound described in the paper)",
    )
    p.add_argument(
        "--real-noisy",
        default=None,
        help=(
            "SIDD-Medium root: train N2V directly on real noisy shots "
            "(128 train scenes, GT never read) instead of synthetic Gaussian "
            "on --data-root"
        ),
    )
    p.add_argument(
        "--sidd-npy-root",
        default=None,
        help="native-resolution mmap tree from prep_sidd_npy.py (with --real-noisy)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--sigma-lo", type=float, default=5.0)
    p.add_argument("--sigma-hi", type=float, default=50.0)
    p.add_argument(
        "--length",
        type=int,
        default=2000,
        help="virtual epoch length (random crops per epoch). With CBSD400 and "
             "default batch=32 this is ~62 steps/epoch, vs the old 12 -- the "
             "old setting + 0.39%% mask coverage left N2V undersupervised.",
    )
    p.add_argument(
        "--frac-masked",
        type=float,
        default=0.015,
        help="fraction of pixels masked per patch (Krull 2019: 0.005--0.015)",
    )
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="disable h/v flip + 90 deg rotation augmentation",
    )
    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; default 0 for macOS stability, 2-4 on Win/Linux",
    )
    args = p.parse_args()

    if args.real_noisy is not None:
        ds: Dataset = _N2VRealDataset(
            args.real_noisy,
            patch=args.patch_size,
            length=args.length,
            npy_root=args.sidd_npy_root,
            augment=not args.no_augment,
        )
        print(f"[real-noisy] {len(ds.inner.scenes)} train scenes, length={len(ds)}")
    else:
        ds = _N2VDataset(
            args.data_root,
            args.patch_size,
            args.sigma_lo,
            args.sigma_hi,
            length=args.length,
            augment=not args.no_augment,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    masker = _BlindSpotMasker(frac_masked=args.frac_masked)
    model = make_n2v_model().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for x in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            x = x.to(args.device, non_blocking=True)
            masked, mask = masker(x)
            pred = model(masked)
            loss = n2v_loss(pred, x, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        avg = running / max(n, 1)
        history.append({"epoch": epoch, "loss": avg})
        print(f"epoch {epoch+1}: loss={avg:.5f}")

    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, out)
    with out.with_suffix(".json").open("w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
