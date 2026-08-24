"""Qualitative comparison figure: zoomed crops across methods.

Produces a grid: rows = scenes (one synthetic Kodak, one real SIDD), columns
= clean / noisy / BM3D / N2V / selected N2N variants. Each cell is a zoomed
crop of the raw method output -- consistent with the paper's no-post-
processing protocol.

Usage::

    python make_qualitative.py \
        --kodak-image ../data/Kodak24/kodim05.png --kodak-noise impulse30 \
        --sidd-scene-index 0 \
        --noise2noise-ckpts MSE=ckpts/n2n_l2_v2.pt MAE=ckpts/n2n_l1_v2.pt \
            MSE-real=ckpts/n2n_l2_real_v2.pt \
        --n2v-ckpt ckpts/n2v_v2.pt --bm3d \
        --crop 160 --out ../paper/figures/qualitative.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from baselines import bm3d_denoise
from data import RealNoisePairDataset
from evaluate import _denoise, _load_unet
from metrics import psnr
from noise import standard_eval_grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kodak-image", default="../data/Kodak24/kodim05.png")
    p.add_argument("--kodak-noise", default="impulse30")
    p.add_argument("--sidd-root", default="../data/SIDD_Medium_Srgb")
    p.add_argument("--sidd-scene-index", type=int, default=0, help="index into eval-split pairs")
    p.add_argument("--noise2noise-ckpts", nargs="*", default=[], help="label=path pairs")
    p.add_argument("--n2v-ckpt", default=None)
    p.add_argument("--bm3d", action="store_true")
    p.add_argument("--crop", type=int, default=160, help="zoom crop side")
    p.add_argument("--crop-xy", type=int, nargs=2, default=None, help="crop top-left (i j); default centre")
    p.add_argument("--tile", type=int, default=512, help="tiled inference for large SIDD images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../paper/figures/qualitative.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    models = {}
    for spec in args.noise2noise_ckpts:
        label, path = spec.split("=", 1)
        models[label] = _load_unet(path, args.device)
    n2v = _load_unet(args.n2v_ckpt, args.device) if args.n2v_ckpt else None

    rows = []  # (row_title, [(col_title, chw tensor in [0,1]), ...])

    # --- synthetic row -------------------------------------------------------
    clean = TF.to_tensor(Image.open(args.kodak_image).convert("RGB")).to(args.device)
    spec = next(s for s in standard_eval_grid() if s.name == args.kodak_noise)
    noisy = spec.apply(clean, generator=gen)
    row = [("clean (GT)", clean), (f"noisy ({args.kodak_noise})", noisy)]
    if args.bm3d:
        row.append(("BM3D", bm3d_denoise(noisy, sigma=None)))
    if n2v is not None:
        row.append(("N2V", _denoise(n2v, noisy)))
    for label, m in models.items():
        row.append((label, _denoise(m, noisy)))
    rows.append((Path(args.kodak_image).stem, [(t, x, psnr(x, clean)) for t, x in row]))

    # --- real SIDD row -------------------------------------------------------
    sidd = RealNoisePairDataset(args.sidd_root, max_side=None, split="eval")
    noisy_r, clean_r, name = sidd[args.sidd_scene_index]
    noisy_r, clean_r = noisy_r.to(args.device), clean_r.to(args.device)
    row = [("clean (GT)", clean_r), ("noisy (SIDD)", noisy_r)]
    if args.bm3d:
        row.append(("BM3D", bm3d_denoise(noisy_r, sigma=None)))
    if n2v is not None:
        row.append(("N2V", _denoise(n2v, noisy_r, tile=args.tile)))
    for label, m in models.items():
        row.append((label, _denoise(m, noisy_r, tile=args.tile)))
    rows.append((name, [(t, x, psnr(x, clean_r)) for t, x in row]))

    # --- render --------------------------------------------------------------
    ncols = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(
        len(rows), ncols, figsize=(1.9 * ncols, 2.15 * len(rows)), squeeze=False
    )
    cs = args.crop
    for ri, (row_name, cells) in enumerate(rows):
        _, ref, _ = cells[0]
        h, w = ref.shape[-2:]
        if args.crop_xy is not None:
            ci, cj = args.crop_xy
        else:
            ci, cj = (h - cs) // 2, (w - cs) // 2
        ci, cj = max(0, min(ci, h - cs)), max(0, min(cj, w - cs))
        for xi in range(ncols):
            ax = axes[ri][xi]
            ax.axis("off")
            if xi >= len(cells):
                continue
            title, img, val = cells[xi]
            crop = img[:, ci : ci + cs, cj : cj + cs].clamp(0, 1)
            ax.imshow(crop.permute(1, 2, 0).cpu().numpy())
            label = title if xi == 0 else f"{title}\n{val:.2f} dB"
            ax.set_title(label, fontsize=11)
    fig.tight_layout(pad=0.4, h_pad=2.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
