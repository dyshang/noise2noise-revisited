"""Evaluate on the official SIDD validation blocks (sRGB).

This is the evaluation protocol used across the SIDD literature: 1280 blocks
of 256x256 (40 images x 32 blocks), distributed as
``ValidationNoisyBlocksSrgb.mat`` / ``ValidationGtBlocksSrgb.mat``. Scoring
here makes our numbers directly comparable to published methods, unlike the
project-internal 32-scene split (which remains useful because its training
counterpart is scene-disjoint by construction).

Usage::

    python eval_sidd_blocks.py \
        --blocks-dir ../data/SIDD_Blocks \
        --noise2noise-ckpts l2_real_v2=ckpts/n2n_l2_real_v2.pt ... \
        --bm3d \
        --out results/sidd_val_blocks_v2.json

Metrics are computed on the raw network output per block, then averaged.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from baselines import bm3d_denoise
from evaluate import _denoise, _load_unet
from metrics import all_metrics


def _load_blocks(path: Path) -> np.ndarray:
    """Return uint8 array of shape (n_images, n_blocks, 256, 256, 3)."""
    try:
        from scipy.io import loadmat

        mat = loadmat(str(path))
        key = next(k for k in mat if not k.startswith("__"))
        arr = mat[key]
    except NotImplementedError:  # v7.3 -> HDF5
        import h5py

        with h5py.File(path, "r") as f:
            key = next(iter(f.keys()))
            arr = np.array(f[key]).transpose()  # h5py stores MATLAB arrays reversed
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.5 else (
            np.clip(arr, 0.0, 1.0) * 255
        ).astype(np.uint8)
    return arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks-dir", default="../data/SIDD_Blocks")
    p.add_argument(
        "--noise2noise-ckpts",
        nargs="*",
        default=[],
        help="space-separated name=path pairs, e.g. l2_real_v2=ckpts/n2n_l2_real_v2.pt",
    )
    p.add_argument("--n2v-ckpt", default=None)
    p.add_argument("--bm3d", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    blocks_dir = Path(args.blocks_dir)
    noisy = _load_blocks(blocks_dir / "ValidationNoisyBlocksSrgb.mat")
    gt = _load_blocks(blocks_dir / "ValidationGtBlocksSrgb.mat")
    assert noisy.shape == gt.shape, (noisy.shape, gt.shape)
    n_img, n_blk = noisy.shape[:2]
    print(f"[blocks] {n_img} images x {n_blk} blocks, {noisy.shape[2:]}")

    n2n_models = {}
    for spec in args.noise2noise_ckpts:
        name, path = spec.split("=", 1)
        n2n_models[f"n2n_{name}"] = _load_unet(path, args.device)
    n2v_model = _load_unet(args.n2v_ckpt, args.device) if args.n2v_ckpt else None

    results: dict[str, list[dict[str, float]]] = defaultdict(list)
    for i in tqdm(range(n_img), desc="images"):
        for b in range(n_blk):
            x = torch.from_numpy(noisy[i, b]).permute(2, 0, 1).float().div_(255.0).to(args.device)
            y = torch.from_numpy(gt[i, b]).permute(2, 0, 1).float().div_(255.0).to(args.device)
            name = f"img{i:02d}_blk{b:02d}"
            for tag, model in n2n_models.items():
                pred = _denoise(model, x)
                results[tag].append({"image": name, **all_metrics(pred, y)})
            if n2v_model is not None:
                pred = _denoise(n2v_model, x)
                results["n2v"].append({"image": name, **all_metrics(pred, y)})
            if args.bm3d:
                pred = bm3d_denoise(x, sigma=None)
                results["bm3d"].append({"image": name, **all_metrics(pred, y)})
            results["noisy"].append({"image": name, **all_metrics(x, y)})

    summary = {}
    for method, lst in results.items():
        keys = [k for k in lst[0] if k != "image"]
        summary[method] = {"sidd_val_blocks": {k: sum(d[k] for d in lst) / len(lst) for k in keys}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    with out.with_suffix(".per_image.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'method':<22}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
    for m, v in summary.items():
        s = v["sidd_val_blocks"]
        print(f"{m:<22}{s['psnr']:>8.2f}{s['ssim']:>8.4f}{s['lpips']:>8.4f}")


if __name__ == "__main__":
    main()
