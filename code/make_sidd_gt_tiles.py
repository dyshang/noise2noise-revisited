"""Extract clean GT tiles from the SIDD
training scenes so N2N can be trained on SIDD *scene content* with synthetic
Gaussian pairs. Disentangles scene-content adaptation from noise-distribution
adaptation in the ~8 dB regime gap.

Ground truth is used only as the clean base for synthetic pairs -- the same
role CBSD400 plays in the synthetic arm. The resulting model is a disclosed
diagnostic control, not a self-supervised result. Uses the same sorted-name
128-scene training split as RealN2NPairDataset; evaluation scenes are never
touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from data import _list_sidd_scenes, sidd_scene_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sidd-root", default="../data/SIDD_Medium_Srgb")
    p.add_argument("--out", default="../data/SIDD_GT_tiles_r4")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--tiles-per-image", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = _list_sidd_scenes(args.sidd_root)
    keep = set(sidd_scene_split([s.name for s in scenes], "train"))
    rng = np.random.default_rng(args.seed)

    n_written = 0
    for scene in scenes:
        if scene.name not in keep:
            continue
        for gt_path in sorted(scene.glob("*GT_SRGB_*.PNG")):
            img = np.asarray(Image.open(gt_path).convert("RGB"))
            h, w = img.shape[:2]
            for k in range(args.tiles_per_image):
                y = int(rng.integers(0, h - args.tile + 1))
                x = int(rng.integers(0, w - args.tile + 1))
                tile = img[y : y + args.tile, x : x + args.tile]
                name = f"{scene.name}__{gt_path.stem}__t{k}.png"
                Image.fromarray(tile).save(out / name)
                n_written += 1
    print(f"wrote {n_written} tiles ({args.tile}x{args.tile}) to {out}")


if __name__ == "__main__":
    main()
