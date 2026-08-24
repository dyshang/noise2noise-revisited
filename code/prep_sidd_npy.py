"""Decode SIDD-Medium noisy PNGs once into uint8 .npy files for mmap training.

Native-resolution real-pair training cannot RAM-cache all shots (~13 GB >
WSL2 budget) and per-step PNG decode of ~5328x3000 images is far too slow.
np.load(mmap_mode='r') on pre-decoded arrays gives random-crop reads that
touch only the crop's pages: near-RAM speed, near-zero resident memory.

Usage::

    python prep_sidd_npy.py --sidd-root ../data/SIDD_Medium_Srgb \
        --npy-root ../data/SIDD_Medium_Srgb_npy

Idempotent: existing .npy files are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from data import _list_sidd_scenes, sidd_scene_split

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sidd-root", default="../data/SIDD_Medium_Srgb")
    p.add_argument("--npy-root", default="../data/SIDD_Medium_Srgb_npy")
    p.add_argument("--noisy-glob", default="*NOISY_SRGB_*.PNG")
    p.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "eval"],
        help="limit conversion to one side of the deterministic scene split "
        "(e.g. --split train for GT shots that only training needs)",
    )
    args = p.parse_args()

    scenes = _list_sidd_scenes(args.sidd_root)
    if not scenes:
        raise SystemExit(f"no scenes under {args.sidd_root}")
    if args.split != "all":
        keep = set(sidd_scene_split([s.name for s in scenes], args.split))
        scenes = [s for s in scenes if s.name in keep]
    npy_root = Path(args.npy_root)

    todo: list[tuple[Path, Path]] = []
    for scene in scenes:
        for png in sorted(scene.glob(args.noisy_glob)):
            dst = npy_root / scene.name / f"{png.stem}.npy"
            if not dst.exists():
                todo.append((png, dst))
    print(f"{len(todo)} shots to convert ({len(scenes)} scenes)")

    for png, dst in tqdm(todo, desc="decode->npy"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
        # np.save silently appends ".npy" to bare paths; write via handle so
        # the tmp file keeps its exact name and the final rename is atomic.
        tmp = dst.parent / (dst.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        tmp.replace(dst)


if __name__ == "__main__":
    main()
