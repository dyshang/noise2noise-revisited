"""Are SIDD's two noisy shots per scene independent?

N2N assumes the noise in the two shots is independent given the signal. Real
cameras can share fixed-pattern noise / row noise / PRNU between consecutive
shots; any shared component is *kept* by N2N training rather than removed.
This script quantifies the violation directly:

  r_i = noisy_i - GT_i          (per scene, native resolution, float32)

* cross-shot Pearson corr(r_0, r_1) on a common random pixel subsample
  -> shared (scene-static) noise component
* lag-1 spatial autocorrelation of each residual (horizontal / vertical,
  on a centre crop) -> within-shot spatial correlation, the quantity that
  breaks blind-spot pixel-independence assumptions

GT is read for *analysis only*; training never touches it.

Usage::

    python sidd_shot_correlation.py --root ../data/SIDD_Medium_Srgb \
        --out results/sidd_shot_correlation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data import _index_by_shot, _list_sidd_scenes, sidd_scene_split

Image.MAX_IMAGE_PIXELS = None


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _lag1(r: np.ndarray) -> tuple[float, float]:
    """Lag-1 spatial autocorrelation (horizontal, vertical), channels pooled."""
    h = _pearson(r[:, :-1, :].ravel(), r[:, 1:, :].ravel())
    v = _pearson(r[:-1, :, :].ravel(), r[1:, :, :].ravel())
    return h, v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="../data/SIDD_Medium_Srgb")
    p.add_argument("--split", default="all", choices=["all", "train", "eval"])
    p.add_argument("--subsample", type=int, default=2_000_000, help="pixels for cross-shot corr")
    p.add_argument("--crop", type=int, default=1024, help="centre-crop side for autocorrelation")
    p.add_argument("--out", default="results/sidd_shot_correlation.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    scenes = _list_sidd_scenes(args.root)
    keep = set(sidd_scene_split([s.name for s in scenes], args.split))

    records = []
    for scene in scenes:
        if scene.name not in keep:
            continue
        gts = _index_by_shot(sorted(scene.glob("*GT_SRGB_*.PNG")))
        noisy = _index_by_shot(sorted(scene.glob("*NOISY_SRGB_*.PNG")))
        shots = sorted(gts.keys() & noisy.keys())
        if len(shots) < 2:
            continue
        s0, s1 = shots[:2]
        residuals = []
        stats = {}
        for tag, sid in (("r0", s0), ("r1", s1)):
            n = _load(noisy[sid]).astype(np.float32)
            g = _load(gts[sid]).astype(np.float32)
            r = (n - g) / 255.0
            del n, g
            stats[f"{tag}_std"] = float(r.std())
            ch, cv = _lag1(_centre_crop(r, args.crop))
            stats[f"{tag}_lag1_h"] = ch
            stats[f"{tag}_lag1_v"] = cv
            residuals.append(r)
        r0, r1 = residuals
        hh = min(r0.shape[0], r1.shape[0])
        ww = min(r0.shape[1], r1.shape[1])
        flat0 = r0[:hh, :ww].reshape(-1)
        flat1 = r1[:hh, :ww].reshape(-1)
        idx = rng.choice(flat0.size, size=min(args.subsample, flat0.size), replace=False)
        cross = _pearson(flat0[idx], flat1[idx])
        del r0, r1, flat0, flat1
        records.append({"scene": scene.name, "cross_corr": cross, **stats})
        print(
            f"{scene.name}: cross={cross:+.4f} "
            f"lag1_h={stats['r0_lag1_h']:+.3f}/{stats['r1_lag1_h']:+.3f} "
            f"lag1_v={stats['r0_lag1_v']:+.3f}/{stats['r1_lag1_v']:+.3f} "
            f"std={stats['r0_std']:.4f}/{stats['r1_std']:.4f}",
            flush=True,
        )

    def agg(key: str) -> dict[str, float]:
        vals = np.array([rec[key] for rec in records], dtype=np.float64)
        return {"mean": float(vals.mean()), "std": float(vals.std()), "max_abs": float(np.abs(vals).max())}

    summary = {
        "n_scenes": len(records),
        "cross_corr": agg("cross_corr"),
        "lag1_h": agg("r0_lag1_h"),
        "lag1_v": agg("r0_lag1_v"),
        "residual_std": agg("r0_std"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"summary": summary, "scenes": records}, f, indent=2)
    print("\nSUMMARY", json.dumps(summary, indent=2))


def _centre_crop(r: np.ndarray, side: int) -> np.ndarray:
    h, w = r.shape[:2]
    if h <= side and w <= side:
        return r
    i = max(0, (h - side) // 2)
    j = max(0, (w - side) // 2)
    return r[i : i + side, j : j + side]


if __name__ == "__main__":
    main()
