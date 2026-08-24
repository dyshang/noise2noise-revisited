"""Cross-noise generalisation evaluation, with external baselines.

For each (model, noise) pair we report PSNR, SSIM and LPIPS, computed on the
*raw* network output. No brightness, contrast, sharpness or colour correction
is applied; the prior draft's PIL `ImageEnhance` step has been deliberately
removed because it inflates PSNR without improving the actual denoising
output, and is a confound for cross-method comparison.

Usage example::

    python evaluate.py \
        --eval-root ../data/Kodak24 \
        --sidd-root ../data/SIDD_Medium_Srgb \
        --noise2noise-ckpts \
            l1=ckpts/n2n_l1.pt l2=ckpts/n2n_l2.pt \
            l1_real=ckpts/n2n_l1_real.pt l2_real=ckpts/n2n_l2_real.pt \
        --n2v-ckpt ckpts/n2v.pt \
        --bm3d \
        --out results/cross_noise_kodak.json

The SIDD eval set is the deterministic 32-scene held-out split of
SIDD-Medium (see ``data.sidd_scene_split``); the same split's *training*
scenes are used for ``train.py --real-pairs ...``, so training and eval
scenes never overlap.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from baselines import bm3d_denoise
from data import CleanEvalDataset, RealNoisePairDataset
from metrics import all_metrics
from model import UNet
from noise import standard_eval_grid

SIDD_KEY = "sidd_real"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-root",
        default="../data/Kodak24",
        help="clean image folder for synthetic-noise eval (relative to code/)",
    )
    p.add_argument(
        "--sidd-root",
        default=None,
        help=(
            "SIDD-Medium sRGB root for real-noise eval, e.g. "
            "../data/SIDD_Medium_Srgb. Pass to enable; omit to skip. "
            "Evaluation uses the held-out 32-scene split (see "
            "data.sidd_scene_split); training scenes are excluded."
        ),
    )
    p.add_argument(
        "--sidd-split",
        default="eval",
        choices=["eval", "train", "all"],
        help="which SIDD split to evaluate on (default: eval = 32 held-out scenes)",
    )
    p.add_argument("--sidd-max-side", type=int, default=1024, help="downsample SIDD long side")
    p.add_argument("--sidd-patch", type=int, default=None, help="optional centre-crop patch on SIDD")
    p.add_argument(
        "--noise2noise-ckpts",
        nargs="*",
        default=[],
        help="space-separated name=path pairs, e.g. l1=ckpts/n2n_l1.pt",
    )
    p.add_argument("--n2v-ckpt", default=None)
    p.add_argument("--bm3d", action="store_true", help="include BM3D baseline")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )
    p.add_argument("--max-side", type=int, default=512, help="downsample large eval images")
    p.add_argument(
        "--tile",
        type=int,
        default=0,
        help=(
            "tiled inference window for large images (0 = whole image). "
            "Use ~512 for native-resolution SIDD on a 10 GB GPU."
        ),
    )
    p.add_argument(
        "--skip-synth",
        action="store_true",
        help=(
            "skip the synthetic Kodak grid and evaluate only the SIDD split. "
            "Lets slow passes (BM3D) run as separate stages whose results "
            "persist independently -- a 7h combined pass that dies late "
            "loses everything."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_unet(ckpt_path: str, device: str) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    ckpt_args = state.get("args", {})
    m = UNet(
        residual=ckpt_args.get("residual", False),
        batchnorm=not ckpt_args.get("no_bn", False),
    ).to(device).eval()
    m.load_state_dict(state["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _denoise(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    tile: int = 0,
    tile_overlap: int = 64,
) -> torch.Tensor:
    """Forward pass, optionally tiled for images too large for one shot.

    ``tile=0`` disables tiling. Otherwise the image is processed in
    ``tile``-sized windows with ``tile_overlap`` context on each side;
    only the interior of each window is written to the output, so seams
    carry full receptive-field context and no blending weights are needed.
    """
    if tile <= 0 or (noisy.shape[-2] <= tile and noisy.shape[-1] <= tile):
        with torch.no_grad():
            out = model(noisy.unsqueeze(0))
        return out.squeeze(0).clamp(0.0, 1.0)

    _, h, w = noisy.shape
    ov = tile_overlap
    out = torch.empty_like(noisy)
    with torch.no_grad():
        for ti in range(0, h, tile):
            for tj in range(0, w, tile):
                i0, i1 = max(ti - ov, 0), min(ti + tile + ov, h)
                j0, j1 = max(tj - ov, 0), min(tj + tile + ov, w)
                pred = model(noisy[:, i0:i1, j0:j1].unsqueeze(0)).squeeze(0)
                ie, je = min(ti + tile, h), min(tj + tile, w)
                out[:, ti:ie, tj:je] = pred[:, ti - i0 : ie - i0, tj - j0 : je - j0]
    return out.clamp(0.0, 1.0)


def _eval_pair(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    *,
    noise_key: str,
    image_name: str,
    n2n_models: dict[str, torch.nn.Module],
    n2v_model: torch.nn.Module | None,
    use_bm3d: bool,
    bm3d_sigma: float | None,
    results: dict[str, dict[str, list[dict[str, float]]]],
    tile: int = 0,
) -> None:
    """Run every method on one (noisy, clean) pair and record metrics.

    Shared between the synthetic-noise grid and the SIDD real-noise loop so
    the two paths cannot drift apart in metric / pre-processing logic.
    Each record carries the image name so per-image paired statistics
    (e.g. Wilcoxon signed-rank across methods) can be computed downstream.
    """
    for tag, model in n2n_models.items():
        pred = _denoise(model, noisy, tile=tile)
        results[tag][noise_key].append({"image": image_name, **all_metrics(pred, clean)})
    if n2v_model is not None:
        pred = _denoise(n2v_model, noisy, tile=tile)
        results["n2v"][noise_key].append({"image": image_name, **all_metrics(pred, clean)})
    if use_bm3d:
        pred = bm3d_denoise(noisy, sigma=bm3d_sigma)
        results["bm3d"][noise_key].append({"image": image_name, **all_metrics(pred, clean)})
    results["noisy"][noise_key].append({"image": image_name, **all_metrics(noisy, clean)})


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    # CPU generator with explicit transfer is the most portable option:
    # MPS RNG implementations are partial (e.g. randint, poisson) in current PyTorch.
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    n2n_models = {}
    for spec in args.noise2noise_ckpts:
        if "=" not in spec:
            raise ValueError(f"expected name=path, got '{spec}'")
        name, path = spec.split("=", 1)
        n2n_models[f"n2n_{name}"] = _load_unet(path, args.device)

    n2v_model = _load_unet(args.n2v_ckpt, args.device) if args.n2v_ckpt else None

    grid = standard_eval_grid()

    # results[method][noise_name] = list of per-image metric dicts
    results: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    ds = [] if args.skip_synth else CleanEvalDataset(args.eval_root, max_side=args.max_side)
    for clean, name in tqdm(ds, desc="synthetic"):
        clean = clean.to(args.device)
        for spec in grid:
            noisy = spec.apply(clean, generator=gen)
            sigma_arg = spec.kwargs.get("sigma") if spec.name.startswith("gauss") else None
            _eval_pair(
                noisy,
                clean,
                noise_key=spec.name,
                image_name=name,
                n2n_models=n2n_models,
                n2v_model=n2v_model,
                use_bm3d=args.bm3d,
                bm3d_sigma=sigma_arg,
                results=results,
                tile=args.tile,
            )

    if args.sidd_root is not None:
        sidd = RealNoisePairDataset(
            args.sidd_root,
            max_side=args.sidd_max_side,
            patch_size=args.sidd_patch,
            split=args.sidd_split,
        )
        print(f"[sidd] {len(sidd)} (noisy, gt) pairs, split={args.sidd_split!r}")
        for noisy, clean, pair_name in tqdm(sidd, desc="sidd"):
            noisy = noisy.to(args.device)
            clean = clean.to(args.device)
            # BM3D with unknown real-noise sigma: pass None so the wrapper falls
            # back to the MAD-based estimator. Do *not* use a synthetic sigma.
            _eval_pair(
                noisy,
                clean,
                noise_key=SIDD_KEY,
                image_name=pair_name,
                n2n_models=n2n_models,
                n2v_model=n2v_model,
                use_bm3d=args.bm3d,
                bm3d_sigma=None,
                results=results,
                tile=args.tile,
            )

    # Aggregate means (schema consumed by render_tables.py, unchanged) and
    # dump the raw per-image records to a sibling file for paired statistics.
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method, by_noise in results.items():
        summary[method] = {}
        for noise_name, lst in by_noise.items():
            keys = [k for k in lst[0] if k != "image"]
            agg = {k: sum(d[k] for d in lst) / len(lst) for k in keys}
            summary[method][noise_name] = agg

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    per_image_out = out.with_suffix(".per_image.json")
    with per_image_out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[out] summary -> {out}\n[out] per-image -> {per_image_out}")

    _print_table(summary)


def _print_table(summary: dict[str, dict[str, dict[str, float]]]) -> None:
    methods = list(summary.keys())
    # Union over all methods so SIDD-only / synthetic-only entries both render.
    noises: list[str] = []
    for by_noise in summary.values():
        for n in by_noise:
            if n not in noises:
                noises.append(n)
    for metric in ("psnr", "ssim", "lpips"):
        print(f"\n=== {metric.upper()} (rows: method, columns: test noise) ===")
        header = f"{'method':<12}" + "".join(f"{n:>11}" for n in noises)
        print(header)
        for m in methods:
            row = f"{m:<12}"
            for n in noises:
                cell = summary[m].get(n, {}).get(metric)
                row += f"{cell:>11.3f}" if cell is not None else f"{'-':>11}"
            print(row)


if __name__ == "__main__":
    main()
