"""Render the SIDD-vs-Kodak comparison figure (paper Fig. 2).

Horizontal grouped bars, one row per method, three panels (PSNR/SSIM/LPIPS)
sharing the method axis so each name is printed once, horizontally. Zebra
striping (every other row, extended under the y labels on the left panel)
keeps rows traceable across the three panels. X = metric
gain over the noisy input, averaged over the Kodak synthetic grid (blue)
versus the official SIDD validation blocks (orange). Above the dashed
separator: baselines + the five synth-Gaussian-trained N2N variants (strong
on synthetic, +0.8..+3.7 dB on real). Below (shaded): the real-noise-trained
block -- N2V on single SIDD shots plus the same five N2N losses on SIDD
noisy/noisy pairs (+9..+11 dB on real, at the cost of synthetic
performance). Only the real-noise (orange) bars carry number annotations;
the caption says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


METHODS = [
    ("bm3d", "BM3D"),
    ("n2v", "Noise2Void"),
    ("n2n_l2", "N2N-MSE"),
    ("n2n_l2_lasso", "N2N-MSE+Lasso"),
    ("n2n_l1", "N2N-MAE"),
    ("n2n_charb", "N2N-Charb."),
    ("n2n_huber_d005", "N2N-Huber"),
    ("n2v_real", "N2V-real"),
    ("n2n_l2_real", "N2N-MSE-real"),
    ("n2n_l2_lasso_real", "N2N-MSE+Lasso-real"),
    ("n2n_l1_real", "N2N-MAE-real"),
    ("n2n_charb_real", "N2N-Charb.-real"),
    ("n2n_huber_d005_real", "N2N-Huber-real"),
]
# Index where the synth-trained block ends and the real-noise-trained block
# begins. A dashed separator + light shading marks the real block; the
# rendering does not depend on this being a particular N2N variant -- it's
# purely a layout hint.
REAL_GROUP_START = 7

KODAK_COLS = [
    "gauss15", "gauss25", "gauss50",
    "impulse10", "impulse30", "impulse60",
    "sp05", "sp10",
    "poisson30", "poisson60",
    "speckle05", "speckle10", "speckle20", "speckle50",
]
SIDD_COL = "sidd_val_blocks"  # official validation blocks (literature protocol)


def gain(summary: dict, method: str, col: str, metric: str) -> float:
    """PSNR/SSIM/LPIPS gain over the noisy input. Sign convention matches metric."""
    v_method = summary[method][col][metric]
    v_noisy = summary["noisy"][col][metric]
    return (v_method - v_noisy) if metric in ("psnr", "ssim") else (v_noisy - v_method)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/cross_noise_kodak.json")
    p.add_argument(
        "--blocks-results",
        default=None,
        help="eval_sidd_blocks.py JSON supplying the official-blocks column",
    )
    p.add_argument("--out", default="figures/sidd_vs_kodak.png")
    args = p.parse_args()

    with open(args.results) as f:
        summary = json.load(f)
    if args.blocks_results is not None:
        with open(args.blocks_results) as f:
            for method, by_noise in json.load(f).items():
                summary.setdefault(method, {}).update(by_noise)

    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    metrics = [
        ("psnr", "PSNR gain (dB)"),
        ("ssim", "SSIM gain"),
        ("lpips", "LPIPS reduction"),
    ]

    method_keys = [m[0] for m in METHODS]
    method_labels = [m[1] for m in METHODS]
    y = np.arange(len(method_keys))
    height = 0.36
    n = len(method_keys)

    for i, (ax, (metric, xlabel)) in enumerate(zip(axes, metrics)):
        kodak_vals = np.array([
            np.mean([gain(summary, m, c, metric) for c in KODAK_COLS])
            for m in method_keys
        ])
        sidd_vals = np.array([
            gain(summary, m, SIDD_COL, metric) for m in method_keys
        ])

        b1 = ax.barh(y - height / 2, kodak_vals, height,
                     label="Kodak24 (synthetic, mean)", color="#3b7dd8")
        b2 = ax.barh(y + height / 2, sidd_vals, height,
                     label="SIDD validation blocks (real)", color="#e0703c")

        ax.axvline(0, color="black", linewidth=0.8)
        # Shading + dashed line mark the real-noise-trained block.
        ax.axhspan(REAL_GROUP_START - 0.5, n - 0.5, color="grey", alpha=0.08, zorder=0)
        ax.axhline(REAL_GROUP_START - 0.5, color="grey", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        # Zebra stripes on every other row keep rows traceable across the
        # three panels; on the leftmost panel they extend under the y tick
        # labels so label->row assignment is unambiguous.
        for yi in range(0, n, 2):
            if i == 0:
                ax.axhspan(yi - 0.5, yi + 0.5, xmin=-0.35, xmax=1,
                           color="#4a6fa5", alpha=0.08, zorder=0, clip_on=False)
            else:
                ax.axhspan(yi - 0.5, yi + 0.5, color="#4a6fa5", alpha=0.08,
                           zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(method_labels, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        # Extra top margin holds the synth-group caption; headroom on the
        # right holds the horizontal bar annotations.
        ax.set_ylim(n - 0.5, -1.35)
        right = max(kodak_vals.max(), sidd_vals.max())
        left = min(0.0, kodak_vals.min(), sidd_vals.min())
        ax.set_xlim(left * 1.1 if left < 0 else 0, right * 1.24)

        # Annotate with numbers for clarity (only the SIDD bars). Horizontal
        # text at the bar end, nudged down into the gap below the orange bar
        # so it clears the blue bar of the same row.
        for yi, val in zip(y, sidd_vals):
            ax.annotate(
                f"{val:+.2f}",
                xy=(max(val, 0), yi + height / 2),
                xytext=(3, -1.5),
                textcoords="offset points",
                ha="left", va="center", fontsize=9,
            )

    # Group captions once, on the leftmost panel: x in axes fraction (stays
    # right-aligned regardless of data range), y in data coords (tracks rows).
    ax0 = axes[0]
    blend = mtransforms.blended_transform_factory(ax0.transAxes, ax0.transData)
    ax0.text(0.97, -0.75, "trained on synthetic Gaussian", transform=blend,
             fontsize=10.5, style="italic", color="dimgrey", ha="right", va="center")
    ax0.text(0.97, REAL_GROUP_START - 0.5 + 0.42, "trained on real SIDD pairs",
             transform=blend, fontsize=10.5, style="italic", color="dimgrey",
             ha="right", va="center")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=14, frameon=False)
    # Manual margins: tight_layout mis-handles sharey + clip_on=False spans
    # and clips the longest y label.
    fig.subplots_adjust(left=0.142, right=0.995, top=0.90, bottom=0.115,
                        wspace=0.06)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
