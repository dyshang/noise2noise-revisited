"""Paired significance tests between two methods from a per-image metrics dump.

Reads a ``*.per_image.json`` file and runs a Wilcoxon signed-rank test per
(noise, metric) cell, pairing records by image name. Two schemas are accepted:

* evaluate.py:          data[method][noise] = [ {image, psnr, ssim, lpips}, ... ]
* eval_sidd_blocks.py:  data[method]        = [ {image, psnr, ssim, lpips}, ... ]
  (flat; auto-wrapped under the single key ``sidd_val_blocks``)

``--group-blocks`` aggregates block records ``imgNN_blkMM`` to per-image means
before testing (n=40 for the official SIDD validation set) -- blocks of one
image are not independent samples, so image-level pairing is the conservative
choice for the paper.

All p-values additionally get a Holm-Bonferroni correction across every
(noise, metric) cell tested in the run, reported as ``p_holm``.

Usage::

    python stats_tests.py results/cross_noise_kodak_v4.per_image.json \
        --a n2n_l1 --b n2n_l2
    python stats_tests.py results/sidd_val_blocks_v4.per_image.json \
        --a n2n_l2 --b n2n_l1 --group-blocks
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from scipy.stats import wilcoxon

HIGHER_BETTER = {"psnr": True, "ssim": True, "lpips": False}


def _normalise(data: dict) -> dict:
    """Wrap the flat blocks schema so both inputs look like data[m][noise]."""
    out = {}
    for method, v in data.items():
        out[method] = {"sidd_val_blocks": v} if isinstance(v, list) else v
    return out


def _group_blocks(records: list[dict]) -> list[dict]:
    """Aggregate imgNN_blkMM records to one mean record per image."""
    by_img: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_img[r["image"].split("_blk")[0]].append(r)
    out = []
    for img, lst in sorted(by_img.items()):
        keys = [k for k in lst[0] if k != "image"]
        out.append({"image": img, **{k: sum(d[k] for d in lst) / len(lst) for k in keys}})
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("per_image_json")
    p.add_argument("--a", required=True, help="method key, e.g. n2n_l1")
    p.add_argument("--b", required=True, help="method key, e.g. n2n_l2")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--group-blocks",
        action="store_true",
        help="aggregate imgNN_blkMM records to per-image means before pairing",
    )
    args = p.parse_args()

    with open(args.per_image_json) as f:
        data = _normalise(json.load(f))
    if args.a not in data or args.b not in data:
        raise SystemExit(f"methods available: {sorted(data)}")

    rows = []  # (noise, metric, n, mean_delta, p_raw)
    for noise in data[args.a]:
        if noise not in data[args.b]:
            continue
        recs_a, recs_b = data[args.a][noise], data[args.b][noise]
        if args.group_blocks:
            recs_a, recs_b = _group_blocks(recs_a), _group_blocks(recs_b)
        rec_a = {r["image"]: r for r in recs_a}
        rec_b = {r["image"]: r for r in recs_b}
        common = sorted(rec_a.keys() & rec_b.keys())
        for metric in HIGHER_BETTER:
            deltas = [rec_a[i][metric] - rec_b[i][metric] for i in common]
            if not deltas or all(d == 0 for d in deltas):
                continue
            stat = wilcoxon(deltas)
            rows.append((noise, metric, len(deltas), sum(deltas) / len(deltas), stat.pvalue))

    adj = holm([r[4] for r in rows])

    print(
        f"Wilcoxon signed-rank, {args.a} vs {args.b} (positive delta = A better); "
        f"Holm over {len(rows)} tests"
    )
    header = (
        f"{'noise':<16}{'metric':<8}{'n':>5}{'mean dA-B':>12}"
        f"{'p':>10}{'p_holm':>10}  verdict"
    )
    print(header)
    print("-" * len(header))
    for (noise, metric, n, mean_d, p_raw), p_h in zip(rows, adj):
        better = (mean_d > 0) == HIGHER_BETTER[metric]
        tag = "A>B" if better else "B>A"
        sig = "*" if p_h < args.alpha else ("+" if p_raw < args.alpha else " (n.s.)")
        print(
            f"{noise:<16}{metric:<8}{n:>5}{mean_d:>12.4f}"
            f"{p_raw:>10.4f}{p_h:>10.4f}  {tag}{sig}"
        )
    print("\n*: significant after Holm at alpha; +: raw p < alpha only")


if __name__ == "__main__":
    main()
