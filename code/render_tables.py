"""Render booktabs LaTeX tables from a cross-noise results JSON.

Usage:
    python render_tables.py --results results/cross_noise_kodak_v4.json \
                            --out ../paper/tables.tex

The output is three full-width ``table*`` environments (PSNR / SSIM / LPIPS),
matching the captions and labels referenced from the paper source.

Method rows are organised in four groups separated by ``\\midrule``:
    1. Noisy input (no method).
    2. Non-learning + self-supervised baselines (BM3D, Noise2Void).
    3. Synthetic-Gaussian-trained N2N variants (5 losses).
    4. SIDD-Medium real-noise-trained N2N variants (5 losses, ``_real`` ckpts).

Groups 2-4 form the candidate set for bolding; bold marks the per-column
best across every method except the raw noisy row (BM3D included -- the
caption says "best method", not "best learned method").

Rows whose method key is missing from the input JSON are skipped silently
and an empty group does not emit a stray midrule, so the same template
renders correctly whether or not the real-trained block has been run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Method groups: (label_or_None, [(json_key, latex_name), ...]).
# Group label is unused for rendering today but documents intent.
GROUPS: list[tuple[str | None, list[tuple[str, str]]]] = [
    (None, [("noisy", r"\emph{noisy input}")]),
    ("baselines", [
        ("bm3d", "BM3D"),
        ("n2v", "Noise2Void"),
    ]),
    ("n2n-synth", [
        ("n2n_l2", "N2N-MSE"),
        ("n2n_l2_lasso", r"\;\;+ Lasso"),
        ("n2n_l1", "N2N-MAE"),
        ("n2n_charb", "N2N-Charb."),
        ("n2n_huber_d005", "N2N-Huber"),
    ]),
    ("n2n-real", [
        ("n2v_real", "Noise2Void (real)"),
        ("n2n_l2_real", "N2N-MSE (real)"),
        ("n2n_l2_lasso_real", r"\;\;+ Lasso (real)"),
        ("n2n_l1_real", "N2N-MAE (real)"),
        ("n2n_charb_real", "N2N-Charb. (real)"),
        ("n2n_huber_d005_real", "N2N-Huber (real)"),
    ]),
]

# All learned methods (everything except the raw noisy row).
LEARNED = [k for label, rows in GROUPS if label is not None for k, _ in rows]

SYNTH_COLS = [
    ("gauss15", r"$\sigma{=}15$"),
    ("gauss25", r"$\sigma{=}25$"),
    ("gauss50", r"$\sigma{=}50$"),
    ("impulse10", r"imp\,10"),
    ("impulse30", r"imp\,30"),
    ("impulse60", r"imp\,60"),
    ("sp05", r"sp\,5"),
    ("sp10", r"sp\,10"),
    ("poisson30", r"poi\,30"),
    ("poisson60", r"poi\,60"),
    ("speckle05", r"spk\,5"),
    ("speckle10", r"spk\,10"),
    ("speckle20", r"spk\,20"),
    ("speckle50", r"spk\,50"),
]
SIDD = ("sidd_real", "SIDD")
SIDD_VAL = ("sidd_val_blocks", r"SIDD\,val")

# Column subset for the compact layout's SSIM/LPIPS table (one representative
# per noise family plus both real-noise columns).
COMPACT_COLS = [
    ("gauss25", r"$\sigma{=}25$"),
    ("impulse30", r"imp\,30"),
    ("sp10", r"sp\,10"),
    ("poisson60", r"poi\,60"),
    ("speckle10", r"spk\,10"),
    ("speckle50", r"spk\,50"),
]


def fmt(v: float | None, metric: str) -> str:
    if v is None:
        return "--"
    return f"{v:.2f}" if metric == "psnr" else f"{v:.3f}"


def best_in_col(summary: dict, metric: str, col_key: str, lower_better: bool) -> str | None:
    pairs = [(m, summary[m][col_key][metric]) for m in LEARNED if col_key in summary.get(m, {})]
    if not pairs:
        return None
    return (min if lower_better else max)(pairs, key=lambda kv: kv[1])[0]


def render_one(
    summary: dict,
    metric: str,
    lower_better: bool,
    caption: str,
    label: str,
    cols: list[tuple[str, str]] | None = None,
    n_real_cols: int = 1,
    size: str = r"\small",
) -> str:
    if cols is None:
        cols = SYNTH_COLS + [SIDD]
    spec = "l" + "r" * len(cols)
    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering" + size + r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\caption{" + caption + r"}")
    out.append(r"\label{" + label + r"}")
    out.append(r"\begin{tabular}{" + spec + r"}")
    out.append(r"\toprule")
    header = ["Method"] + [r"\multicolumn{1}{c}{" + h + r"}" for _, h in cols]
    out.append(" & ".join(header) + r" \\")
    n_synth = len(cols) - n_real_cols
    real_lo, real_hi = n_synth + 2, len(cols) + 1
    out.append(
        rf"\cmidrule(lr){{2-{n_synth + 1}}} \cmidrule(lr){{{real_lo}-{real_hi}}}"
    )
    bests = {c: best_in_col(summary, metric, c, lower_better) for c, _ in cols}

    first_group = True
    for _, rows in GROUPS:
        printable = [(k, n) for k, n in rows if k in summary]
        if not printable:
            continue
        if not first_group:
            out.append(r"\midrule")
        first_group = False
        for mkey, mname in printable:
            row = [mname]
            for ckey, _ in cols:
                v = summary[mkey].get(ckey, {}).get(metric)
                cell = fmt(v, metric)
                if mkey == bests.get(ckey):
                    cell = r"\textbf{" + cell + r"}"
                row.append(cell)
            out.append(" & ".join(row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table*}")
    return "\n".join(out)


def render_seed_gap(main: dict, seeds: dict, a: str = "n2n_l1", b: str = "n2n_l2") -> str:
    """Per-seed PSNR gap between two losses across the whole noise grid.

    ``main`` supplies seed 0 (the Table~I run); ``seeds`` supplies the
    ``*_s1`` / ``*_s2`` replicates. Transposed relative to the other tables --
    one column per noise cell, one row per seed -- so the whole grid fits in
    a few lines.
    """
    import statistics as st

    cols = SYNTH_COLS
    gaps = {}
    for ckey, _ in cols:
        gaps[ckey] = [
            main[a][ckey]["psnr"] - main[b][ckey]["psnr"],
            *(seeds[f"{a}_s{i}"][ckey]["psnr"] - seeds[f"{b}_s{i}"][ckey]["psnr"]
              for i in (1, 2)),
        ]
    out = [r"\begin{table*}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}"]
    out.append(
        r"\caption{\MAE{}$-$\MSE{} PSNR gap (dB) per training seed over the "
        r"full synthetic grid; positive favors \MAE{}. The gap holds under "
        r"all three seeds on 13 of 14 columns and all five noise families; "
        r"the exception is the heaviest speckle level, where one seed "
        r"reverses the sign. Parameters as in Table~\ref{tab:cross-psnr}.}"
    )
    out.append(r"\label{tab:seed-gap}")
    out.append(r"\begin{tabular}{l" + "r" * len(cols) + r"}")
    out.append(r"\toprule")
    out.append(
        " & ".join(["Seed"] + [r"\multicolumn{1}{c}{" + h + r"}" for _, h in cols])
        + r" \\"
    )
    out.append(rf"\cmidrule(lr){{2-{len(cols) + 1}}}")
    for i in range(3):
        out.append(
            " & ".join([str(i)] + [f"{gaps[c][i]:+.2f}" for c, _ in cols]) + r" \\"
        )
    out.append(r"\midrule")
    out.append(
        " & ".join(["mean"] + [f"{st.mean(gaps[c]):+.2f}" for c, _ in cols]) + r" \\"
    )
    out.append(
        " & ".join(["s.d."] + [f"{st.stdev(gaps[c]):.2f}" for c, _ in cols]) + r" \\"
    )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table*}")
    return "\n".join(out)


def render_dual_metric(
    summary: dict,
    cols: list[tuple[str, str]],
    caption: str,
    label: str,
) -> str:
    """One compact table: SSIM and LPIPS side by side on a column subset."""
    blocks = [("ssim", False), ("lpips", True)]
    spec = "l" + "r" * len(cols) + "@{\\hspace{6pt}}" + "r" * len(cols)
    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}")
    out.append(r"\caption{" + caption + r"}")
    out.append(r"\label{" + label + r"}")
    out.append(r"\begin{tabular}{" + spec + r"}")
    out.append(r"\toprule")
    n = len(cols)
    out.append(
        rf" & \multicolumn{{{n}}}{{c}}{{SSIM $\uparrow$}} & "
        rf"\multicolumn{{{n}}}{{c}}{{LPIPS $\downarrow$}} \\"
    )
    out.append(rf"\cmidrule(lr){{2-{n + 1}}} \cmidrule(lr){{{n + 2}-{2 * n + 1}}}")
    header = ["Method"] + [r"\multicolumn{1}{c}{" + h + r"}" for _, h in cols] * 2
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    bests = {
        (metric, c): best_in_col(summary, metric, c, lb)
        for metric, lb in blocks
        for c, _ in cols
    }
    first_group = True
    for _, rows in GROUPS:
        printable = [(k, nm) for k, nm in rows if k in summary]
        if not printable:
            continue
        if not first_group:
            out.append(r"\midrule")
        first_group = False
        for mkey, mname in printable:
            row = [mname]
            for metric, _ in blocks:
                for ckey, _ in cols:
                    v = summary[mkey].get(ckey, {}).get(metric)
                    cell = fmt(v, metric)
                    if mkey == bests.get((metric, ckey)):
                        cell = r"\textbf{" + cell + r"}"
                    row.append(cell)
            out.append(" & ".join(row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table*}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/cross_noise_kodak_v4.json")
    p.add_argument(
        "--blocks-results",
        default=None,
        help="optional eval_sidd_blocks.py JSON; merged in as the SIDD-val column",
    )
    p.add_argument("--out", default="../paper/tables.tex")
    p.add_argument(
        "--seeds-results",
        default=None,
        help=(
            "optional kodak_seeds_*.json (the *_s1/*_s2 replicates). Pass with "
            "--seeds-out to emit the seed-gap table into its own file, so the "
            "main tables file stays byte-comparable across runs."
        ),
    )
    p.add_argument("--seeds-out", default="../paper/table_seeds.tex")
    p.add_argument(
        "--layout",
        choices=["full", "compact"],
        default="full",
        help=(
            "full: three full-grid tables (PSNR/SSIM/LPIPS). "
            "compact: PSNR full grid + one combined SSIM/LPIPS table on a "
            "column subset -- fits the 6-page IEEE budget."
        ),
    )
    args = p.parse_args()

    with open(args.results) as f:
        summary = json.load(f)

    n_real = 1
    if args.blocks_results is not None:
        with open(args.blocks_results) as f:
            blocks = json.load(f)
        for method, by_noise in blocks.items():
            summary.setdefault(method, {}).update(by_noise)
        n_real = 2

    psnr_cols = SYNTH_COLS + [SIDD] + ([SIDD_VAL] if n_real == 2 else [])
    psnr_caption = (
        "Cross-noise generalization, PSNR (dB, higher is better). Bold marks "
        "the best method per column (noisy input excluded). Top block: "
        "non-learning (BM3D) and self-supervised (Noise2Void) baselines. "
        "Middle block: Noise2Noise trained on synthetic Gaussian noise with "
        "five losses. Bottom block: models trained on SIDD-Medium real noise "
        "without ground truth---Noise2Void on single noisy shots, the five "
        "Noise2Noise losses on noisy/noisy pairs. Column parameters differ "
        "by family and are not a common percentage. $\\sigma$: Gaussian "
        "s.d.\\ in 8-bit levels ($25$ is $25/255$). imp, sp: percent of "
        "pixel values corrupted, drawn per channel (uniform draws; 0 or 1 "
        "at 50/50). poi: Poisson rate $\\lambda$, "
        "$y{=}\\mathrm{Pois}(\\lambda x)/\\lambda$, so larger is "
        "\\emph{less} noise. spk: speckle variance $\\times$100 in "
        "$y{=}x{+}xn$, $n\\sim\\mathcal{N}(0,v)$ ($50$ is $v{=}0.5$). "
        "SIDD: 32 held-out "
        "scenes at native resolution; SIDD val: official validation blocks."
    )
    psnr = render_one(
        summary, "psnr", False, psnr_caption, "tab:cross-psnr",
        cols=psnr_cols, n_real_cols=n_real,
        size=r"\scriptsize" if args.layout == "compact" else r"\small",
    )

    if args.layout == "compact":
        real_cols = [SIDD] + ([SIDD_VAL] if n_real == 2 else [])
        dual = render_dual_metric(
            summary,
            COMPACT_COLS + real_cols,
            r"SSIM and LPIPS (AlexNet) on representative columns. "
            r"Conventions as in Table~\ref{tab:cross-psnr}.",
            "tab:cross-perceptual",
        )
        body = psnr + "\n\n" + dual
    else:
        ssim = render_one(
            summary, "ssim", False,
            r"Cross-noise generalization, SSIM (higher is better). Conventions as "
            r"in Table~\ref{tab:cross-psnr}.",
            "tab:cross-ssim",
            cols=psnr_cols, n_real_cols=n_real,
        )
        lpips = render_one(
            summary, "lpips", True,
            r"Cross-noise generalization, LPIPS (AlexNet, lower is better). "
            r"Conventions as in Table~\ref{tab:cross-psnr}.",
            "tab:cross-lpips",
            cols=psnr_cols, n_real_cols=n_real,
        )
        body = psnr + "\n\n" + ssim + "\n\n" + lpips

    if args.seeds_results is not None:
        with open(args.seeds_results) as f:
            seed_data = json.load(f)
        seed_out = Path(args.seeds_out)
        seed_out.parent.mkdir(parents=True, exist_ok=True)
        seed_out.write_text(
            f"% Auto-generated from {args.seeds_results} by render_tables.py.\n"
            "% Re-run to refresh after a new seed pass.\n\n"
            + render_seed_gap(summary, seed_data) + "\n"
        )
        print(f"wrote {seed_out}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"% Auto-generated from {args.results} by render_tables.py"
        f" (layout={args.layout}).\n"
        "% Re-run to refresh after a new evaluation pass.\n\n" + body + "\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
