"""Plot the convolutional weight histogram for several trained models.

The point of this script is to falsify the prior draft's claim that MAE/L1
*loss* induces weight sparsity. If the claim were correct, the L1-loss model's
histogram would show a spike at zero relative to L2; in practice it does not.
The Lasso-MSE model (whose objective contains an explicit L1 weight penalty)
is the only one that should show such a spike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def collect_conv_weights(state_dict: dict) -> torch.Tensor:
    parts = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.dim() == 4:
            parts.append(v.flatten().detach().cpu())
    return torch.cat(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="name=path entries")
    p.add_argument("--out", required=True)
    p.add_argument("--bins", type=int, default=200)
    p.add_argument("--xlim", type=float, default=0.5)
    args = p.parse_args()

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(6, 3.8))
    # Distinct linestyles: the whole point of the figure is that the MSE and
    # MAE curves coincide -- with two solid lines the one drawn first is
    # invisible under the other, so the coincidence cannot be *seen*.
    styles = ["--", "-", "-."]
    widths = [2.2, 1.4, 1.4]
    for i, spec in enumerate(args.ckpts):
        if "=" not in spec:
            raise ValueError(f"expected name=path, got '{spec}'")
        name, path = spec.split("=", 1)
        state = torch.load(path, map_location="cpu")
        w = collect_conv_weights(state["model"])
        ax.hist(
            w.numpy(),
            bins=args.bins,
            range=(-args.xlim, args.xlim),
            histtype="step",
            label=f"{name} ({w.numel()/1e6:.1f}M weights)",
            density=True,
            linestyle=styles[i % len(styles)],
            linewidth=widths[i % len(widths)],
        )
    ax.set_xlabel("conv weight value")
    ax.set_ylabel("density")
    ax.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
