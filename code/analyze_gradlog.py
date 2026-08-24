"""Per-step gradient-norm statistics, MSE vs MAE.

Reads the grad_log arrays written by ``train.py --log-grad-norms`` and
prints tail statistics of the per-step global gradient norm. The
bounded-influence prediction: the MSE gradient grows linearly in the
residual, so rare large residuals produce a heavy upper tail; the MAE
gradient is sign(r), so its norm is nearly constant across steps.

Usage::

    python analyze_gradlog.py ckpts/n2n_l2_gradlog_v4.json ckpts/n2n_l1_gradlog_v4.json
"""

from __future__ import annotations

import json
import sys

import numpy as np


def stats(path: str) -> None:
    with open(path) as f:
        d = json.load(f)
    g = np.array([r["grad_norm"] for r in d["grad_log"]], dtype=np.float64)
    r = np.array([r["max_abs_residual"] for r in d["grad_log"]], dtype=np.float64)
    # Skip the first 100 steps: init transient dominates both losses equally.
    g, r = g[100:], r[100:]
    q = lambda a, p: float(np.percentile(a, p))
    print(f"\n{path}  (n={g.size} steps, loss={d['args']['loss']})")
    print(
        f"  grad_norm: median={q(g,50):.4f}  P90={q(g,90):.4f}  P99={q(g,99):.4f}"
        f"  P99.9={q(g,99.9):.4f}  max={g.max():.4f}"
    )
    print(
        f"  tail ratios: P99/median={q(g,99)/q(g,50):.2f}"
        f"  max/median={g.max()/q(g,50):.2f}  CV={g.std()/g.mean():.3f}"
    )
    print(f"  max|residual| per batch: median={q(r,50):.3f}  P99={q(r,99):.3f}  max={r.max():.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
