"""Merge follow-up evaluation outputs into the canonical v4 results files.

Adds the follow-up models to results/cross_noise_kodak_v4.json and
results/sidd_val_blocks_v4.json (and their .per_image.json siblings):

* n2n_huber_d005 / n2n_huber_d005_real  (table Huber rows now delta=0.05)
* n2n_l2_lasso_a1e6 / n2n_l2_lasso_a1e4 (alpha sweep, text only)
* n2v -> renamed n2v_real               (N2V trained on real shots)

Old n2n_huber(_real) entries stay in the JSON; render_tables.py simply no
longer selects them. Idempotent: re-running overwrites the same keys.
"""

from __future__ import annotations

import json
from pathlib import Path

RENAME = {"n2v": "n2v_real"}
SKIP = {"noisy"}  # canonical files already carry the noisy row


def merge(dst_path: str, src_path: str) -> None:
    dst_file, src_file = Path(dst_path), Path(src_path)
    with src_file.open() as f:
        src = json.load(f)
    with dst_file.open() as f:
        dst = json.load(f)
    for method, payload in src.items():
        if method in SKIP:
            continue
        name = RENAME.get(method, method)
        if isinstance(payload, dict):
            dst.setdefault(name, {}).update(payload)
        else:  # per-image blocks schema: method -> list
            dst[name] = payload
    with dst_file.open("w") as f:
        json.dump(dst, f, indent=2)
    print(f"merged {src_file.name} -> {dst_file.name}: +{list(src.keys())}")


if __name__ == "__main__":
    merge("results/cross_noise_kodak_v4.json", "results/r2_new_models_kodak_v4.json")
    merge(
        "results/cross_noise_kodak_v4.per_image.json",
        "results/r2_new_models_kodak_v4.per_image.json",
    )
    merge("results/sidd_val_blocks_v4.json", "results/r2_new_models_blocks_v4.json")
    merge(
        "results/sidd_val_blocks_v4.per_image.json",
        "results/r2_new_models_blocks_v4.per_image.json",
    )
