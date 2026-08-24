#!/usr/bin/env bash
# Evaluation orchestration. Run after train_all.sh.
#
#   bash eval_all.sh gpu     # learned methods: full grid + native SIDD + official blocks
#   bash eval_all.sh bm3d    # BM3D-only pass (CPU, hours) -- run in parallel/overnight
#   bash eval_all.sh tables  # merge + render compact tables into ../paper/tables_v4.tex
#
# Outputs:
#   results/cross_noise_kodak_v4.json        (+ .per_image.json)  learned, synth+SIDD32
#   results/sidd_val_blocks_v4.json          (+ .per_image.json)  learned, official blocks
#   results/cross_noise_kodak_v4_bm3d.json   (+ .per_image.json)  BM3D, synth+SIDD32
#   results/sidd_val_blocks_v4_bm3d.json     (+ .per_image.json)  BM3D, official blocks
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

N2N_CKPTS="l2=ckpts/n2n_l2_v4.pt l2_lasso=ckpts/n2n_l2_lasso_v4.pt l1=ckpts/n2n_l1_v4.pt charb=ckpts/n2n_charb_v4.pt huber=ckpts/n2n_huber_v4.pt l2_real=ckpts/n2n_l2_real_v4.pt l2_lasso_real=ckpts/n2n_l2_lasso_real_v4.pt l1_real=ckpts/n2n_l1_real_v4.pt charb_real=ckpts/n2n_charb_real_v4.pt huber_real=ckpts/n2n_huber_real_v4.pt"

case "${1:-gpu}" in
  gpu)
    $PY evaluate.py \
      --eval-root ../data/Kodak24 --max-side 4096 \
      --sidd-root ../data/SIDD_Medium_Srgb --sidd-max-side 8192 --tile 512 \
      --noise2noise-ckpts $N2N_CKPTS \
      --n2v-ckpt ckpts/n2v_v4.pt \
      --out results/cross_noise_kodak_v4.json
    $PY eval_sidd_blocks.py \
      --blocks-dir ../data/SIDD_Blocks \
      --noise2noise-ckpts $N2N_CKPTS \
      --n2v-ckpt ckpts/n2v_v4.pt \
      --out results/sidd_val_blocks_v4.json
    echo GPU_EVAL_DONE
    ;;
  bm3d)
    # Three separately persisted stages: a single 7h pass that dies late
    # loses everything (results are only written at the end of a pass).
    if [ ! -f results/cross_noise_kodak_v4_bm3d.json ]; then
      $PY evaluate.py \
        --eval-root ../data/Kodak24 --max-side 4096 \
        --bm3d \
        --out results/cross_noise_kodak_v4_bm3d.json
    fi
    echo BM3D_STAGE_KODAK_DONE
    if [ ! -f results/sidd_val_blocks_v4_bm3d.json ]; then
      $PY eval_sidd_blocks.py \
        --blocks-dir ../data/SIDD_Blocks \
        --bm3d \
        --out results/sidd_val_blocks_v4_bm3d.json
    fi
    echo BM3D_STAGE_BLOCKS_DONE
    if [ ! -f results/sidd32_v4_bm3d.json ]; then
      $PY evaluate.py \
        --eval-root ../data/Kodak24 --skip-synth \
        --sidd-root ../data/SIDD_Medium_Srgb --sidd-max-side 8192 \
        --bm3d \
        --out results/sidd32_v4_bm3d.json
    fi
    echo BM3D_EVAL_DONE
    ;;
  tables)
    $PY - << 'EOF'
import json

def merge(dst_path, src_path):
    with open(dst_path) as f:
        dst = json.load(f)
    with open(src_path) as f:
        src = json.load(f)
    for method, by_noise in src.items():
        if method == "noisy":
            continue  # noisy row already present from the learned pass
        dst.setdefault(method, {}).update(by_noise)
    with open(dst_path, "w") as f:
        json.dump(dst, f, indent=2)
    print(f"merged {src_path} -> {dst_path}")

# _cbm3d, not _bm3d: the *_bm3d.json files are the superseded per-channel
# grayscale-BM3D run kept only as evidence for the paper's removed-confounds
# discussion. Merging them here would silently put the ~2 dB-low baseline back
# into the canonical tables.
merge("results/cross_noise_kodak_v4.json", "results/cross_noise_kodak_v4_cbm3d.json")
merge("results/cross_noise_kodak_v4.json", "results/sidd32_v4_cbm3d.json")
merge("results/sidd_val_blocks_v4.json", "results/sidd_val_blocks_v4_cbm3d.json")
EOF
    $PY render_tables.py \
      --results results/cross_noise_kodak_v4.json \
      --blocks-results results/sidd_val_blocks_v4.json \
      --layout compact \
      --out ../paper/tables_v4.tex
    echo TABLES_DONE
    ;;
  *)
    echo "usage: $0 {gpu|bm3d|tables}" >&2; exit 2
    ;;
esac
