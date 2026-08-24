#!/usr/bin/env bash
# Re-run the BM3D baseline as colour CBM3D
# (bm3d.bm3d_rgb) with the corrected wavelet-MAD sigma estimator.
# Old per-channel results (*_bm3d.json) are kept untouched; new outputs use
# the _cbm3d suffix. Merge into the canonical v4 jsons only after the Kodak
# Gaussian columns validate against published CBM3D references
# (~34.3/31.7/27.7 dB at sigma 15/25/50).
#
# Three separately persisted stages (a single long pass that dies late loses
# everything). CPU-only; do NOT run concurrently with other heavy jobs (15GB RAM).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ ! -f results/cross_noise_kodak_v4_cbm3d.json ]; then
  $PY evaluate.py \
    --eval-root ../data/Kodak24 --max-side 4096 \
    --bm3d \
    --out results/cross_noise_kodak_v4_cbm3d.json
fi
echo CBM3D_STAGE_KODAK_DONE
if [ ! -f results/sidd_val_blocks_v4_cbm3d.json ]; then
  $PY eval_sidd_blocks.py \
    --blocks-dir ../data/SIDD_Blocks \
    --bm3d \
    --out results/sidd_val_blocks_v4_cbm3d.json
fi
echo CBM3D_STAGE_BLOCKS_DONE
if [ ! -f results/sidd32_v4_cbm3d.json ]; then
  $PY evaluate.py \
    --eval-root ../data/Kodak24 --skip-synth \
    --sidd-root ../data/SIDD_Medium_Srgb --sidd-max-side 8192 \
    --bm3d \
    --out results/sidd32_v4_cbm3d.json
fi
echo CBM3D_EVAL_ALL_DONE
