#!/usr/bin/env bash
# Reproduction batch: N2C supervised anchor + real-cell seeds.
# Idempotent: training skips if ckpt exists, eval skips if results json exists.
# Detach with:  setsid bash n2c_and_real_seeds.sh &
# Sequential on purpose -- 10 GB VRAM / 15 GB RAM budget, no parallel stages.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

REAL="--real-pairs ../data/SIDD_Medium_Srgb --sidd-npy-root ../data/SIDD_Medium_Srgb_npy"
COMMON="--epochs 50 --batch-size 16 --num-workers 4 --residual"

train_if_missing() {  # $1 ckpt, $2 log, rest = train args
  local ckpt="$1" log="$2"; shift 2
  if [ -f "$ckpt" ]; then echo "skip $ckpt"; return 0; fi
  $PY train.py --out "$ckpt" "$@" > "$log" 2>&1
  echo "done $ckpt"
}

echo "=== PREP: GT npy for the 128 training scenes (idempotent) ==="
$PY prep_sidd_npy.py --sidd-root ../data/SIDD_Medium_Srgb \
  --npy-root ../data/SIDD_Medium_Srgb_npy \
  --noisy-glob "*GT_SRGB_*.PNG" --split train > logs/r5_prep_gt_npy.log 2>&1
echo "PREP_GT_DONE"

echo "=== N2C: supervised anchor (reads GT, disclosed control; MSE) ==="
train_if_missing ckpts/n2c_l2_v4.pt logs/n2c_l2_v4.log \
  $REAL $COMMON --loss l2 --supervised-gt --seed 0

echo "=== SEEDS: real cell, five losses x seeds 1,2 ==="
for seed in 1 2; do
  train_if_missing "ckpts/n2n_l2_real_v4_s${seed}.pt" "logs/n2n_l2_real_v4_s${seed}.log" \
    $REAL $COMMON --loss l2 --seed $seed
  train_if_missing "ckpts/n2n_l1_real_v4_s${seed}.pt" "logs/n2n_l1_real_v4_s${seed}.log" \
    $REAL $COMMON --loss l1 --seed $seed
  train_if_missing "ckpts/n2n_charb_real_v4_s${seed}.pt" "logs/n2n_charb_real_v4_s${seed}.log" \
    $REAL $COMMON --loss charbonnier --seed $seed
  train_if_missing "ckpts/n2n_huber_d005_real_v4_s${seed}.pt" "logs/n2n_huber_d005_real_v4_s${seed}.log" \
    $REAL $COMMON --loss huber --huber-delta 0.05 --seed $seed
  train_if_missing "ckpts/n2n_l2_lasso_real_v4_s${seed}.pt" "logs/n2n_l2_lasso_real_v4_s${seed}.log" \
    $REAL $COMMON --loss l2 --lasso 1e-5 --seed $seed
done

echo "=== EVAL: N2C anchor (official blocks + Kodak grid + native SIDD32) ==="
if [ ! -f results/r5_n2c_blocks_v4.json ]; then
  $PY eval_sidd_blocks.py \
    --blocks-dir ../data/SIDD_Blocks \
    --noise2noise-ckpts n2c_l2=ckpts/n2c_l2_v4.pt \
    --out results/r5_n2c_blocks_v4.json > logs/r5_eval_n2c_blocks.log 2>&1
fi
echo "EVAL_N2C_BLOCKS_DONE"
if [ ! -f results/r5_n2c_kodak_v4.json ]; then
  $PY evaluate.py \
    --eval-root ../data/Kodak24 --max-side 4096 \
    --sidd-root ../data/SIDD_Medium_Srgb --sidd-max-side 8192 --tile 512 \
    --noise2noise-ckpts n2c_l2=ckpts/n2c_l2_v4.pt \
    --out results/r5_n2c_kodak_v4.json > logs/r5_eval_n2c_kodak.log 2>&1
fi
echo "EVAL_N2C_KODAK_DONE"

echo "=== EVAL: real-cell seed replicates (official blocks only) ==="
if [ ! -f results/r5_real_seeds_blocks_v4.json ]; then
  $PY eval_sidd_blocks.py \
    --blocks-dir ../data/SIDD_Blocks \
    --noise2noise-ckpts \
      l2_real_s1=ckpts/n2n_l2_real_v4_s1.pt l2_real_s2=ckpts/n2n_l2_real_v4_s2.pt \
      l1_real_s1=ckpts/n2n_l1_real_v4_s1.pt l1_real_s2=ckpts/n2n_l1_real_v4_s2.pt \
      charb_real_s1=ckpts/n2n_charb_real_v4_s1.pt charb_real_s2=ckpts/n2n_charb_real_v4_s2.pt \
      huber_d005_real_s1=ckpts/n2n_huber_d005_real_v4_s1.pt huber_d005_real_s2=ckpts/n2n_huber_d005_real_v4_s2.pt \
      l2_lasso_real_s1=ckpts/n2n_l2_lasso_real_v4_s1.pt l2_lasso_real_s2=ckpts/n2n_l2_lasso_real_v4_s2.pt \
    --out results/r5_real_seeds_blocks_v4.json > logs/r5_eval_seeds_blocks.log 2>&1
fi
echo "EVAL_SEEDS_BLOCKS_DONE"

echo "R5_BATCH_ALL_DONE"
