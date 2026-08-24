#!/usr/bin/env bash
# Reproduction batch.
# Idempotent: training runs skip if the ckpt exists, eval stages skip if the
# results json exists. Detach with:  setsid bash diagnostics_batch.sh &
# Sequential on purpose -- 10 GB VRAM / 15 GB RAM budget, no parallel stages.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

SYNTH="--data-root ../data/CBSD400 --virtual-length 5000"
REAL="--real-pairs ../data/SIDD_Medium_Srgb --sidd-npy-root ../data/SIDD_Medium_Srgb_npy"
COMMON="--epochs 50 --batch-size 16 --num-workers 4 --residual"

train_if_missing() {  # $1 ckpt, $2 log, rest = train args
  local ckpt="$1" log="$2"; shift 2
  if [ -f "$ckpt" ]; then echo "skip $ckpt"; return 0; fi
  $PY train.py --out "$ckpt" "$@" > "$log" 2>&1
  echo "done $ckpt"
}

echo "=== Huber delta=0.05, synth + real ==="
train_if_missing ckpts/n2n_huber_d005_v4.pt logs/n2n_huber_d005_v4.log \
  $SYNTH $COMMON --loss huber --huber-delta 0.05 --seed 0
train_if_missing ckpts/n2n_huber_d005_real_v4.pt logs/n2n_huber_d005_real_v4.log \
  $REAL $COMMON --loss huber --huber-delta 0.05 --seed 0

echo "=== Grad-norm logging runs (10 epochs, diagnostic only) ==="
for loss in l2 l1; do
  train_if_missing "ckpts/n2n_${loss}_gradlog_v4.pt" "logs/n2n_${loss}_gradlog_v4.log" \
    $SYNTH --epochs 10 --batch-size 16 --num-workers 4 --residual \
    --loss $loss --seed 0 --log-grad-norms
done

echo "=== Target-outlier contamination (5% salt/pepper on x2) ==="
for loss in l2 l1; do
  train_if_missing "ckpts/n2n_${loss}_out5_v4.pt" "logs/n2n_${loss}_out5_v4.log" \
    $SYNTH $COMMON --loss $loss --seed 0 --target-outlier-frac 0.05
done

echo "=== N2V trained on real SIDD noisy shots ==="
if [ -f ckpts/n2v_real_v4.pt ]; then echo "skip ckpts/n2v_real_v4.pt"; else
  $PY train_n2v.py --real-noisy ../data/SIDD_Medium_Srgb \
    --sidd-npy-root ../data/SIDD_Medium_Srgb_npy \
    --out ckpts/n2v_real_v4.pt --epochs 50 --length 5000 --batch-size 16 \
    --num-workers 4 > logs/n2v_real_v4.log 2>&1
  echo "done ckpts/n2v_real_v4.pt"
fi

echo "=== Lasso alpha sweep ==="
train_if_missing ckpts/n2n_l2_lasso_a1e6_v4.pt logs/n2n_l2_lasso_a1e6_v4.log \
  $SYNTH $COMMON --loss l2 --lasso 1e-6 --seed 0
train_if_missing ckpts/n2n_l2_lasso_a1e4_v4.pt logs/n2n_l2_lasso_a1e4_v4.log \
  $SYNTH $COMMON --loss l2 --lasso 1e-4 --seed 0

echo "=== Grid seeds 1,2 for charb / huber(d005) / lasso ==="
for seed in 1 2; do
  train_if_missing "ckpts/n2n_charb_v4_s${seed}.pt" "logs/n2n_charb_v4_s${seed}.log" \
    $SYNTH $COMMON --loss charbonnier --seed $seed
  train_if_missing "ckpts/n2n_huber_d005_v4_s${seed}.pt" "logs/n2n_huber_d005_v4_s${seed}.log" \
    $SYNTH $COMMON --loss huber --huber-delta 0.05 --seed $seed
  train_if_missing "ckpts/n2n_l2_lasso_v4_s${seed}.pt" "logs/n2n_l2_lasso_v4_s${seed}.log" \
    $SYNTH $COMMON --loss l2 --lasso 1e-5 --seed $seed
done

echo "=== EVAL: new main-table models (Kodak grid + native SIDD32) ==="
NEW_CKPTS="huber_d005=ckpts/n2n_huber_d005_v4.pt huber_d005_real=ckpts/n2n_huber_d005_real_v4.pt l2_lasso_a1e6=ckpts/n2n_l2_lasso_a1e6_v4.pt l2_lasso_a1e4=ckpts/n2n_l2_lasso_a1e4_v4.pt"
if [ ! -f results/r2_new_models_kodak_v4.json ]; then
  $PY evaluate.py \
    --eval-root ../data/Kodak24 --max-side 4096 \
    --sidd-root ../data/SIDD_Medium_Srgb --sidd-max-side 8192 --tile 512 \
    --noise2noise-ckpts $NEW_CKPTS \
    --n2v-ckpt ckpts/n2v_real_v4.pt \
    --out results/r2_new_models_kodak_v4.json > logs/r2_eval_kodak.log 2>&1
fi
echo "EVAL_KODAK_DONE"

if [ ! -f results/r2_new_models_blocks_v4.json ]; then
  $PY eval_sidd_blocks.py \
    --blocks-dir ../data/SIDD_Blocks \
    --noise2noise-ckpts $NEW_CKPTS \
    --n2v-ckpt ckpts/n2v_real_v4.pt \
    --out results/r2_new_models_blocks_v4.json > logs/r2_eval_blocks.log 2>&1
fi
echo "EVAL_BLOCKS_DONE"

echo "=== EVAL: outlier-contamination pair (Kodak grid only) ==="
if [ ! -f results/r2_outlier_kodak_v4.json ]; then
  $PY evaluate.py \
    --eval-root ../data/Kodak24 --max-side 4096 \
    --noise2noise-ckpts l2_out5=ckpts/n2n_l2_out5_v4.pt l1_out5=ckpts/n2n_l1_out5_v4.pt \
    --out results/r2_outlier_kodak_v4.json > logs/r2_eval_outlier.log 2>&1
fi
echo "EVAL_OUTLIER_DONE"

echo "=== EVAL: new seed replicates (Kodak grid only) ==="
if [ ! -f results/kodak_seeds_r2.json ]; then
  $PY evaluate.py \
    --eval-root ../data/Kodak24 --max-side 4096 \
    --noise2noise-ckpts \
      charb_s1=ckpts/n2n_charb_v4_s1.pt charb_s2=ckpts/n2n_charb_v4_s2.pt \
      huber_d005_s1=ckpts/n2n_huber_d005_v4_s1.pt huber_d005_s2=ckpts/n2n_huber_d005_v4_s2.pt \
      l2_lasso_s1=ckpts/n2n_l2_lasso_v4_s1.pt l2_lasso_s2=ckpts/n2n_l2_lasso_v4_s2.pt \
    --out results/kodak_seeds_r2.json > logs/r2_eval_seeds.log 2>&1
fi
echo "EVAL_SEEDS_DONE"

echo "R2_BATCH_ALL_DONE"
