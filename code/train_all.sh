#!/usr/bin/env bash
# Batch retraining for the paper.
# ARCH_FLAGS picks the U-Net variant; the paper uses the default.
# All runs: 50 epochs x 5000 virtual crops x batch 16 (matched budgets).
# Idempotent: runs whose checkpoint already exists are skipped, so the
# script can be relaunched after an interruption. Delete a .pt to redo it.
# Detach with setsid so it survives the launching terminal/session:
#   setsid bash train_all.sh &
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

# Either "--residual" (paper setting) or "--residual --no-bn".
ARCH_FLAGS="${ARCH_FLAGS:---residual}"
COMMON="--epochs 50 --batch-size 16 --num-workers 4 --seed 0"

run_if_missing() {  # $1 = ckpt path, rest = command
  local ckpt="$1"; shift
  if [ -f "$ckpt" ]; then
    echo "skip $ckpt (exists)"
    return 0
  fi
  "$@"
  echo "done $ckpt"
}

echo "=== Synthetic arm (ARCH_FLAGS=$ARCH_FLAGS) ==="
for loss in l1 charbonnier huber; do
  name=$([ "$loss" = charbonnier ] && echo charb || echo "$loss")
  run_if_missing "ckpts/n2n_${name}_v4.pt" \
    bash -c "$PY train.py --data-root ../data/CBSD400 --out ckpts/n2n_${name}_v4.pt \
      --loss $loss --virtual-length 5000 $COMMON $ARCH_FLAGS \
      > logs/n2n_${name}_v4.log 2>&1"
done
run_if_missing ckpts/n2n_l2_lasso_v4.pt \
  bash -c "$PY train.py --data-root ../data/CBSD400 --out ckpts/n2n_l2_lasso_v4.pt \
    --loss l2 --lasso 1e-5 --virtual-length 5000 $COMMON $ARCH_FLAGS \
    > logs/n2n_l2_lasso_v4.log 2>&1"

echo "=== Real-pair arm, native resolution ==="
for loss in l2 l1 charbonnier huber; do
  name=$([ "$loss" = charbonnier ] && echo charb || echo "$loss")
  run_if_missing "ckpts/n2n_${name}_real_v4.pt" \
    bash -c "$PY train.py --real-pairs ../data/SIDD_Medium_Srgb \
      --sidd-npy-root ../data/SIDD_Medium_Srgb_npy \
      --out ckpts/n2n_${name}_real_v4.pt --loss $loss $COMMON $ARCH_FLAGS \
      > logs/n2n_${name}_real_v4.log 2>&1"
done
run_if_missing ckpts/n2n_l2_lasso_real_v4.pt \
  bash -c "$PY train.py --real-pairs ../data/SIDD_Medium_Srgb \
    --sidd-npy-root ../data/SIDD_Medium_Srgb_npy \
    --out ckpts/n2n_l2_lasso_real_v4.pt --loss l2 --lasso 1e-5 $COMMON $ARCH_FLAGS \
    > logs/n2n_l2_lasso_real_v4.log 2>&1"

echo "=== Noise2Void, matched budget ==="
run_if_missing ckpts/n2v_v4.pt \
  bash -c "$PY train_n2v.py --data-root ../data/CBSD400 --out ckpts/n2v_v4.pt \
    --epochs 50 --length 5000 --batch-size 16 --num-workers 4 \
    > logs/n2v_v4.log 2>&1"

echo "ALL_BATCH_DONE"
