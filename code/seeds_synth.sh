#!/usr/bin/env bash
# Seed replicates for the key MSE-vs-MAE comparison. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
for seed in 1 2; do
  for loss in l2 l1; do
    ckpt="ckpts/n2n_${loss}_v4_s${seed}.pt"
    [ -f "$ckpt" ] && { echo "skip $ckpt"; continue; }
    $PY train.py --data-root ../data/CBSD400 --out "$ckpt" \
      --loss $loss --virtual-length 5000 --epochs 50 --batch-size 16 \
      --num-workers 4 --seed $seed --residual > "logs/n2n_${loss}_v4_s${seed}.log" 2>&1
    echo "done $ckpt"
  done
done
echo SEEDS_DONE
