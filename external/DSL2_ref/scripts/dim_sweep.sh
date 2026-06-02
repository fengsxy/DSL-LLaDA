#!/usr/bin/env bash
# This is probably pointless... dim noise has to be changed with schedule.
set -u
mkdir -p logs   # ← ensure log dir exists

GPUS=(5 6 7)
DS=(8 16 32)

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  lr=${DS[$i]}
  echo "[LAUNCH] GPU $gpu  DIM $lr"
  CUDA_VISIBLE_DEVICES=$gpu WANDB_NAME="d_${lr}" \
  nohup python -m train data.dim_embed=$lr > "logs/ds_${lr}.log" 2>&1 &
done
wait
echo "All sweep jobs finished."