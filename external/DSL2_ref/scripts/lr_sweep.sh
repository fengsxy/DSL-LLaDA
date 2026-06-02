#!/usr/bin/env bash
set -u
mkdir -p logs   # ← ensure log dir exists

GPUS=(1 2 3 4 5 6)
LRS=(5e-5 1e-4 3e-4 6e-4 1e-3 2e-3 4e-3)

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  lr=${LRS[$i]}
  echo "[LAUNCH] GPU $gpu  LR $lr"
  CUDA_VISIBLE_DEVICES=$gpu WANDB_NAME="lr_${lr}" \
  nohup python -m train optim.lr=$lr > "logs/lr_${lr}.log" 2>&1 &
done
wait
echo "All sweep jobs finished."