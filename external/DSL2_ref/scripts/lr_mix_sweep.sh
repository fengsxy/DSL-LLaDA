#!/usr/bin/env bash
set -u
mkdir -p logs   # ensure log dir exists

# Baseline sweep (mix_lambda = 0.0) on GPUs 0-3
BASE_GPUS=(0 1 2 3)
# Mixed-loss sweep (mix_lambda = 0.5) on GPUs 4-7
MIX_GPUS=(4 5 6 7)

# Shared learning rates
LRS=(5e-4 1e-3 2e-3 4e-3)

# --- Launch baseline (mix_lambda=0.0) ---
for i in "${!BASE_GPUS[@]}"; do
  gpu=${BASE_GPUS[$i]}
  lr=${LRS[$i]}
  echo "[LAUNCH] BASE GPU $gpu  LR $lr  (mix_lambda=0.0)"
  CUDA_VISIBLE_DEVICES=$gpu WANDB_NAME="base_lr_${lr}" \
  nohup python -m train optim.lr=$lr backbone=dit-small train.mix_lambda=0.0 > "logs/base_lr_${lr}.log" 2>&1 &
done

# --- Launch mixed (mix_lambda=0.5) ---
for i in "${!MIX_GPUS[@]}"; do
  gpu=${MIX_GPUS[$i]}
  lr=${LRS[$i]}
  echo "[LAUNCH] MIX  GPU $gpu  LR $lr  (mix_lambda=0.5)"
  CUDA_VISIBLE_DEVICES=$gpu WANDB_NAME="mix_lr_${lr}" \
  nohup python -m train optim.lr=$lr backbone=dit-small train.mix_lambda=0.5 > "logs/mix_lr_${lr}.log" 2>&1 &
done

wait
echo "All sweep jobs finished."