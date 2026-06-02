#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

GPUS=(0 1)
MU=(1. 2. 4. 8.)
SIGMA=(2. 4.)

job=0
for mu in "${MU[@]}"; do
  for sigma in "${SIGMA[@]}"; do
    gpu="${GPUS[$((job % 2))]}"
    name_log="lognorm_d64_mu_${mu}_sigma_${sigma}"
    echo "[QUEUE] GPU ${gpu} ${name_log}"

    (
      CUDA_VISIBLE_DEVICES="${gpu}" WANDB_NAME="${name_log}" \
        nohup python -m train \
          snrpath.name=lognormal \
          snrpath.mu="${mu}" \
          snrpath.sigma="${sigma}" \
          train.max_steps=1000 \
          train.mix_lambda=0. \
          > "logs/${name_log}.log" 2>&1
    ) &

    job=$((job+1))
  done
done

wait