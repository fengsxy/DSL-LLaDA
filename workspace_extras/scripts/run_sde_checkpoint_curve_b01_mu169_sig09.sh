#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_ID="${SWEEP_ID:-sde_ckpt_curve_b01_mu169_sig09_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/sde_param_search}"
LIMIT="${LIMIT:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

mkdir -p "$LOG_DIR"
echo "[curve] id=$SWEEP_ID limit=$LIMIT max_parallel=$MAX_PARALLEL gpus=($GPUS)"

run_one() {
  local gpu="$1"
  local model="$2"
  local label="$3"
  local beta="$4"
  local ns="$5"
  local gen_len="$6"

  local out_tag="${SWEEP_ID}_${label}"
  local log_file="${LOG_DIR}/${out_tag}.log"
  echo "[curve] gpu=$gpu model=$model label=$label beta=$beta ns=$ns gen=$gen_len"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "$model" \
    --gpu 0 \
    --nfe 32 \
    --gen_length "$gen_len" \
    --limit "$LIMIT" \
    --sde_beta_infer "$beta" \
    --sde_noise_scale "$ns" \
    --sde_schedule sensitive \
    --sde_snr_min 0.01 \
    --sde_snr_max 100 \
    --sde_sensitive_low 7 \
    --sde_sensitive_high 74 \
    --sde_top_k 512 \
    --out_tag "$out_tag" \
    >"$log_file" 2>&1
}

CONFIGS=(
  # model label beta noise gen_len
  "uu_b01_mu169_sig09_1k 1k_b1_ns0_g128 1.0 0.0 128"
  "uu_b01_mu169_sig09_1k 1k_b2_ns0_g128 2.0 0.0 128"
  "uu_b01_mu169_sig09_1k 1k_b2_ns0_g64 2.0 0.0 64"
  "uu_b01_mu169_sig09_1k 1k_b4_ns0_g128 4.0 0.0 128"

  "uu_b01_mu169_sig09_3k 3k_b1_ns0_g128 1.0 0.0 128"
  "uu_b01_mu169_sig09_3k 3k_b2_ns0_g128 2.0 0.0 128"
  "uu_b01_mu169_sig09_3k 3k_b2_ns0_g64 2.0 0.0 64"
  "uu_b01_mu169_sig09_3k 3k_b4_ns0_g128 4.0 0.0 128"

  "uu_b01_mu169_sig09_5k 5k_b1_ns0_g128 1.0 0.0 128"
  "uu_b01_mu169_sig09_5k 5k_b2_ns0_g128 2.0 0.0 128"
  "uu_b01_mu169_sig09_5k 5k_b2_ns0_g64 2.0 0.0 64"
  "uu_b01_mu169_sig09_5k 5k_b4_ns0_g128 4.0 0.0 128"

  "uu_b01_mu169_sig09_10k 10k_b1_ns0_g128 1.0 0.0 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0_g128 2.0 0.0 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0_g64 2.0 0.0 64"
  "uu_b01_mu169_sig09_10k 10k_b4_ns0_g128 4.0 0.0 128"
)

gpu_arr=($GPUS)
gpu_i=0
for cfg in "${CONFIGS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 5
  done
  read -r model label beta ns gen_len <<<"$cfg"
  gpu="${gpu_arr[$((gpu_i % ${#gpu_arr[@]}))]}"
  gpu_i=$((gpu_i + 1))
  run_one "$gpu" "$model" "$label" "$beta" "$ns" "$gen_len" &
done

wait
echo "[curve] complete id=$SWEEP_ID"
