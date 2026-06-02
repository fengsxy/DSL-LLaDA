#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_ID="${SWEEP_ID:-sde_infer_sweep_10k_b1_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/sde_param_search}"
LIMIT="${LIMIT:-25}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

mkdir -p "$LOG_DIR"
echo "[sweep] id=$SWEEP_ID limit=$LIMIT max_parallel=$MAX_PARALLEL gpus=($GPUS)"

run_one() {
  local gpu="$1"
  local model="$2"
  local label="$3"
  local beta="$4"
  local ns="$5"
  local sched="$6"
  local snr_min="$7"
  local snr_max="$8"
  local s_low="$9"
  local s_high="${10}"
  local topk="${11}"
  local nfe="${12}"
  local gen_len="${13}"

  local out_tag="${SWEEP_ID}_${label}"
  local log_file="${LOG_DIR}/${out_tag}.log"
  echo "[sweep] gpu=$gpu model=$model label=$label beta=$beta ns=$ns sched=$sched snr=[$snr_min,$snr_max] sens=[$s_low,$s_high] topk=$topk nfe=$nfe gen=$gen_len"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "$model" \
    --gpu 0 \
    --nfe "$nfe" \
    --gen_length "$gen_len" \
    --limit "$LIMIT" \
    --sde_beta_infer "$beta" \
    --sde_noise_scale "$ns" \
    --sde_schedule "$sched" \
    --sde_snr_min "$snr_min" \
    --sde_snr_max "$snr_max" \
    --sde_sensitive_low "$s_low" \
    --sde_sensitive_high "$s_high" \
    --sde_top_k "$topk" \
    --out_tag "$out_tag" \
    >"$log_file" 2>&1
}

CONFIGS=(
  # model label beta noise schedule snr_min snr_max sens_low sens_high topk nfe gen_len
  "uu_b01_mu169_sig09_10k 10k_b0p36_ns0_sens 0.36 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b1_ns0_sens 1.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b1_ns0p005_sens 1.0 0.005 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0_sens 2.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0p005_sens 2.0 0.005 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b4_ns0_sens 4.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b4_ns0p005_sens 4.0 0.005 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b8_ns0_sens 8.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0_uniform 2.0 0.0 uniform 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b4_ns0_uniform 4.0 0.0 uniform 0.01 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0_snr1 2.0 0.0 sensitive 1.0 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b4_ns0_snr1 4.0 0.0 sensitive 1.0 100 7 74 512 32 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0_top128 2.0 0.0 sensitive 0.01 100 7 74 128 32 128"
  "uu_b01_mu169_sig09_10k 10k_b4_ns0_top128 4.0 0.0 sensitive 0.01 100 7 74 128 32 128"
  "uu_b01_mu169_sig09_10k 10k_b2_ns0_gen64 2.0 0.0 sensitive 0.01 100 7 74 512 32 64"
  "uu_b01_mu169_sig09_10k 10k_b4_ns0_gen64 4.0 0.0 sensitive 0.01 100 7 74 512 32 64"

  "uu_trainable_1k b1_b1_ns0_sens 1.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b2_ns0_sens 2.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b2_ns0p005_sens 2.0 0.005 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b3_ns0_sens 3.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b3_ns0p005_sens 3.0 0.005 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b4_ns0_sens 4.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b4_ns0p005_sens 4.0 0.005 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b6_ns0_sens 6.0 0.0 sensitive 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b2_ns0_uniform 2.0 0.0 uniform 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b3_ns0_uniform 3.0 0.0 uniform 0.01 100 7 74 512 32 128"
  "uu_trainable_1k b1_b2_ns0_snr1 2.0 0.0 sensitive 1.0 100 7 74 512 32 128"
  "uu_trainable_1k b1_b3_ns0_snr1 3.0 0.0 sensitive 1.0 100 7 74 512 32 128"
  "uu_trainable_1k b1_b2_ns0_top128 2.0 0.0 sensitive 0.01 100 7 74 128 32 128"
  "uu_trainable_1k b1_b3_ns0_top128 3.0 0.0 sensitive 0.01 100 7 74 128 32 128"
  "uu_trainable_1k b1_b2_ns0_gen64 2.0 0.0 sensitive 0.01 100 7 74 512 32 64"
  "uu_trainable_1k b1_b3_ns0_gen64 3.0 0.0 sensitive 0.01 100 7 74 512 32 64"
)

gpu_arr=($GPUS)
gpu_i=0
for cfg in "${CONFIGS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 5
  done
  read -r model label beta ns sched snr_min snr_max s_low s_high topk nfe gen_len <<<"$cfg"
  gpu="${gpu_arr[$((gpu_i % ${#gpu_arr[@]}))]}"
  gpu_i=$((gpu_i + 1))
  run_one "$gpu" "$model" "$label" "$beta" "$ns" "$sched" "$snr_min" "$snr_max" "$s_low" "$s_high" "$topk" "$nfe" "$gen_len" &
done

wait
echo "[sweep] complete id=$SWEEP_ID"
