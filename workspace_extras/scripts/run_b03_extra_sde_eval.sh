#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
PAIR_ID=${PAIR_ID:-beta_snr_matched_20260512_225231}
RUN_ID=${RUN_ID:-unit_uniform_b03_mu369_sig06_snrln100_1k_${PAIR_ID}}
MODEL_KEY=${MODEL_KEY:-probe_${RUN_ID}}
LIMIT=${LIMIT:-8}
GPUS=${GPUS:-0 1 2 3 4}
EVAL_LOG_DIR=${EVAL_LOG_DIR:-${REPO_ROOT}/logs/sde_param_search}

cd "${REPO_ROOT}"
mkdir -p "${EVAL_LOG_DIR}"

run_one() {
  local gpu="$1"
  local label="$2"
  local beta_infer="$3"
  local snr_min="$4"
  local snr_max="$5"
  local gen_len="$6"

  local out_tag="${PAIR_ID}_${RUN_ID}_${label}"
  local log_file="${EVAL_LOG_DIR}/${out_tag}.log"
  echo "[b03-extra] gpu=${gpu} label=${label} beta=${beta_infer} snr=[${snr_min},${snr_max}] gen=${gen_len}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "${MODEL_KEY}" \
    --gpu 0 \
    --nfe 32 \
    --gen_length "${gen_len}" \
    --limit "${LIMIT}" \
    --sde_beta_infer "${beta_infer}" \
    --sde_noise_scale 0.0 \
    --sde_schedule sensitive \
    --sde_snr_min "${snr_min}" \
    --sde_snr_max "${snr_max}" \
    --sde_sensitive_low 7 \
    --sde_sensitive_high 74 \
    --sde_top_k 512 \
    --out_tag "${out_tag}" \
    >"${log_file}" 2>&1
}

configs=(
  "bi3_snr1_150_g64 3.0 1 150 64"
  "bi3_snr1_150_g128 3.0 1 150 128"
  "bi4_snr1_150_g64 4.0 1 150 64"
  "bi1_snr1_100_g128 1.0 1 100 128"
  "bi2_snr1_100_g64 2.0 1 100 64"
)

gpu_arr=(${GPUS})
gpu_i=0
for cfg in "${configs[@]}"; do
  read -r label beta_infer snr_min snr_max gen_len <<<"${cfg}"
  gpu="${gpu_arr[$((gpu_i % ${#gpu_arr[@]}))]}"
  gpu_i=$((gpu_i + 1))
  run_one "${gpu}" "${label}" "${beta_infer}" "${snr_min}" "${snr_max}" "${gen_len}" &
done

wait
echo "[b03-extra] complete"
