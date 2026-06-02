#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
cd "${REPO_ROOT}"

PAIR_ID=${PAIR_ID:-beta_snr_matched_20260512_225231}
LIMIT=${LIMIT:-8}
MAX_PARALLEL=${MAX_PARALLEL:-8}
GPUS=${GPUS:-0 1 2 3 4 5 6 7}
EVAL_LOG_DIR=${EVAL_LOG_DIR:-${REPO_ROOT}/logs/sde_param_search}
mkdir -p "${EVAL_LOG_DIR}"

run_eval_one() {
  local gpu="$1"
  local model_key="$2"
  local label="$3"
  local beta_infer="$4"
  local snr_min="$5"
  local snr_max="$6"
  local gen_len="$7"

  local out_tag="${PAIR_ID}_${label}"
  local log_file="${EVAL_LOG_DIR}/${out_tag}.log"
  echo "[resume:${PAIR_ID}] eval gpu=${gpu} key=${model_key} label=${label} beta_infer=${beta_infer} snr=[${snr_min},${snr_max}] gen=${gen_len}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "${model_key}" \
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

run_eval_sweep() {
  local run_id="$1"
  local beta_family="$2"
  local model_key="probe_${run_id}"
  local -a configs=()

  if [[ "${beta_family}" == "b05" ]]; then
    configs=(
      "${model_key} ${run_id}_bi0p5_snr1_100_g128 0.5 1 100 128"
      "${model_key} ${run_id}_bi1_snr1_100_g128 1.0 1 100 128"
      "${model_key} ${run_id}_bi1_snr10_100_g128 1.0 10 100 128"
      "${model_key} ${run_id}_bi2_snr1_100_g64 2.0 1 100 64"
      "${model_key} ${run_id}_bi2_snr1_100_g128 2.0 1 100 128"
      "${model_key} ${run_id}_bi3_snr1_100_g64 3.0 1 100 64"
    )
  else
    configs=(
      "${model_key} ${run_id}_bi0p3_snr1_150_g128 0.3 1 150 128"
      "${model_key} ${run_id}_bi0p5_snr1_150_g128 0.5 1 150 128"
      "${model_key} ${run_id}_bi1_snr1_150_g128 1.0 1 150 128"
      "${model_key} ${run_id}_bi1_snr10_150_g128 1.0 10 150 128"
      "${model_key} ${run_id}_bi2_snr1_150_g64 2.0 1 150 64"
      "${model_key} ${run_id}_bi2_snr1_150_g128 2.0 1 150 128"
    )
  fi

  local gpu_arr=(${GPUS})
  local gpu_i=0
  for cfg in "${configs[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
      sleep 5
    done
    read -r model_key label beta_infer snr_min snr_max gen_len <<<"${cfg}"
    local gpu="${gpu_arr[$((gpu_i % ${#gpu_arr[@]}))]}"
    gpu_i=$((gpu_i + 1))
    run_eval_one "${gpu}" "${model_key}" "${label}" "${beta_infer}" "${snr_min}" "${snr_max}" "${gen_len}" &
  done
  wait
}

run_train() {
  local run_id="$1"
  local beta="$2"
  local mu="$3"
  local sigma="$4"
  local snr_max_ln="$5"

  echo "[resume:${PAIR_ID}] train ${run_id} beta=${beta} mu=${mu} sigma=${sigma} snr_max_ln=${snr_max_ln}"
  RUN_ID="${run_id}" \
  STEPS=1000 \
  BETA_INIT="${beta}" \
  SNR_MU="${mu}" \
  SNR_SIGMA="${sigma}" \
  SNR_MAX=100 \
  SNR_MAX_LN="${snr_max_ln}" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  WANDB_PROJECT=dsl-llada-unit-uniform-probes \
  WANDB_NAME="${run_id}" \
    bash scripts/run_unit_uniform_probe.sh
}

run_a="unit_uniform_b05_mu322_sig06_snrln80_1k_${PAIR_ID}"
run_b="unit_uniform_b03_mu369_sig06_snrln100_1k_${PAIR_ID}"

run_eval_sweep "${run_a}" "b05"
run_train "${run_b}" "0.3" "3.69" "0.6" "100"
run_eval_sweep "${run_b}" "b03"

echo "[resume:${PAIR_ID}] complete"
