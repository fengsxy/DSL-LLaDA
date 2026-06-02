#!/usr/bin/env bash
# Run fengsxy/DSLLaDA unit-norm geometry training as an isolated patch experiment.
# This does not modify the repository's existing dsl_llada implementation.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
REMOTE_DIR=${REMOTE_DIR:-/tmp/DSLLaDA_fengsxy}
REMOTE_URL=${REMOTE_URL:-https://github.com/fengsxy/DSLLaDA.git}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_remote_10k_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}
REGISTRY=${REGISTRY:-${REPO_ROOT}/eval_results/${RUN_ID}_registry.json}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results"

if [ ! -d "${REMOTE_DIR}/.git" ]; then
  git clone --depth 1 "${REMOTE_URL}" "${REMOTE_DIR}"
fi

source "${VENV}/bin/activate"

cd "${REMOTE_DIR}"

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-dsl-llada-unit-uniform}
export DSL_MAX_STEPS=${DSL_MAX_STEPS:-10000}
export DSL_SAVE_STEPS=${DSL_SAVE_STEPS:-1000}
export DSL_TRAIN_NOISE_EMBED=${DSL_TRAIN_NOISE_EMBED:-1}
export DSL_NOISE_INIT=${DSL_NOISE_INIT:-random}
export DSL_NOISE_DIM=${DSL_NOISE_DIM:-128}
export DSL_BETA_INIT=${DSL_BETA_INIT:-0.10}
export DSL_FREEZE_CONVERTER=${DSL_FREEZE_CONVERTER:-0}
export DSL_CONVERTER_LR_SCALE=${DSL_CONVERTER_LR_SCALE:-5}
export DSL_CONVERTER_TOKEN_CHUNK=${DSL_CONVERTER_TOKEN_CHUNK:-256}
export DSL_FFOUT_LORA_R=${DSL_FFOUT_LORA_R:-32}
export DSL_FREEZE_FFOUT=${DSL_FREEZE_FFOUT:-1}
export DSL_SNR_MAX=${DSL_SNR_MAX:-50}
export DSL_SNR_MU=${DSL_SNR_MU:-1.4}
export DSL_SNR_SIGMA=${DSL_SNR_SIGMA:-0.55}
export DSL_SNR_MAX_LN=${DSL_SNR_MAX_LN:-40}
export DSL_EVAL_BUFFER_SIZE=${DSL_EVAL_BUFFER_SIZE:-0}
export DSL_EVAL_NLL=${DSL_EVAL_NLL:-0}
export DSL_GRAD_ACCUM=${DSL_GRAD_ACCUM:-1}
export DSL_BLOCK_SIZE=${DSL_BLOCK_SIZE:-4096}
export DSL_DIAG_INTERVAL=${DSL_DIAG_INTERVAL:-200}
export DSL_GEOMETRY_PROXY=${DSL_GEOMETRY_PROXY:-0}
export DSL_EMBED_HEALTH_INTERVAL=${DSL_EMBED_HEALTH_INTERVAL:-1000}

echo "[run] output_dir=${OUTPUT_DIR}"
echo "[run] registry=${REGISTRY}"
echo "[run] log_dir=${LOG_DIR}"

bash dsl_llada/scripts/train_dsl_geometry_10k.sh "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}_train.log"

python - "${REGISTRY}" "${OUTPUT_DIR}" <<'PY'
import json
import sys

registry_path, output_dir = sys.argv[1], sys.argv[2]
entries = {}
for step in (1000, 5000, 10000):
    key = f"unit_uniform_{step // 1000}k"
    entries[key] = {
        "path": f"{output_dir}/checkpoint-{step}",
        "type": "local",
        "description": f"fengsxy DSLLaDA unit-norm trainable random embedding, {step} steps",
        "dsl": True,
        "dsl_config": {
            "beta_init": 0.10,
            "noise_dim": 128,
            "noise_init": "random",
            "train_noise_embed": True,
        },
        "gen_methods": ["sde"],
    }
with open(registry_path, "w") as f:
    json.dump(entries, f, indent=2)
PY

export DSL_LLADA_REGISTRY="${REGISTRY}"
for step in 1000 5000 10000; do
  key="unit_uniform_$((step / 1000))k"
  for nfe in 16 64; do
    python dsl_llada/eval/eval_sde_gen_formal.py \
      --model_key "${key}" \
      --method sde \
      --nfe "${nfe}" \
      --gen_length 256 \
      --prompts "${REPO_ROOT}/eval_data/sde_prompts_200.json" \
      --gpu 0 \
      --tag "${RUN_ID}_prompted" \
      2>&1 | tee "${LOG_DIR}/${RUN_ID}_${key}_sde_nfe${nfe}.log"
    mkdir -p "${REPO_ROOT}/eval_results/sde_gen_formal"
    cp -n eval_results/sde_gen_formal/"${RUN_ID}_prompted_${key}_sde_nfe${nfe}"_gen256*.json \
      "${REPO_ROOT}/eval_results/sde_gen_formal/" 2>/dev/null || true
  done
done

echo "[done] ${RUN_ID}"
