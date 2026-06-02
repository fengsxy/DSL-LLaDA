#!/usr/bin/env bash
# Train a fresh paper-aligned unit-uniform DSL run for 10k steps with
# beta_init=0.1 and LogNormal SNR mu=1.69, sigma=0.9.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
REMOTE_DIR=${REMOTE_DIR:-/tmp/DSLLaDA_fengsxy}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_b01_mu169_sig09_trainable_10k_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results"

source "${VENV}/bin/activate"
cd "${REMOTE_DIR}"

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-dsl-llada-unit-uniform-paper-trainable}
export WANDB_NAME=${WANDB_NAME:-${RUN_ID}}

# Match the evaluated paper-b1 trainable run, except beta_init=0.1.
export DSL_TRAIN_NOISE_EMBED=1
export DSL_NOISE_INIT=random
export DSL_NOISE_DIM=100
export DSL_BETA_INIT=0.10
export DSL_BBEMB_INIT=wte
export DSL_FREEZE_CONVERTER=0
export DSL_CONVERTER_LR_SCALE=25
export DSL_CONVERTER_TOKEN_CHUNK=256
export DSL_DEEPSPEED_CONFIG="${REPO_ROOT}/dsl_llada/ds_config.json"
export DSL_MAX_STEPS=10000
export DSL_SAVE_STEPS=1000
export DSL_GRAD_ACCUM=4
export DSL_BLOCK_SIZE=2048
export DSL_FFOUT_LORA_R=32
export DSL_FREEZE_FFOUT=1
export DSL_SNR_MAX=100
export DSL_SNR_MU=1.69
export DSL_SNR_SIGMA=0.9
export DSL_SNR_MAX_LN=40
export DSL_EVAL_BUFFER_SIZE=0
export DSL_EVAL_NLL=0
export DSL_DIAG_INTERVAL=100
export DSL_GEOMETRY_PROXY=1
export DSL_EMBED_HEALTH_INTERVAL=100

TRAIN_LOG="${LOG_DIR}/${RUN_ID}_train.log"

echo "[run] ${RUN_ID}"
echo "[output] ${OUTPUT_DIR}"
echo "[train_log] ${TRAIN_LOG}"
echo "[wandb] project=${WANDB_PROJECT} name=${WANDB_NAME} mode=${WANDB_MODE}"

set +e
bash dsl_llada/scripts/train_dsl_geometry_10k.sh "${OUTPUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"
train_rc=${PIPESTATUS[0]}
set -e

if [ ! -d "${OUTPUT_DIR}/checkpoint-10000" ]; then
  echo "[error] training exited rc=${train_rc}, and checkpoint-10000 is missing" >&2
  exit "${train_rc}"
fi

if [ "${train_rc}" -ne 0 ]; then
  echo "[warn] training command exited rc=${train_rc}, but checkpoint-10000 exists; continuing"
fi

cd "${REPO_ROOT}"

python - <<PY
import json
from pathlib import Path

path = Path("eval_results/registry.json")
reg = json.loads(path.read_text()) if path.exists() else {}
reg["uu_b01_mu169_sig09_10k"] = {
    "path": "checkpoints/${RUN_ID}/checkpoint-10000",
    "type": "local",
    "description": "Remote DSL unit-uniform beta_init=0.1, trainable unit-norm noise embedding, d=100, mu=1.69 sigma=0.9, 10K steps",
    "dsl": True,
    "dsl_config": {
        "beta_init": 0.1,
        "noise_dim": 100,
        "noise_init": "random_unit_uniform",
        "train_noise_embed": True,
        "snr_mu": 1.69,
        "snr_sigma": 0.9,
        "snr_max": 100,
        "snr_max_ln": 40,
    },
    "sde_config": {
        "beta_infer": 2.0,
        "noise_scale": 0.05,
        "schedule": [3, 100],
        "steps": 32,
        "solver": "heun",
    },
    "gen_methods": ["sde", "remask_free", "remask_suppress_block32"],
}
path.write_text(json.dumps(reg, indent=2) + "\\n")
print("[registry] updated uu_b01_mu169_sig09_10k -> checkpoints/${RUN_ID}/checkpoint-10000")
PY

echo "[done] ${RUN_ID}"
