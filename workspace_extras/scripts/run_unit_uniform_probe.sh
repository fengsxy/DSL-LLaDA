#!/usr/bin/env bash
# Reusable short probe launcher for unit-uniform trainable DSL runs.
# This is intentionally not auto-launched by any watcher. Use after reading the
# current 10k eval report to test beta/SNR hypotheses for 100 or 1000 steps.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
REMOTE_DIR=${REMOTE_DIR:-/tmp/DSLLaDA_fengsxy}
VENV=${VENV:-${REPO_ROOT}/.venv}
STEPS=${STEPS:-100}
BETA_INIT=${BETA_INIT:-0.5}
SNR_MU=${SNR_MU:-1.69}
SNR_SIGMA=${SNR_SIGMA:-0.9}
SNR_MAX=${SNR_MAX:-100}
SNR_MAX_LN=${SNR_MAX_LN:-40}
CONVERTER_LR_SCALE=${CONVERTER_LR_SCALE:-25}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
RUN_ID=${RUN_ID:-unit_uniform_probe_b${BETA_INIT}_mu${SNR_MU}_sig${SNR_SIGMA}_${STEPS}step_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results"

source "${VENV}/bin/activate"
cd "${REMOTE_DIR}"

export CUDA_VISIBLE_DEVICES
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-dsl-llada-unit-uniform-probes}
export WANDB_NAME=${WANDB_NAME:-${RUN_ID}}

export DSL_TRAIN_NOISE_EMBED=1
export DSL_NOISE_INIT=random
export DSL_NOISE_DIM=100
export DSL_BETA_INIT="${BETA_INIT}"
export DSL_BBEMB_INIT=wte
export DSL_FREEZE_CONVERTER=0
export DSL_CONVERTER_LR_SCALE="${CONVERTER_LR_SCALE}"
export DSL_CONVERTER_TOKEN_CHUNK=256
export DSL_DEEPSPEED_CONFIG="${REPO_ROOT}/dsl_llada/ds_config.json"
export DSL_MAX_STEPS="${STEPS}"
export DSL_SAVE_STEPS="${STEPS}"
export DSL_GRAD_ACCUM=4
export DSL_BLOCK_SIZE=2048
export DSL_FFOUT_LORA_R=32
export DSL_FREEZE_FFOUT=1
export DSL_SNR_MAX="${SNR_MAX}"
export DSL_SNR_MU="${SNR_MU}"
export DSL_SNR_SIGMA="${SNR_SIGMA}"
export DSL_SNR_MAX_LN="${SNR_MAX_LN}"
export DSL_EVAL_BUFFER_SIZE=4
export DSL_EVAL_NLL=0
export DSL_DIAG_INTERVAL=25
export DSL_GEOMETRY_PROXY=1
export DSL_EMBED_HEALTH_INTERVAL=25

TRAIN_LOG="${LOG_DIR}/${RUN_ID}_train.log"

echo "[probe] ${RUN_ID}"
echo "[output] ${OUTPUT_DIR}"
echo "[steps] ${STEPS}"
echo "[beta_init] ${BETA_INIT}"
echo "[snr] mu=${SNR_MU} sigma=${SNR_SIGMA} max=${SNR_MAX} max_ln=${SNR_MAX_LN}"
echo "[converter_lr_scale] ${CONVERTER_LR_SCALE}"
echo "[gpus] ${CUDA_VISIBLE_DEVICES}"
echo "[train_log] ${TRAIN_LOG}"
echo "[wandb] project=${WANDB_PROJECT} name=${WANDB_NAME} mode=${WANDB_MODE}"

set +e
bash dsl_llada/scripts/train_dsl_geometry_10k.sh "${OUTPUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"
train_status=${PIPESTATUS[0]}
set -e

if [ ! -d "${OUTPUT_DIR}/checkpoint-${STEPS}" ]; then
  echo "[error] expected checkpoint-${STEPS} missing under ${OUTPUT_DIR}" >&2
  exit 1
fi
if [ "${train_status}" -ne 0 ]; then
  echo "[warn] train command exited with status ${train_status}, but checkpoint-${STEPS} exists; continuing"
fi

cd "${REPO_ROOT}"

"${VENV}/bin/python" - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
steps = int("${STEPS}")
key = f"probe_{run_id}"
path = Path("eval_results/registry.json")
reg = json.loads(path.read_text()) if path.exists() else {}
reg[key] = {
    "path": f"checkpoints/{run_id}/checkpoint-{steps}",
    "type": "local",
    "description": "Short unit-uniform trainable DSL probe",
    "dsl": True,
    "dsl_config": {
        "beta_init": float("${BETA_INIT}"),
        "noise_dim": 100,
        "noise_init": "random_unit_uniform",
        "train_noise_embed": True,
        "snr_mu": float("${SNR_MU}"),
        "snr_sigma": float("${SNR_SIGMA}"),
        "snr_max": float("${SNR_MAX}"),
        "snr_max_ln": float("${SNR_MAX_LN}"),
    },
    "sde_config": {
        "beta_infer": 2.0,
        "noise_scale": 0.05,
        "schedule": [3, 100],
        "steps": 32,
        "solver": "heun",
    },
}
path.write_text(json.dumps(reg, indent=2) + "\\n")
print(f"[registry] updated {key} -> checkpoints/{run_id}/checkpoint-{steps}")
PY

echo "[done] ${RUN_ID}"
