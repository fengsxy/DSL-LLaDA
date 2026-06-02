#!/usr/bin/env bash
# Train the paper-b1 unit-uniform trainable-embedding DSL run to 5k steps,
# then run the two agreed checks: XSum SDE and DSL embedding structure.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
REMOTE_DIR=${REMOTE_DIR:-/tmp/DSLLaDA_fengsxy}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_paper_b1_trainable_5k_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}
ANALYSIS_DIR=${ANALYSIS_DIR:-${REPO_ROOT}/eval_results/embedding_analysis}

mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}" "${REPO_ROOT}/eval_results"

source "${VENV}/bin/activate"
cd "${REMOTE_DIR}"

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-dsl-llada-unit-uniform-paper-trainable}

# Match the evaluated 1k run:
# checkpoints/unit_uniform_paper_b1_trainable_1k_20260508_0320/checkpoint-1000
export DSL_TRAIN_NOISE_EMBED=1
export DSL_NOISE_INIT=random
export DSL_NOISE_DIM=100
export DSL_BETA_INIT=1.0
export DSL_BBEMB_INIT=wte
export DSL_FREEZE_CONVERTER=0
export DSL_CONVERTER_LR_SCALE=25
export DSL_CONVERTER_TOKEN_CHUNK=256
export DSL_DEEPSPEED_CONFIG="${REPO_ROOT}/dsl_llada/ds_config.json"
export DSL_MAX_STEPS=5000
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
XSUM_LOG="${LOG_DIR}/${RUN_ID}_xsum_sde_nfe64.log"
EMBED_LOG="${LOG_DIR}/${RUN_ID}_embedding_analysis.log"

echo "[run] ${RUN_ID}"
echo "[output] ${OUTPUT_DIR}"
echo "[train_log] ${TRAIN_LOG}"

set +e
bash dsl_llada/scripts/train_dsl_geometry_10k.sh "${OUTPUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"
train_rc=${PIPESTATUS[0]}
set -e

if [ ! -d "${OUTPUT_DIR}/checkpoint-5000" ]; then
  echo "[error] training exited rc=${train_rc}, and checkpoint-5000 is missing" >&2
  exit "${train_rc}"
fi

if [ "${train_rc}" -ne 0 ]; then
  echo "[warn] training command exited rc=${train_rc}, but checkpoint-5000 exists; continuing"
fi

cd "${REPO_ROOT}"

python - <<PY
import json
from pathlib import Path

path = Path("eval_results/registry.json")
reg = json.loads(path.read_text())
reg["uu_trainable_5k"] = {
    "path": "checkpoints/${RUN_ID}/checkpoint-5000",
    "type": "local",
    "description": "Remote DSL unit-uniform beta=1, trainable unit-norm noise embedding, d=100, 5K steps",
    "dsl": True,
    "dsl_config": {
        "beta_init": 1.0,
        "noise_dim": 100,
        "noise_init": "random_unit_uniform",
        "train_noise_embed": True,
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
print("[registry] updated uu_trainable_5k -> checkpoints/${RUN_ID}/checkpoint-5000")
PY

CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0} \
  "${VENV}/bin/python" dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key uu_trainable_5k \
    --nfe 64 \
    --gpu 0 \
    --out_tag "${RUN_ID}" \
    2>&1 | tee "${XSUM_LOG}"

"${VENV}/bin/python" dsl_llada/analyze_dsl_embedding_structure.py \
  --checkpoint "${OUTPUT_DIR}/checkpoint-5000" \
  --baseline-checkpoint "${REPO_ROOT}/checkpoints/unit_uniform_paper_b1_trainable_1k_20260508_0320/checkpoint-1000" \
  --output "${ANALYSIS_DIR}/${RUN_ID}_checkpoint5000_embedding.json" \
  2>&1 | tee "${EMBED_LOG}"

echo "[done] ${RUN_ID}"
