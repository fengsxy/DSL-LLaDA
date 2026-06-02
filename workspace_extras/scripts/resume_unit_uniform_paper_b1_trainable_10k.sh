#!/usr/bin/env bash
# Continue the paper-b1 unit-uniform trainable-embedding run from 5k to 10k,
# then run XSum SDE and DSL embedding analysis on checkpoint-10000.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
REMOTE_DIR=${REMOTE_DIR:-/tmp/DSLLaDA_fengsxy}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_paper_b1_trainable_5k_20260508_1716}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}}
RESUME_CKPT=${RESUME_CKPT:-${OUTPUT_DIR}/checkpoint-5000}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}
ANALYSIS_DIR=${ANALYSIS_DIR:-${REPO_ROOT}/eval_results/embedding_analysis}

mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}" "${REPO_ROOT}/eval_results"

if [ ! -d "${RESUME_CKPT}" ]; then
  echo "[error] missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

source "${VENV}/bin/activate"
cd "${REMOTE_DIR}"

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-dsl-llada-unit-uniform-paper-trainable}
export WANDB_NAME=${WANDB_NAME:-${RUN_ID}_resume10k}

# Match the 1k/5k paper-b1 trainable run.
export DSL_TRAIN_NOISE_EMBED=1
export DSL_NOISE_INIT=random
export DSL_NOISE_DIM=100
export DSL_BETA_INIT=1.0
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

TRAIN_LOG="${LOG_DIR}/${RUN_ID}_resume10k_train.log"
XSUM_LOG="${LOG_DIR}/${RUN_ID}_checkpoint10000_xsum_sde_nfe64.log"
EMBED_LOG="${LOG_DIR}/${RUN_ID}_checkpoint10000_embedding_analysis.log"

echo "[run] resume ${RUN_ID} to 10k"
echo "[output] ${OUTPUT_DIR}"
echo "[resume] ${RESUME_CKPT}"
echo "[train_log] ${TRAIN_LOG}"
echo "[wandb] project=${WANDB_PROJECT} name=${WANDB_NAME} mode=${WANDB_MODE}"

cd "${REMOTE_DIR}"

set +e
deepspeed dsl_llada/train/llada_cpt_dsl.py \
  --deepspeed "${DSL_DEEPSPEED_CONFIG}" \
  --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
  --config_name "GSAI-ML/LLaDA-8B-Instruct" \
  --tokenizer_name "GSAI-ML/LLaDA-8B-Instruct" \
  --dataset_name "HuggingFaceFW/fineweb-edu" \
  --dataset_config_name sample-10BT \
  --streaming \
  --block_size "${DSL_BLOCK_SIZE}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from_checkpoint "${RESUME_CKPT}" \
  --bf16 \
  --gradient_accumulation_steps "${DSL_GRAD_ACCUM}" \
  --lr_scheduler_type "constant_with_warmup" \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_steps 250 \
  --logging_steps 2 \
  --save_steps "${DSL_SAVE_STEPS}" \
  --max_steps "${DSL_MAX_STEPS}" \
  --dataloader_num_workers 1 \
  --preprocessing_num_workers 16 \
  --report_to wandb \
  --run_name "${WANDB_NAME}" \
  --remove_unused_columns False \
  --ddp_find_unused_parameters False \
  --load_best_model_at_end False \
  --training_method "dsl" \
  --trust_remote_code True \
  2>&1 | tee "${TRAIN_LOG}"
train_rc=${PIPESTATUS[0]}
set -e

if [ ! -d "${OUTPUT_DIR}/checkpoint-10000" ]; then
  echo "[error] resume exited rc=${train_rc}, and checkpoint-10000 is missing" >&2
  exit "${train_rc}"
fi

if [ "${train_rc}" -ne 0 ]; then
  echo "[warn] resume command exited rc=${train_rc}, but checkpoint-10000 exists; continuing"
fi

cd "${REPO_ROOT}"

python - <<PY
import json
from pathlib import Path

path = Path("eval_results/registry.json")
reg = json.loads(path.read_text())
reg["uu_trainable_10k"] = {
    "path": "checkpoints/${RUN_ID}/checkpoint-10000",
    "type": "local",
    "description": "Remote DSL unit-uniform beta=1, trainable unit-norm noise embedding, d=100, 10K steps",
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
print("[registry] updated uu_trainable_10k -> checkpoints/${RUN_ID}/checkpoint-10000")
PY

CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0} \
  "${VENV}/bin/python" dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key uu_trainable_10k \
    --nfe 64 \
    --gpu 0 \
    --out_tag "${RUN_ID}_checkpoint10000" \
    2>&1 | tee "${XSUM_LOG}"

"${VENV}/bin/python" dsl_llada/analyze_dsl_embedding_structure.py \
  --checkpoint "${OUTPUT_DIR}/checkpoint-10000" \
  --baseline-checkpoint "${REPO_ROOT}/checkpoints/unit_uniform_paper_b1_trainable_1k_20260508_0320/checkpoint-1000" \
  --output "${ANALYSIS_DIR}/${RUN_ID}_checkpoint10000_embedding.json" \
  2>&1 | tee "${EMBED_LOG}"

echo "[done] ${RUN_ID} checkpoint-10000"
