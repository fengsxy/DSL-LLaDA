#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
REMOTE_DIR=${REMOTE_DIR:-/tmp/DSLLaDA_fengsxy}
VENV=${VENV:-${REPO_ROOT}/.venv}
PAIR_ID=${PAIR_ID:-beta_snr_matched_20260512_225231}
RUN_ID=${RUN_ID:-unit_uniform_b05_mu322_sig06_snrln80_1k_${PAIR_ID}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}}
RESUME_CKPT=${RESUME_CKPT:-${OUTPUT_DIR}/checkpoint-1000}
TARGET_STEPS=${TARGET_STEPS:-5000}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}
EVAL_LOG_DIR=${EVAL_LOG_DIR:-${REPO_ROOT}/logs/sde_param_search}
LIMIT=${LIMIT:-8}

mkdir -p "${LOG_DIR}" "${EVAL_LOG_DIR}" "${REPO_ROOT}/eval_results"

if [ ! -d "${RESUME_CKPT}" ]; then
  echo "[error] missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

source "${VENV}/bin/activate"
cd "${REMOTE_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-dsl-llada-unit-uniform-probes}
export WANDB_NAME=${WANDB_NAME:-${RUN_ID}_resume${TARGET_STEPS}}

export DSL_TRAIN_NOISE_EMBED=1
export DSL_NOISE_INIT=random
export DSL_NOISE_DIM=100
export DSL_BETA_INIT=0.5
export DSL_BBEMB_INIT=wte
export DSL_FREEZE_CONVERTER=0
export DSL_CONVERTER_LR_SCALE=25
export DSL_CONVERTER_TOKEN_CHUNK=256
export DSL_DEEPSPEED_CONFIG="${REPO_ROOT}/dsl_llada/ds_config.json"
export DSL_MAX_STEPS="${TARGET_STEPS}"
export DSL_SAVE_STEPS=1000
export DSL_GRAD_ACCUM=4
export DSL_BLOCK_SIZE=2048
export DSL_FFOUT_LORA_R=32
export DSL_FREEZE_FFOUT=1
export DSL_SNR_MAX=100
export DSL_SNR_MU=3.22
export DSL_SNR_SIGMA=0.6
export DSL_SNR_MAX_LN=80
export DSL_EVAL_BUFFER_SIZE=4
export DSL_EVAL_NLL=0
export DSL_DIAG_INTERVAL=100
export DSL_GEOMETRY_PROXY=1
export DSL_EMBED_HEALTH_INTERVAL=100

DS_ARGS=""
IFS=',' read -r -a _visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
_num_visible_gpus=${#_visible_gpu_array[@]}
_local_slots=$(seq 0 $((_num_visible_gpus - 1)) | paste -sd, -)
DS_ARGS="--include localhost:${_local_slots}"

TRAIN_LOG="${LOG_DIR}/${RUN_ID}_resume${TARGET_STEPS}_train.log"
EVAL_TAG="${PAIR_ID}_${RUN_ID}_checkpoint${TARGET_STEPS}_bi1_snr1_100_g128"
EVAL_LOG="${EVAL_LOG_DIR}/${EVAL_TAG}.log"

echo "[resume-b05] run_id=${RUN_ID}"
echo "[resume-b05] resume=${RESUME_CKPT}"
echo "[resume-b05] target_steps=${TARGET_STEPS}"
echo "[resume-b05] output=${OUTPUT_DIR}"
echo "[resume-b05] wandb project=${WANDB_PROJECT} name=${WANDB_NAME} mode=${WANDB_MODE}"
echo "[resume-b05] train_log=${TRAIN_LOG}"

set +e
deepspeed ${DS_ARGS} dsl_llada/train/llada_cpt_dsl.py \
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

if [ ! -d "${OUTPUT_DIR}/checkpoint-${TARGET_STEPS}" ]; then
  echo "[error] training rc=${train_rc}; checkpoint-${TARGET_STEPS} missing" >&2
  exit "${train_rc}"
fi
if [ "${train_rc}" -ne 0 ]; then
  echo "[warn] training exited rc=${train_rc}, but checkpoint-${TARGET_STEPS} exists; continuing"
fi

cd "${REPO_ROOT}"

"${VENV}/bin/python" - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
steps = int("${TARGET_STEPS}")
key = f"probe_{run_id}_ckpt{steps}"
path = Path("eval_results/registry.json")
reg = json.loads(path.read_text()) if path.exists() else {}
reg[key] = {
    "path": f"checkpoints/{run_id}/checkpoint-{steps}",
    "type": "local",
    "description": "b05 beta/SNR-matched unit-uniform trainable DSL scale checkpoint",
    "dsl": True,
    "dsl_config": {
        "beta_init": 0.5,
        "noise_dim": 100,
        "noise_init": "random_unit_uniform",
        "train_noise_embed": True,
        "snr_mu": 3.22,
        "snr_sigma": 0.6,
        "snr_max": 100.0,
        "snr_max_ln": 80.0,
    },
}
path.write_text(json.dumps(reg, indent=2) + "\\n")
print(f"[registry] updated {key}")
PY

CUDA_VISIBLE_DEVICES=0 "${VENV}/bin/python" dsl_llada/eval_summarization.py \
  --dataset xsum \
  --method sde \
  --model_key "probe_${RUN_ID}_ckpt${TARGET_STEPS}" \
  --gpu 0 \
  --nfe 32 \
  --gen_length 128 \
  --limit "${LIMIT}" \
  --sde_beta_infer 1.0 \
  --sde_noise_scale 0.0 \
  --sde_schedule sensitive \
  --sde_snr_min 1 \
  --sde_snr_max 100 \
  --sde_sensitive_low 7 \
  --sde_sensitive_high 74 \
  --sde_top_k 512 \
  --out_tag "${EVAL_TAG}" \
  2>&1 | tee "${EVAL_LOG}"

echo "[resume-b05] complete"
