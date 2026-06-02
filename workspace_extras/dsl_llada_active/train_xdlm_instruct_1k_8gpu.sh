#!/usr/bin/env bash
# Train an XDLM-CPT baseline from LLaDA-8B-Instruct on FineWeb-Edu.
#
# This is intentionally not the official Base checkpoint setting. It matches
# the paper workspace's Instruct-starting-point baselines while using XDLM's
# k1=0.1 mixed noise objective.
set -euo pipefail

ROOT="/home/ylong030/dsllada/AWS_private_2"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-opencompass}"
source /home/ylong030/miniconda3/bin/activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ylong030_triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ylong030_xdg_cache}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29731}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/ylong030/checkpoints/xdlm_instruct_fineweb_1k}"
NUM_GPUS="$(awk -F',' '{print NF}' <<< "${GPUS}")"
DS_CONFIG="${DS_CONFIG:-dsl_llada/ds_config.json}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-sample-10BT}"
TRAIN_FILE="${TRAIN_FILE:-}"
VALIDATION_FILE="${VALIDATION_FILE:-}"
STREAMING="${STREAMING:-true}"

BLOCK_SIZE="${BLOCK_SIZE:-2048}"
MICRO_BS="${MICRO_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-128}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-2}"
OPTIM="${OPTIM:-adamw_torch}"

echo "=== LLaDA-Instruct XDLM-CPT ==="
echo "  gpus: ${GPUS}"
echo "  output: ${OUTPUT_DIR}"
echo "  conda env: ${CONDA_ENV}"
echo "  deepspeed config: ${DS_CONFIG}"
if [[ -n "${TRAIN_FILE}" ]]; then
    echo "  train file: ${TRAIN_FILE}"
    if [[ -n "${VALIDATION_FILE}" ]]; then
        echo "  validation file: ${VALIDATION_FILE}"
    fi
else
    echo "  dataset: ${DATASET_NAME}/${DATASET_CONFIG_NAME}"
fi
echo "  block_size: ${BLOCK_SIZE}"
echo "  per-device micro batch: ${MICRO_BS}"
echo "  grad accumulation: ${GRAD_ACCUM}"
echo "  optimizer: ${OPTIM}"
echo "  global sequences/update: $((NUM_GPUS * MICRO_BS * GRAD_ACCUM))"
echo "  global tokens/update: $((NUM_GPUS * MICRO_BS * GRAD_ACCUM * BLOCK_SIZE))"

DATA_ARGS=()
if [[ -n "${TRAIN_FILE}" ]]; then
    DATA_ARGS+=(--train_file "${TRAIN_FILE}")
    if [[ -n "${VALIDATION_FILE}" ]]; then
        DATA_ARGS+=(--validation_file "${VALIDATION_FILE}")
    fi
else
    DATA_ARGS+=(--dataset_name "${DATASET_NAME}" --dataset_config_name "${DATASET_CONFIG_NAME}")
fi
if [[ "${STREAMING}" == "true" ]]; then
    DATA_ARGS+=(--streaming)
fi

deepspeed --include "localhost:${GPUS}" --master_port "${MASTER_PORT}" LLaDA-XDLM/llada_cpt.py \
    --deepspeed "${DS_CONFIG}" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --config_name "GSAI-ML/LLaDA-8B-Instruct" \
    --tokenizer_name "GSAI-ML/LLaDA-8B-Instruct" \
    --torch_dtype bfloat16 \
    "${DATA_ARGS[@]}" \
    --block_size "${BLOCK_SIZE}" \
    --per_device_train_batch_size "${MICRO_BS}" \
    --per_device_eval_batch_size 1 \
    --do_train \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --bf16 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --lr_scheduler_type "constant_with_warmup" \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --optim "${OPTIM}" \
    --max_grad_norm 1.0 \
    --warmup_steps "${WARMUP_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --max_steps "${MAX_STEPS}" \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to none \
    --run_name "xdlm_instruct_fineweb_1k" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "xdm" \
    --trust_remote_code True
