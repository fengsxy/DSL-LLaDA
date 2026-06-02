#!/bin/bash
# Experiment 2: Continue SM-1K (dim=48) to 5K steps
# Purpose: isolate dim=48 vs dim=64 effect
# Resume from checkpoint-1000, train to step 5000
# Expected: ~8-12 hours on 8xA100

set -e

source /home/ubuntu/efs/Retrieval_head/retrieval_head/.venv/bin/activate

# CRITICAL: dim=48 to match SM-1K (not 64 like SM-5K)
export DSL_NOISE_DIM=48
export DSL_SNR_MU=${DSL_SNR_MU:-1.69}
export DSL_SNR_SIGMA=${DSL_SNR_SIGMA:-0.9}
export DSL_SNR_MAX_LN=${DSL_SNR_MAX_LN:-40}
export DSL_EVAL_BUFFER_SIZE=${DSL_EVAL_BUFFER_SIZE:-8}

RESUME_CKPT="checkpoints/dsl_1000step/checkpoint-1000"
OUTPUT_DIR="./checkpoints/dsl_exp2_dim48_5k_$(date +%Y%m%d_%H%M%S)"

echo "=== Experiment 2: SM-1K → 5K (dim=48) ==="
echo "Load weights from: ${RESUME_CKPT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "DSL: noise_dim=${DSL_NOISE_DIM}, SNR mu=${DSL_SNR_MU} sigma=${DSL_SNR_SIGMA}"
echo "NOTE: optimizer resets (checkpoint has no DSL params for full resume)"

deepspeed dsl_llada/llada_cpt_dsl.py \
    --deepspeed dsl_llada/ds_config.json \
    --model_name_or_path "${RESUME_CKPT}" \
    --config_name    "GSAI-ML/LLaDA-8B-Instruct" \
    --tokenizer_name "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset_name "HuggingFaceFW/fineweb-edu" \
    --dataset_config_name sample-10BT \
    --streaming \
    --block_size 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --bf16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type "constant_with_warmup" \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --logging_steps 2 \
    --save_steps 1000 \
    --max_steps 4000 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to wandb \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
