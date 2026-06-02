#!/bin/bash
# dsl_llada/train_dsl_pca128.sh
# Experiment: PCA semantic noise_embed, dim=128, 1000 steps
# Key change: DSL_NOISE_INIT=pca → noise embedding from PCA(wte), preserves semantic structure
# Expected: ~3 hours on 6xA100
# Usage: bash dsl_llada/train_dsl_pca128.sh [output_dir]

set -e

source /home/ubuntu/efs/Retrieval_head/retrieval_head/.venv/bin/activate

export DSL_NOISE_DIM=128
export DSL_NOISE_INIT=pca        # KEY: PCA of wte instead of random
export DSL_SNR_MU=${DSL_SNR_MU:-1.69}
export DSL_SNR_SIGMA=${DSL_SNR_SIGMA:-0.9}
export DSL_SNR_MAX_LN=${DSL_SNR_MAX_LN:-40}
export DSL_EVAL_BUFFER_SIZE=${DSL_EVAL_BUFFER_SIZE:-8}
export DSL_FREEZE_FFOUT=0   # train ff_out via separate fp32 optimizer

OUTPUT_DIR=${1:-"./checkpoints/dsl_pca128_$(date +%Y%m%d_%H%M%S)"}

echo "=== DSL PCA-128: Semantic noise_embed ==="
echo "Output dir: ${OUTPUT_DIR}"
echo "DSL: noise_dim=${DSL_NOISE_DIM}, noise_init=${DSL_NOISE_INIT}"
echo "SNR: mu=${DSL_SNR_MU} sigma=${DSL_SNR_SIGMA} max_ln=${DSL_SNR_MAX_LN}"

deepspeed dsl_llada/llada_cpt_dsl.py \
    --deepspeed dsl_llada/ds_config.json \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
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
    --save_steps 500 \
    --max_steps 1000 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to wandb \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
