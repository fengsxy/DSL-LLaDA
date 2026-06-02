#!/bin/bash
# dsl_llada/train_dsl.sh
# Run from: /home/ubuntu/efs/RMDM
# Usage: bash dsl_llada/train_dsl.sh [output_dir]
#
# Requires:
#   - conda activate xdlm  (or equivalent env with deepspeed, transformers==4.38.2)
#   - HF_DATASETS_CACHE set to where FineWeb-Edu is cached, OR internet access
#   - 8× A100/H100 GPUs with ≥40GB each

set -e

# DSL hyperparams scaled for LLaDA 128k vocab (see CLAUDE.md for derivation)
export DSL_NOISE_DIM=${DSL_NOISE_DIM:-48}
export DSL_SNR_MU=${DSL_SNR_MU:-1.69}
export DSL_SNR_SIGMA=${DSL_SNR_SIGMA:-0.9}
export DSL_SNR_MAX_LN=${DSL_SNR_MAX_LN:-40}

OUTPUT_DIR=${1:-"./checkpoints/dsl_phase1_$(date +%Y%m%d_%H%M%S)"}

echo "Output dir: ${OUTPUT_DIR}"
echo "DSL: noise_dim=${DSL_NOISE_DIM}, SNR mu=${DSL_SNR_MU} sigma=${DSL_SNR_SIGMA}"

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
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type "constant_with_warmup" \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --logging_steps 2 \
    --save_steps 100 \
    --max_steps 1000 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to wandb \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
