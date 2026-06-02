#!/bin/bash
# Experiment B: Converter LR ×50 + focused SNR distribution
# Goal: Test if concentrating SNR on mid-range (5-11) helps further
# Key changes: DSL_CONVERTER_LR_SCALE=50 + mu=2.0, sigma=0.5
#   mu=2.0 → median SNR ≈ 7.4 (was 5.4)
#   sigma=0.5 → 66% in [5, 11] (was [2, 15], too wide)
set -e
source /home/ubuntu/efs/Retrieval_head/retrieval_head/.venv/bin/activate
cd /home/ubuntu/efs/RMDM

export DSL_NOISE_DIM=48
export DSL_SNR_MU=2.0
export DSL_SNR_SIGMA=0.5
export DSL_SNR_MAX_LN=40
export DSL_CONVERTER_LR_SCALE=50
export DSL_FREEZE_FFOUT=1

OUTPUT_DIR="./checkpoints/expB_convlr50_snr20"
echo "=== Experiment B: Converter LR ×50 + Focused SNR ==="
echo "  SNR: mu=2.0, sigma=0.5 → median=7.4, 66% in [5,11]"
echo "  Converter grad scale: 50 (effective lr=1e-3)"
echo "  Output: ${OUTPUT_DIR}"

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
    --save_steps 100 \
    --max_steps 500 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to wandb \
    --run_name "expB_convlr50_snr20" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
