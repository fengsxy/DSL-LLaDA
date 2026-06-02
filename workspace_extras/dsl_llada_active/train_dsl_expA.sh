#!/bin/bash
# Experiment A: Converter LR ×50, same SNR distribution
# Goal: Test if higher converter LR alone fixes the mid-SNR plateau
# Key change: DSL_CONVERTER_LR_SCALE=50 → effective converter lr = 1e-3
set -e
source /home/ubuntu/efs/Retrieval_head/retrieval_head/.venv/bin/activate
cd /home/ubuntu/efs/RMDM

export DSL_NOISE_DIM=48
export DSL_SNR_MU=1.69
export DSL_SNR_SIGMA=0.9
export DSL_SNR_MAX_LN=40
export DSL_CONVERTER_LR_SCALE=50
export DSL_FREEZE_FFOUT=1

OUTPUT_DIR="./checkpoints/expA_convlr50"
echo "=== Experiment A: Converter LR ×50 ==="
echo "  SNR: mu=1.69, sigma=0.9 (same as baseline)"
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
    --run_name "expA_convlr50" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
