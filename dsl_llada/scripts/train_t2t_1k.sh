#!/bin/bash
# T2T (Token-to-Token) training: replace MASK with model's own predictions.
# No converter, no noise embedding, no SNR. Just self-conditioning.
# 50% MASK (standard MDM), 50% model's own predictions (self-conditioning).
set -e
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

export T2T_SELF_COND_PROB=0.5

OUTPUT_DIR="./checkpoints/t2t_selfcond_1k"

echo "=== T2T Self-Conditioning, 1K steps ==="

deepspeed --include localhost:0,1,4,5,6,7 --master_port 29700 LLaDA-XDLM/llada_cpt.py \
    --deepspeed dsl_llada/configs/ds_config.json \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --config_name    "GSAI-ML/LLaDA-8B-Instruct" \
    --tokenizer_name "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset_name "HuggingFaceFW/fineweb-edu" \
    --dataset_config_name sample-10BT \
    --streaming \
    --block_size 2048 \
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
    --max_steps 1000 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to wandb \
    --run_name "t2t_selfcond_1k" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "t2t" \
    --trust_remote_code True
