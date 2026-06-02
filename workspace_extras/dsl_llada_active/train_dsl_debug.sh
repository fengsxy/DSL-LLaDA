#!/bin/bash
# Debug training: 8-GPU, bf16+ZeRO-2, 50 steps to verify convergence
# Key settings: noise_dim=48, SNR mu=1.69/sigma=0.9 (mid-range focus), beta=5.0
set -e
source /home/ubuntu/efs/Retrieval_head/retrieval_head/.venv/bin/activate
export DSL_NOISE_DIM=48
export DSL_SNR_MU=1.69
export DSL_SNR_SIGMA=0.9
export DSL_SNR_MAX_LN=40
deepspeed dsl_llada/llada_cpt_dsl.py \
    --deepspeed dsl_llada/ds_config.json \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --config_name    GSAI-ML/LLaDA-8B-Instruct \
    --tokenizer_name GSAI-ML/LLaDA-8B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --streaming \
    --block_size 4096 \
    --per_device_train_batch_size 1 \
    --do_train \
    --output_dir ./checkpoints/dsl_debug \
    --overwrite_output_dir \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type constant_with_warmup \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 1 \
    --max_steps 50 \
    --dataloader_num_workers 0 \
    --report_to none \
    --bf16 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --training_method dsl \
    --trust_remote_code True \
    2>&1 | tee logs/dsl_debug.log
