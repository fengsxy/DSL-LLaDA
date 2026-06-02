#!/bin/bash
# Debug: ZeRO-3 + ff_out unfrozen, 10 steps to verify no NaN
set -e

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

export DSL_NOISE_DIM=48
export DSL_SNR_MU=1.69
export DSL_SNR_SIGMA=0.9
export DSL_SNR_MAX_LN=40
export DSL_FREEZE_FFOUT=0   # <-- KEY: unfreeze ff_out

OUTPUT_DIR="./checkpoints/z3_debug_$(date +%Y%m%d_%H%M%S)"

echo "=== ZeRO-3 Debug: ff_out UNFROZEN ==="
echo "Output dir: ${OUTPUT_DIR}"

deepspeed dsl_llada/train/llada_cpt_dsl.py \
    --deepspeed dsl_llada/configs/ds_config_stage3.json \
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
    --warmup_steps 2 \
    --logging_steps 1 \
    --save_steps 100 \
    --max_steps 10 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to none \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
