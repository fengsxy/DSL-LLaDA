#!/usr/bin/env bash
set -euo pipefail

# Same-compute MDM-CPT control used by the paper tables.
# This is standard binary masked-diffusion continued pretraining for 1k steps,
# without DSL continuous-noise modules.

export HF_HOME="${HF_HOME:-/data2/ylong030/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data2/ylong030/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data2/ylong030/huggingface/datasets}"

OUTPUT_DIR="${1:-checkpoints/mdm_baseline_1k}"
mkdir -p "$OUTPUT_DIR"

deepspeed dsl_llada/llada_cpt_dsl.py \
  --deepspeed dsl_llada/ds_config.json \
  --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
  --config_name "GSAI-ML/LLaDA-8B-Instruct" \
  --tokenizer_name "GSAI-ML/LLaDA-8B-Instruct" \
  --dataset_name "HuggingFaceFW/fineweb-edu" \
  --dataset_config_name sample-10BT \
  --streaming \
  --block_size 4096 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --output_dir "$OUTPUT_DIR" \
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
  --training_method "mdm" \
  --trust_remote_code True
