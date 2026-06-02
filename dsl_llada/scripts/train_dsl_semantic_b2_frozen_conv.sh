#!/bin/bash
# Semantic embed + FROZEN converter (wte.T init)
# Only backbone learns to adapt to fixed converter output
set -e
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

export DSL_NOISE_DIM=100
export DSL_NOISE_INIT=ae_contrastive
export DSL_AE_EMBED_PATH=results/wte_ae_contrastive_embedding.pt
export DSL_BETA_INIT=2.0
export DSL_BBEMB_INIT=wte
export DSL_FREEZE_CONVERTER=1         # freeze converter entirely
export DSL_CONVERTER_LR_SCALE=1       # irrelevant since frozen, but set anyway
export DSL_SNR_MU=1.69
export DSL_SNR_SIGMA=0.9
export DSL_SNR_MAX_LN=40
export DSL_FREEZE_FFOUT=1

OUTPUT_DIR="./checkpoints/semantic_b2_frozenconv_1k"
EXCLUDE_GPUS=${EXCLUDE_GPUS:-""}
DS_ARGS=""
if [ -n "$EXCLUDE_GPUS" ]; then
    DS_ARGS="--exclude localhost:${EXCLUDE_GPUS}"
fi

echo "=== Semantic + Frozen Converter ==="
echo "  converter: FROZEN"
echo "  output: ${OUTPUT_DIR}"

deepspeed ${DS_ARGS} dsl_llada/train/llada_cpt_dsl.py \
    --deepspeed dsl_llada/configs/ds_config.json \
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
    --gradient_accumulation_steps 8 \
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
    --run_name "semantic_b2_frozenconv_1k" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
