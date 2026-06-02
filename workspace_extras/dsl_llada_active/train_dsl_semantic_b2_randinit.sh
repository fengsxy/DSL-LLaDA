#!/bin/bash
# Semantic embed + random backbone_embedding init
# Tests if wte.T init is the bottleneck for semantic embed training
set -e
source /home/ubuntu/efs/RMDM/.venv/bin/activate

export DSL_NOISE_DIM=100
export DSL_NOISE_INIT=ae_contrastive
export DSL_AE_EMBED_PATH=results/wte_ae_contrastive_embedding.pt
export DSL_BETA_INIT=2.0
export DSL_BBEMB_INIT=random          # key change: random init
export DSL_SNR_MU=1.69
export DSL_SNR_SIGMA=0.9
export DSL_SNR_MAX_LN=40
export DSL_CONVERTER_LR_SCALE=1       # same LR as backbone, co-adapt together
export DSL_FREEZE_FFOUT=1

OUTPUT_DIR="./checkpoints/semantic_b2_randinit_1k"
EXCLUDE_GPUS=${EXCLUDE_GPUS:-""}
DS_ARGS=""
if [ -n "$EXCLUDE_GPUS" ]; then
    DS_ARGS="--exclude localhost:${EXCLUDE_GPUS}"
fi

echo "=== Semantic + Random bbemb init ==="
echo "  bbemb_init: ${DSL_BBEMB_INIT}"
echo "  converter_lr_scale: ${DSL_CONVERTER_LR_SCALE}"
echo "  output: ${OUTPUT_DIR}"

deepspeed ${DS_ARGS} dsl_llada/llada_cpt_dsl.py \
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
    --run_name "semantic_b2_randinit_1k" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "dsl" \
    --trust_remote_code True
