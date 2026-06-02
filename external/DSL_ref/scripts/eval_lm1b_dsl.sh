#!/bin/bash

export NCCL_TIMEOUT=3600000  # 设置为60分钟

python main.py \
  mode=ppl_eval \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=lm1b \
  model.dim_embed=128 \
  model.length=128 \
  model=small \
  backbone=dit \
  +wandb.offline=true \
  sampling.sampling_method=uniform_t \
  eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/lm1b/2025.05.05/150244/checkpoints/best.ckpt \
  eval.compute_generative_perplexity=True > eval_lm1b_dsl.log