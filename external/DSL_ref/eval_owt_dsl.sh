#!/bin/bash

python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  backbone=dit \
  +wandb.offline=true \
  sampling.sampling_method=uniform_t  \
  eval.compute_generative_perplexity=True \
  model.length=1024 \
  model.dim_embed=64 \
  eval.checkpoint_path=/gpfs/data/wulab/jiayi/DSL/outputs/openwebtext-train/2025.04.19/034431/checkpoints/last.ckpt \