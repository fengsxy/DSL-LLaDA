#!/bin/bash

# 指定可见的 GPU：只暴露卡号 2 和 3
python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  backbone=dit \
  sampling.sampling_method=uniform_t  \
  eval.compute_generative_perplexity=True \
  model.length=1024 \
  model.dim_embed=16 \
  eval.checkpoint_path=/gpfs/data/wulab/jiayi/DSL/outputs/openwebtext-train/2025.04.19/214916/checkpoints/last.ckpt \
  # /gpfs/data/wulab/jiayi/DSL/outputs/openwebtext-train/2025.04.19/224147/checkpoints/last.ckpt \
  # /gpfs/data/wulab/jiayi/DSL/outputs/openwebtext-train/2025.04.19/034431/checkpoints/last.ckpt
  