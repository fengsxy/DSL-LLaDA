#!/bin/bash

# 指定可见的 GPU：只暴露卡号 2 和 3
export CUDA_VISIBLE_DEVICES=2,3

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  model=small \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  sampling.num_sample_batches=2 \
  backbone=dit \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/openwebtext-train/2025.04.20/195459/checkpoints/last.ckpt
