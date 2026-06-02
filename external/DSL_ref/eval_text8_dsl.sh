#!/bin/bash

# 指定可见的 GPU：只暴露卡号 2 和 3
# export CUDA_VISIBLE_DEVICES=2,3

python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=text8 \
  model=small \
  backbone=dit \
  +wandb.offline=true \
  sampling.sampling_method=per_sentence  \
  eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/text8/2025.05.14_best_ckpt/192015/checkpoints/best.ckpt \
  eval.compute_generative_perplexity=True
  # eval.checkpoint_path=/gpfs/data/wulab/jiayi/DSL/outputs/text8/current_best/best.ckpt
  # model.length=1024 \