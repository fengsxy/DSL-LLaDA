#!/bin/bash

# 指定可见的 GPU：只暴露卡号 2 和 3
export CUDA_VISIBLE_DEVICES=2,3

python main.py \
  mode=sample_eval \
  data=text8 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  sampling.sampling_method=per_sentence \
  sampling.steps=100 \
  backbone=dit \
  eval.compute_generative_perplexity=True \
  eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/text8/2025.04.11/095554/checkpoints/best.ckpt 
  # eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/text8/2025.04.13/115827/checkpoints/best.ckpt \