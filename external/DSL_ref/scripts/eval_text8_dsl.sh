#!/bin/bash


python main.py \
  mode=ppl_eval \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  sampling.num_sample_batches=2 \
  sampling.sampling_method=per_token \
  sampling.steps=10 \
  data=text8 \
  model.dim_embed=16 \
  model=small \
  backbone=dit \
  +wandb.offline=true \
  eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/text8/2025.05.14_best_ckpt/192015/checkpoints/best.ckpt \
  eval.compute_generative_perplexity=True  > eval_text8_dsl_per_token.log
  # eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/text8/2025.04.11/095554/checkpoints/best.ckpt \
  # eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/text8/2025.05.11/222900/checkpoints/best.ckpt \
  # eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/text8/2025.04.13/115827/checkpoints/best.ckpt \
  # model.length=1024 \