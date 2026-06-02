#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1 # 使用2个GPU
export NCCL_TIMEOUT=3600000  # 设置为60分钟

# 不再override loader.batch_size 和 loader.eval_batch_size
python -u -m main \
  model=small \
  data=text8 \
  wandb.name=dsl-text8-small \
  model.dim_embed=16 \
  training.t_max=11 \
  eval.compute_generative_perplexity=False \
  sampling.sampling_method=per_sentence \
  sampling.steps=10 \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  optim.lr=3e-4 \
  loader.num_workers=0 \
  training.reconst_weight=1.0 \
  training.val_mse_plot=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 > train_text8_dsl.log
  # checkpointing.resume_ckpt_path=/home/ywu380/DSL_mdml/outputs/text8/2025.05.14/173559/checkpoints/last.ckpt \
  # optim.weight_decay=0.03 \
  # optim.weight_decay=0.0001 \