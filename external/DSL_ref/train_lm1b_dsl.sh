#!/bin/bash

export NCCL_TIMEOUT=3600000  # 设置为60分钟

python -u -m main \
  model=small \
  data=lm1b \
  wandb.name=dsl-lm1b-small \
  eval.compute_generative_perplexity=True \
  sampling.sampling_method=uniform_t \
  model.dim_embed=128 \
  model.length=128 \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=16 \
  optim.lr=3e-4 \
  loader.num_workers=0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 > train_lm1b_dsl.log
  # checkpointing.resume_ckpt_path=/home/ywu380/DSL_mdml/outputs/lm1b/2025.05.05/005043/checkpoints/last.ckpt \
  # training.val_mse_plot=False \
  # optim.weight_decay=0.0001 \