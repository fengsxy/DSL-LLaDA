#!/bin/bash

python main.py \
  mode=sample_eval \
  data=lm1b \
  model.length=128 \
  model.dim_embed=128 \
  loader.eval_batch_size=10 \
  sampling.num_sample_batches=1 \
  backbone=dit \
  sampling.sampling_method=uniform_t \
  eval.compute_generative_perplexity=True \
  eval.checkpoint_path=/home/ywu380/DSL_mdml/outputs/lm1b/2025.05.05/150244/checkpoints/best.ckpt \
  eval.compute_generative_perplexity=True > sample_lm1b_dsl.log