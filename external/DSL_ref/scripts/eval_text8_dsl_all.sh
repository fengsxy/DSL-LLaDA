#!/bin/bash

# Array of sampling methods
METHODS=("per_token" "per_sentence")

# Array of step values
STEPS=(10 20 50 100 500 1000)

# Base checkpoint path
CHECKPOINT_PATH="/home/ywu380/DSL_mdml/outputs/text8/2025.04.11/095554/checkpoints/best.ckpt"

# Loop through all combinations
for method in "${METHODS[@]}"; do
  for step in "${STEPS[@]}"; do
    echo "Running with sampling.sampling_method=${method} and sampling.steps=${step}"
    
    # Create log filename based on parameters
    LOG_FILE="eval_text8_dsl_${method}_steps${step}.log"
    
    # Run the command
    python main.py \
      mode=ppl_eval \
      loader.batch_size=128 \
      loader.eval_batch_size=1024 \
      sampling.num_sample_batches=2 \
      sampling.sampling_method=${method} \
      sampling.steps=${step} \
      data=text8 \
      model.dim_embed=16 \
      model=small \
      backbone=dit \
      +wandb.offline=true \
      eval.checkpoint_path=${CHECKPOINT_PATH} \
      eval.compute_generative_perplexity=True > ${LOG_FILE}
    
    echo "Completed. Results saved to ${LOG_FILE}"
    echo "--------------------------------------"
  done
done

echo "All combinations completed!"