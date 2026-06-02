#!/usr/bin/env bash
# Find all Stage-2 configs where merged file is missing or incomplete,
# then launch only the missing shards (1-GPU per shard) to fill in.
# Safe to run after sweep completes.
set -euo pipefail
cd /home/ubuntu/efs/RMDM

SUM_DIR="eval_results/summarization"
PY=/home/ubuntu/efs/RMDM/.venv/bin/python

declare -A TAG_ARGS=(
  [b1_sde]="--method sde --model_key b1"
  [original_remask]="--method remask --model_key original --block_length 256"
  [original_remask_eosInf_b32]="--method remask --model_key original --eos_inf --block_length 32"
)

needs=()
for d in xsum cnn_dailymail pubmed arxiv billsum; do
  for tag in b1_sde original_remask original_remask_eosInf_b32; do
    for nfe in 8 16 32 64; do
      merged="$SUM_DIR/${d}_${tag}_nfe${nfe}.json"
      ok=0
      if [[ -f "$merged" ]]; then
        N=$($PY -c "import json;print(json.load(open('$merged')).get('n_samples_total',0))" 2>/dev/null || echo 0)
        [[ "$N" -ge 1000 ]] && ok=1
      fi
      if [[ "$ok" -eq 0 ]]; then
        # which shards missing?
        missing_shards=""
        for i in 0 1 2 3 4 5 6 7; do
          shardf="$SUM_DIR/${d}_${tag}_nfe${nfe}_shard${i}of8.json"
          [[ ! -f "$shardf" ]] && missing_shards+="$i "
        done
        needs+=("${d}|${tag}|${nfe}|${missing_shards}")
      fi
    done
  done
done

echo "Missing-shard tasks: ${#needs[@]}"
for item in "${needs[@]}"; do
  echo "  $item"
done

# Launch up to 8 shards in parallel (round-robin GPU assignment)
PIDS=()
GPU_IDX=0
for item in "${needs[@]}"; do
  IFS='|' read -r d tag nfe missing <<< "$item"
  [[ -z "$missing" ]] && continue
  ARGS=${TAG_ARGS[$tag]}
  for i in $missing; do
    [[ -z "$i" ]] && continue
    GPU=$((GPU_IDX % 8))
    GPU_IDX=$((GPU_IDX + 1))
    LOG="logs/summarization/_fill_${d}_${tag}_nfe${nfe}_shard${i}.log"
    $PY dsl_llada/eval_summarization.py \
      --dataset "$d" --data_file "eval_data/${d}_1000.json" \
      --gpu "$GPU" --shard_id "$i" --num_shards 8 --seed 42 \
      --nfe "$nfe" $ARGS >"$LOG" 2>&1 &
    PIDS+=($!)
    # Limit concurrency to 8
    if [[ ${#PIDS[@]} -ge 8 ]]; then
      wait "${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
    fi
  done
done
wait "${PIDS[@]}"
echo ""
echo "All missing shards completed. Now merging..."
$PY dsl_llada/merge_summarization_shards.py --auto
echo "Done."
