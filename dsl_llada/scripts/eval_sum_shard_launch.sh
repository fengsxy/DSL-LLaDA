#!/usr/bin/env bash
# Launch sharded summarization eval across 8 GPUs.
# Usage:
#   bash dsl_llada/scripts/eval_sum_shard_launch.sh <dataset> <n> <method_args...>
# Example (SDE NFE=64 on XSum 1000):
#   bash dsl_llada/scripts/eval_sum_shard_launch.sh xsum 1000 \
#       --method sde --model_key b1 --nfe 64
# Example (LLaDA+EOS+Block NFE=64):
#   bash dsl_llada/scripts/eval_sum_shard_launch.sh xsum 1000 \
#       --method remask --model_key original --nfe 64 --eos_inf --block_length 32
set -euo pipefail

DATASET=$1; shift
N=$1; shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_FILE="eval_data/${DATASET}_${N}.json"
if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: $DATA_FILE not found" >&2; exit 1
fi

LOG_DIR="logs/summarization"
mkdir -p "$LOG_DIR"

# Sanity: kill any leftover zombie python processes from previous config
# that may still hold CUDA memory (caused OOM on subsequent configs).
LEFTOVER=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
           | awk -F', ' '{gsub(/MiB/,"",$2); if ($2+0 > 10000) print $1}' | sort -u)
if [[ -n "$LEFTOVER" ]]; then
  # Only kill pids that are orphans of eval_summarization.py and not children of this shell
  for pid in $LEFTOVER; do
    if ps -p "$pid" -o cmd= 2>/dev/null | grep -q "eval_summarization"; then
      # Check it's not a currently-running shard child of this process tree
      if ! ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' | grep -qE "^$$\b"; then
        kill -9 "$pid" 2>/dev/null && echo "  pre-launch: killed orphan pid=$pid"
      fi
    fi
  done
  sleep 2
fi

TAG=$(echo "${EXTRA_ARGS[@]}" | tr ' ' '_' | tr -cd 'A-Za-z0-9._=-')
echo "Launching 8 shards for $DATASET ($N samples): ${EXTRA_ARGS[*]}"
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  LOG="$LOG_DIR/${DATASET}_${N}_${TAG}_shard${i}.log"
  ${PY:-python} dsl_llada/eval/eval_summarization.py \
    --dataset "$DATASET" --data_file "$DATA_FILE" \
    --gpu "$i" --shard_id "$i" --num_shards 8 \
    --seed 42 "${EXTRA_ARGS[@]}" >"$LOG" 2>&1 &
  PIDS+=($!)
  echo "  shard $i -> GPU $i, pid=${PIDS[-1]}, log=$LOG"
done

echo "Waiting for all 8 shards..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "  pid $pid FAILED"
    FAIL=$((FAIL+1))
  fi
done
echo "Shard run complete: $((8 - FAIL))/8 succeeded"
exit $FAIL
