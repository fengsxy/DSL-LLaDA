#!/usr/bin/env bash
# Run XDLM summarization with the official clean-token replacement sampler.
#
# Defaults:
#   - 1000 samples for xsum/cnn_dailymail/pubmed/arxiv/billsum
#   - 100 samples for aeslc, because eval_data/aeslc_1000.json is not present
#   - NFE grid: 8 16 32 64
#   - GPU shards: 0 2 4 5 6
#
# Override examples:
#   GPUS="0 2 4 5 6" DATASETS="xsum cnn_dailymail" NFES="8 16" \
#     bash dsl_llada/run_xdlm_summarization_sweep.sh
set -euo pipefail

ROOT="/home/ylong030/dsllada/AWS_private_2"
cd "$ROOT"

read -ra GPUS_ARR <<< "${GPUS:-0 2 4 5 6}"
read -ra DATASETS_ARR <<< "${DATASETS:-xsum cnn_dailymail pubmed arxiv billsum aeslc}"
read -ra NFES_ARR <<< "${NFES:-8 16 32 64}"

NUM_SHARDS="${#GPUS_ARR[@]}"
SUM_DIR="eval_results/summarization"
LOG_DIR="logs/summarization_xdlm"
mkdir -p "$SUM_DIR" "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

sample_count_for_dataset() {
  local dataset="$1"
  if [[ -f "eval_data/${dataset}_1000.json" ]]; then
    echo "1000"
  elif [[ -f "eval_data/${dataset}_100.json" ]]; then
    echo "100"
  else
    echo "missing"
  fi
}

is_done() {
  local file="$1"
  local expected="$2"
  [[ -f "$file" ]] || return 1
  python - "$file" "$expected" <<'PY' >/dev/null 2>&1
import json, sys
d = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
assert d.get("n_samples_total") == expected
assert d.get("n_samples_here") == expected
assert d.get("valid", 0) > 0
PY
}

echo "[$(timestamp)] XDLM summarization sweep start"
echo "  GPUs: ${GPUS_ARR[*]}"
echo "  datasets: ${DATASETS_ARR[*]}"
echo "  NFEs: ${NFES_ARR[*]}"
echo "  shards/config: $NUM_SHARDS"

for dataset in "${DATASETS_ARR[@]}"; do
  n_samples="$(sample_count_for_dataset "$dataset")"
  if [[ "$n_samples" == "missing" ]]; then
    echo "[$(timestamp)] SKIP $dataset: no eval_data/${dataset}_{1000,100}.json"
    continue
  fi

  data_file="eval_data/${dataset}_${n_samples}.json"
  out_tag="basePrompt_data${n_samples}"
  method_tag="xdlm_xdm_${out_tag}"

  for nfe in "${NFES_ARR[@]}"; do
    merged="${SUM_DIR}/${dataset}_${method_tag}_nfe${nfe}.json"
    if is_done "$merged" "$n_samples"; then
      echo "[$(timestamp)] SKIP ${dataset} nfe=${nfe}: $merged exists"
      continue
    fi

    echo "[$(timestamp)] START ${dataset} n=${n_samples} nfe=${nfe}"
    pids=()
    for shard_id in "${!GPUS_ARR[@]}"; do
      gpu="${GPUS_ARR[$shard_id]}"
      log="${LOG_DIR}/${dataset}_${method_tag}_nfe${nfe}_shard${shard_id}of${NUM_SHARDS}.log"
      python dsl_llada/eval_summarization.py \
        --dataset "$dataset" \
        --data_file "$data_file" \
        --method xdm \
        --model_key xdlm \
        --gpu "$gpu" \
        --nfe "$nfe" \
        --block_length 256 \
        --xdm_k1 0.1 \
        --prompt_format plain \
        --seed 42 \
        --shard_id "$shard_id" \
        --num_shards "$NUM_SHARDS" \
        --out_tag "$out_tag" >"$log" 2>&1 &
      pids+=("$!")
      echo "  shard ${shard_id}/${NUM_SHARDS} gpu=${gpu} pid=${pids[-1]} log=${log}"
    done

    failures=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failures=$((failures + 1))
      fi
    done

    if [[ "$failures" -ne 0 ]]; then
      echo "[$(timestamp)] FAIL ${dataset} nfe=${nfe}: ${failures}/${NUM_SHARDS} shards failed"
      continue
    fi

    python dsl_llada/merge_summarization_shards.py \
      --dataset "$dataset" \
      --method_tag "$method_tag" \
      --nfe "$nfe"
    echo "[$(timestamp)] DONE ${dataset} nfe=${nfe}"
  done
done

echo "[$(timestamp)] XDLM summarization sweep done"
