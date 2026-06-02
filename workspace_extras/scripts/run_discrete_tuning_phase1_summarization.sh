#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
MODEL="${MODEL:-original}"
NFE="${NFE:-16}"
BLOCKS="${BLOCKS:-16,32,64,128,256}"
EOS_MODES="${EOS_MODES:-default,eosInf,suppress}"
PYTHON="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/discrete_tuning_phase1_sum_${MODEL}_nfe${NFE}_${STAMP}}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
IFS=',' read -r -a BLOCK_ARR <<< "$BLOCKS"
IFS=',' read -r -a EOS_ARR <<< "$EOS_MODES"

declare -a CMDS=()

add_job() {
  local name="$1"
  local cmd="$2"
  CMDS+=("$name"$'\t'"$cmd")
}

for block in "${BLOCK_ARR[@]}"; do
  for eos in "${EOS_ARR[@]}"; do
    eos_args=()
    if [[ "$eos" == "eosInf" ]]; then
      eos_args=(--eos_inf)
    elif [[ "$eos" == "suppress" ]]; then
      eos_args=(--suppress_eos)
    fi
    suffix="b${block}_${eos}"
    add_job "xsum_${suffix}" \
      "$PYTHON -u dsl_llada/eval_summarization.py --dataset xsum --method remask --model_key $MODEL --nfe $NFE --data_file eval_data/xsum_1000.json --limit 200 --block_length $block ${eos_args[*]} --out_tag tune"
    add_job "cnn_${suffix}" \
      "$PYTHON -u dsl_llada/eval_summarization.py --dataset cnn_dailymail --method remask --model_key $MODEL --nfe $NFE --data_file eval_data/cnn_dailymail_1000.json --limit 200 --block_length $block ${eos_args[*]} --out_tag tune"
  done
done

printf "start=%s\n" "$(date -Is)" | tee "$LOG_DIR/progress.log"
printf "model=%s nfe=%s jobs=%s gpus=%s\n" "$MODEL" "$NFE" "${#CMDS[@]}" "${GPU_ARR[*]}" | tee -a "$LOG_DIR/progress.log"

run_worker() {
  local worker_id="$1"
  local gpu="$2"
  local total_gpus="$3"
  for idx in "${!CMDS[@]}"; do
    if (( idx % total_gpus != worker_id )); then
      continue
    fi
    local item="${CMDS[$idx]}"
    local name="${item%%$'\t'*}"
    local cmd="${item#*$'\t'}"
    local log="$LOG_DIR/${idx}_${name}.log"
    printf "[%s] worker=%s gpu=%s start job=%s\n" "$(date -Is)" "$worker_id" "$gpu" "$name" | tee -a "$LOG_DIR/progress.log"
    if CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$cmd --gpu 0" > "$log" 2>&1; then
      printf "[%s] worker=%s done job=%s\n" "$(date -Is)" "$worker_id" "$name" | tee -a "$LOG_DIR/progress.log"
    else
      printf "[%s] worker=%s FAILED job=%s log=%s\n" "$(date -Is)" "$worker_id" "$name" "$log" | tee -a "$LOG_DIR/progress.log"
      return 1
    fi
  done
}

pids=()
for i in "${!GPU_ARR[@]}"; do
  run_worker "$i" "${GPU_ARR[$i]}" "${#GPU_ARR[@]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

printf "done=%s status=%s\n" "$(date -Is)" "$status" | tee -a "$LOG_DIR/progress.log"
exit "$status"
