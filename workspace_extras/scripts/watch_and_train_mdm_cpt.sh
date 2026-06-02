#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/mdm_baseline_1k}"
LOG_DIR="${LOG_DIR:-logs/mdm_cpt_watch}"
mkdir -p "$LOG_DIR"

WATCH_LOG="$LOG_DIR/watch.log"
TRAIN_LOG="$LOG_DIR/train_mdm_cpt_1k.log"

echo "[$(date -Is)] watcher started; interval=${INTERVAL_SECONDS}s threshold=${MEM_THRESHOLD_MB}MB output=${OUTPUT_DIR}" >> "$WATCH_LOG"

while true; do
  if [[ -d "$OUTPUT_DIR/checkpoint-1000" ]]; then
    echo "[$(date -Is)] checkpoint already exists: $OUTPUT_DIR/checkpoint-1000" >> "$WATCH_LOG"
    exit 0
  fi

  mapfile -t used_mb < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "${#used_mb[@]}" -lt 8 ]]; then
    echo "[$(date -Is)] expected 8 GPUs, saw ${#used_mb[@]}: ${used_mb[*]}" >> "$WATCH_LOG"
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  all_free=1
  for value in "${used_mb[@]}"; do
    if (( value > MEM_THRESHOLD_MB )); then
      all_free=0
      break
    fi
  done

  echo "[$(date -Is)] gpu_mem_mb=${used_mb[*]} all_free=${all_free}" >> "$WATCH_LOG"

  if (( all_free == 1 )); then
    echo "[$(date -Is)] launching MDM-CPT training" >> "$WATCH_LOG"
    bash scripts/train_mdm_cpt_1k.sh "$OUTPUT_DIR" >> "$TRAIN_LOG" 2>&1
    status=$?
    echo "[$(date -Is)] training exited status=${status}" >> "$WATCH_LOG"
    exit "$status"
  fi

  sleep "$INTERVAL_SECONDS"
done
