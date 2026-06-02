#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-original,mdm_cpt,xdlm,b1,sm_b2,sem_b05_3k}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
PYTHON="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/realistic_correction_${STAMP}}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"

echo "models=${MODEL_ARR[*]}" | tee "$LOG_DIR/progress.log"
echo "gpus=${GPU_ARR[*]}" | tee -a "$LOG_DIR/progress.log"
echo "start=$(date -Is)" | tee -a "$LOG_DIR/progress.log"

pids=()
for i in "${!MODEL_ARR[@]}"; do
  model="${MODEL_ARR[$i]}"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  log="$LOG_DIR/${model}.log"
  out="eval_results/${model}/realistic_correction.json"
  echo "launch model=${model} gpu=${gpu} out=${out}" | tee -a "$LOG_DIR/progress.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u dsl_llada/eval_realistic_correction.py \
    --model_key "$model" \
    --gpu 0 \
    --out "$out" \
    > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "done=$(date -Is) status=${status}" | tee -a "$LOG_DIR/progress.log"
exit "$status"
