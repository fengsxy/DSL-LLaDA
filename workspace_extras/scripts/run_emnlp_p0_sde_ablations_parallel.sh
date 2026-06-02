#!/usr/bin/env bash
# Parallel low-cost P0 ablations for EMNLP revision.
#
# Uses a fixed pool of GPUs. Each GPU runs a sequential worker over a
# round-robin slice of the task list; existing completed outputs are skipped.
set -euo pipefail

cd /home/ubuntu/efs/RMDM
source .venv/bin/activate

MODEL_KEY="${MODEL_KEY:-b1}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
PROMPT_LIMIT="${PROMPT_LIMIT:-100}"
SUM_LIMIT="${SUM_LIMIT:-200}"
LOG_ROOT="logs/emnlp_p0_sde_ablations_parallel_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_WORKERS="${#GPUS[@]}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_ROOT/progress.log"
}

gen_out() {
  local tag="$1"
  echo "eval_results/sde_gen_formal/${tag}_${MODEL_KEY}_sde_nfe64_gen256.json"
}

xsum_out() {
  local tag="$1"
  echo "eval_results/summarization/xsum_${MODEL_KEY}_sde_${tag}_nfe64.json"
}

has_gen_metrics() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  python - "$file" <<'PY' 2>/dev/null
import json, sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if "metrics" in d else 1)
PY
}

run_gen_task() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local out
  out="$(gen_out "$tag")"
  if has_gen_metrics "$out"; then
    echo "[skip][gpu $gpu] gen $tag"
    return 0
  fi
  echo "[run ][gpu $gpu] gen $tag $*"
  CUDA_VISIBLE_DEVICES="$gpu" python dsl_llada/eval_sde_gen_formal.py \
    --model_key "$MODEL_KEY" --method sde --nfe 64 \
    --tag "$tag" --prompts eval_data/sde_prompts_200.json \
    --gen_length 256 --max_prompts "$PROMPT_LIMIT" --gpu 0 "$@" \
    > "$LOG_ROOT/gen_${tag}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" python dsl_llada/eval_sde_gen_formal.py \
    --metrics "$out" --gpu 0 \
    > "$LOG_ROOT/metrics_${tag}.log" 2>&1
}

run_xsum_task() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local out
  out="$(xsum_out "$tag")"
  if [[ -f "$out" ]]; then
    echo "[skip][gpu $gpu] xsum $tag"
    return 0
  fi
  echo "[run ][gpu $gpu] xsum $tag $*"
  CUDA_VISIBLE_DEVICES="$gpu" python dsl_llada/eval_summarization.py \
    --dataset xsum --method sde --model_key "$MODEL_KEY" --nfe 64 \
    --limit "$SUM_LIMIT" --out_tag "$tag" --gpu 0 "$@" \
    > "$LOG_ROOT/xsum_${tag}.log" 2>&1
}

run_task() {
  local gpu="$1"
  local task="$2"
  IFS='|' read -r kind tag args <<< "$task"
  # shellcheck disable=SC2206
  local argv=( $args )
  if [[ "$kind" == "gen" ]]; then
    run_gen_task "$gpu" "$tag" "${argv[@]}"
  else
    run_xsum_task "$gpu" "$tag" "${argv[@]}"
  fi
}

TASKS=()

for topk in 128 256 512 1024; do
  TASKS+=("gen|supp_topk${topk}|--sde_top_k ${topk}")
  TASKS+=("xsum|supp_topk${topk}|--sde_top_k ${topk}")
done

for schedule in sensitive uniform; do
  for noise in 0.0 0.005 0.01; do
    noise_tag="${noise//./p}"
    tag="supp_${schedule}_ns${noise_tag}"
    TASKS+=("gen|${tag}|--sde_schedule ${schedule} --sde_noise_scale ${noise}")
    TASKS+=("xsum|${tag}|--sde_schedule ${schedule} --sde_noise_scale ${noise}")
  done
done

for beta in 1.5 2.0 2.5; do
  beta_tag="${beta//./p}"
  tag="supp_beta${beta_tag}"
  TASKS+=("gen|${tag}|--sde_beta_infer ${beta}")
  TASKS+=("xsum|${tag}|--sde_beta_infer ${beta}")
done

worker() {
  local worker_id="$1"
  local gpu="${GPUS[$worker_id]}"
  local i
  for i in "${!TASKS[@]}"; do
    if (( i % N_WORKERS == worker_id )); then
      run_task "$gpu" "${TASKS[$i]}"
    fi
  done
}

log "start MODEL_KEY=$MODEL_KEY GPUs=$GPU_LIST PROMPT_LIMIT=$PROMPT_LIMIT SUM_LIMIT=$SUM_LIMIT tasks=${#TASKS[@]}"

pids=()
for wid in "${!GPUS[@]}"; do
  worker "$wid" > "$LOG_ROOT/worker_${wid}_gpu${GPUS[$wid]}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail + 1))
  fi
done

if [[ "$fail" -ne 0 ]]; then
  log "failed workers=$fail"
  exit 1
fi

log "done; logs in $LOG_ROOT"
