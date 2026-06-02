#!/usr/bin/env bash
# Low-cost P0 ablations for the EMNLP revision plan.
#
# Runs paper-b1 SDE implementation checks:
#   E1: top-M sensitivity
#   E2: SNR schedule / stochastic noise ablation
#   E3: beta-inference robustness
#
# Defaults use one GPU and small subsets so the sweep can be repeated quickly.
set -euo pipefail

cd /home/ubuntu/efs/RMDM
source .venv/bin/activate

MODEL_KEY="${MODEL_KEY:-b1}"
GPU="${GPU:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-100}"
SUM_LIMIT="${SUM_LIMIT:-200}"
LOG_ROOT="logs/emnlp_p0_sde_ablations_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_ROOT/progress.log"
}

metric_gen() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    log "missing generation file for metrics: $file"
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python dsl_llada/eval_sde_gen_formal.py \
    --metrics "$file" --gpu 0 \
    > "$LOG_ROOT/metrics_$(basename "$file" .json).log" 2>&1
}

run_gen_sde() {
  local tag="$1"
  shift
  log "gen: tag=$tag args=$*"
  CUDA_VISIBLE_DEVICES="$GPU" python dsl_llada/eval_sde_gen_formal.py \
    --model_key "$MODEL_KEY" --method sde --nfe 64 \
    --tag "$tag" --prompts eval_data/sde_prompts_200.json \
    --gen_length 256 --max_prompts "$PROMPT_LIMIT" --gpu 0 "$@" \
    > "$LOG_ROOT/gen_${tag}.log" 2>&1
  metric_gen "eval_results/sde_gen_formal/${tag}_${MODEL_KEY}_sde_nfe64_gen256.json"
}

run_xsum_sde() {
  local tag="$1"
  shift
  log "xsum: tag=$tag args=$*"
  CUDA_VISIBLE_DEVICES="$GPU" python dsl_llada/eval_summarization.py \
    --dataset xsum --method sde --model_key "$MODEL_KEY" --nfe 64 \
    --limit "$SUM_LIMIT" --out_tag "$tag" --gpu 0 "$@" \
    > "$LOG_ROOT/xsum_${tag}.log" 2>&1
}

log "start MODEL_KEY=$MODEL_KEY GPU=$GPU PROMPT_LIMIT=$PROMPT_LIMIT SUM_LIMIT=$SUM_LIMIT"

log "E1: top-M sensitivity"
for topk in 128 256 512 1024; do
  tag="supp_topk${topk}"
  run_gen_sde "$tag" --sde_top_k "$topk"
  run_xsum_sde "$tag" --sde_top_k "$topk"
done

log "E2: schedule/noise ablation"
for schedule in sensitive uniform; do
  for noise in 0.0 0.005 0.01; do
    noise_tag="${noise//./p}"
    tag="supp_${schedule}_ns${noise_tag}"
    run_gen_sde "$tag" --sde_schedule "$schedule" --sde_noise_scale "$noise"
    run_xsum_sde "$tag" --sde_noise_scale "$noise"
  done
done

log "E3: beta-inference robustness"
for beta in 1.5 2.0 2.5; do
  beta_tag="${beta//./p}"
  tag="supp_beta${beta_tag}"
  run_gen_sde "$tag" --sde_beta_infer "$beta"
  run_xsum_sde "$tag" --sde_beta_infer "$beta"
done

log "done; logs in $LOG_ROOT"
