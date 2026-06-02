#!/usr/bin/env bash
# Paper-aligned evaluation sweep for unit-uniform + trainable embedding.
#
# This is a thin runner over existing eval scripts. It does not change model
# code or sampling logic.
set -euo pipefail

cd /home/ubuntu/efs/RMDM
source .venv/bin/activate

MODEL_KEY="${MODEL_KEY:-uu_trainable_1k}"
GPU_UNIFIED="${GPU_UNIFIED:-0}"
GPU_GEN="${GPU_GEN:-1}"
GPU_BLOCK="${GPU_BLOCK:-2}"
SUM_GPUS="${SUM_GPUS:-0,4,5,6,7}"
LOG_ROOT="logs/unit_uniform/paper_eval_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_ROOT/progress.log"
}

run_metrics() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    log "missing metrics input: $file"
    return 0
  fi
  if python - "$file" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
raise SystemExit(0 if "metrics" in d else 1)
PY
  then
    log "metrics exists: $file"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU_GEN" python dsl_llada/eval_sde_gen_formal.py \
    --metrics "$file" --gpu 0 \
    > "$LOG_ROOT/metrics_$(basename "$file" .json).log" 2>&1
}

run_summary_shards() {
  local dataset="$1"
  local nfe="$2"
  local data_file="eval_data/${dataset}_1000.json"
  local tag="${MODEL_KEY}_sde"

  IFS=',' read -ra gpus <<< "$SUM_GPUS"
  local n_shards="${#gpus[@]}"
  local pids=()

  log "summary launch: dataset=$dataset nfe=$nfe shards=$n_shards gpus=$SUM_GPUS"
  for shard in "${!gpus[@]}"; do
    local gpu="${gpus[$shard]}"
    local shard_log="$LOG_ROOT/sum_${dataset}_sde_nfe${nfe}_shard${shard}.log"
    python dsl_llada/eval_summarization.py \
      --dataset "$dataset" --data_file "$data_file" \
      --gpu "$gpu" --shard_id "$shard" --num_shards "$n_shards" \
      --seed 42 --method sde --model_key "$MODEL_KEY" --nfe "$nfe" \
      > "$shard_log" 2>&1 &
    pids+=($!)
  done

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=$((fail + 1))
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    log "summary failed: dataset=$dataset nfe=$nfe failed_shards=$fail"
    return 1
  fi

  python dsl_llada/merge_summarization_shards.py \
    --dataset "$dataset" --method_tag "$tag" --nfe "$nfe" \
    > "$LOG_ROOT/sum_${dataset}_sde_nfe${nfe}_merge.log" 2>&1
}

log "start MODEL_KEY=$MODEL_KEY"

log "unified eval: T1 correction, T2 generation, T3 mask/ECE"
python dsl_llada/eval_unified.py \
  --model_key "$MODEL_KEY" --tests t1,t2,t3 --skip_existing --gpu "$GPU_UNIFIED" \
  > "$LOG_ROOT/unified_t1_t2_t3.log" 2>&1

log "context robustness: mask=30%, corrupt=0/10/20/30/50"
python dsl_llada/eval_context_robustness.py \
  --model_key "$MODEL_KEY" --gpu "$GPU_UNIFIED" \
  --mask_rates 0.3 --corruption_rates 0,10,20,30,50 \
  > "$LOG_ROOT/context_robustness.log" 2>&1

log "open-ended generation NFE sweep: prompted 200, gen=256"
for nfe in 8 16 32 64 128; do
  for method in sde remask; do
    CUDA_VISIBLE_DEVICES="$GPU_GEN" python dsl_llada/eval_sde_gen_formal.py \
      --model_key "$MODEL_KEY" --method "$method" --nfe "$nfe" \
      --tag nfe_curve --prompts eval_data/sde_prompts_200.json \
      --gen_length 256 --gpu 0 \
      > "$LOG_ROOT/nfe_${method}_${nfe}.log" 2>&1
    run_metrics "eval_results/sde_gen_formal/nfe_curve_${MODEL_KEY}_${method}_nfe${nfe}_gen256.json"
  done
done

log "prefix continuation: 200 WikiText prefixes, NFE=64, gen=256"
for method in sde remask; do
  CUDA_VISIBLE_DEVICES="$GPU_GEN" python dsl_llada/eval_sde_gen_formal.py \
    --model_key "$MODEL_KEY" --method "$method" --nfe 64 \
    --tag prefix --prefixes eval_data/wikitext_prefix_200.json \
    --gen_length 256 --gpu 0 \
    > "$LOG_ROOT/prefix_${method}.log" 2>&1
  run_metrics "eval_results/sde_gen_formal/prefix_${MODEL_KEY}_${method}_nfe64_gen256.json"
done

log "long-form generation: gen=512/1024, 100 prompts, NFE=64"
for gen_len in 512 1024; do
  for method in sde remask; do
    CUDA_VISIBLE_DEVICES="$GPU_GEN" python dsl_llada/eval_sde_gen_formal.py \
      --model_key "$MODEL_KEY" --method "$method" --nfe 64 \
      --tag longform --prompts eval_data/sde_prompts_200.json \
      --gen_length "$gen_len" --max_prompts 100 --gpu 0 \
      > "$LOG_ROOT/longform_${method}_${gen_len}.log" 2>&1
    run_metrics "eval_results/sde_gen_formal/longform_${MODEL_KEY}_${method}_nfe64_gen${gen_len}.json"
  done
done

log "summarization sweep: XSum/CNN-DM/PubMed/arXiv, 1000 samples, NFE=8/16/32/64"
for dataset in xsum cnn_dailymail pubmed arxiv; do
  for nfe in 8 16 32 64; do
    base="eval_results/summarization/${dataset}_${MODEL_KEY}_sde_nfe${nfe}.json"
    if [[ -f "$base" ]]; then
      log "skip summarization exists: $base"
      continue
    fi
    run_summary_shards "$dataset" "$nfe"
  done
done

log "summarization BERTScore: all merged ${MODEL_KEY}_sde files"
mapfile -t sum_files < <(
  find eval_results/summarization -maxdepth 1 -type f \
    -name "*_${MODEL_KEY}_sde_nfe*.json" ! -name "*shard*" | sort
)
if [[ "${#sum_files[@]}" -gt 0 ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_UNIFIED" python dsl_llada/compute_bertscore_summarization.py \
    --files "${sum_files[@]}" --gpu 0 \
    > "$LOG_ROOT/summarization_bertscore.log" 2>&1
else
  log "no merged summarization files found for BERTScore"
fi

log "GSM8K Block-SDE probe: references + best block config on 100 problems"
python dsl_llada/eval_block_sde.py \
  --ckpt "$(python - "$MODEL_KEY" <<'PY'
import json, os, sys
root = "/home/ubuntu/efs/RMDM"
reg = json.load(open(os.path.join(root, "eval_results", "registry.json")))
print(os.path.join(root, reg[sys.argv[1]]["path"]))
PY
)" \
  --gpu "$GPU_BLOCK" --n_problems 100 --block_sizes 8 --steps_per_block 4 \
  > "$LOG_ROOT/block_sde_b8s4.log" 2>&1

log "done; logs in $LOG_ROOT"
