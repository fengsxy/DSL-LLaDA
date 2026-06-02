#!/usr/bin/env bash
# Run full 1000-sample summarization eval across all datasets, methods, NFEs.
# Each config consumes all 8 GPUs (sharded), configs run SEQUENTIALLY.
#
# Usage:
#   bash dsl_llada/scripts/eval_sum_full_sweep.sh            # all configs
#   bash dsl_llada/scripts/eval_sum_full_sweep.sh xsum       # only xsum
#   SKIP_SDE=1 bash dsl_llada/scripts/eval_sum_full_sweep.sh # skip SDE configs (for resume)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASETS_FILTER="${1:-all}"
ALL_DATASETS=(xsum cnn_dailymail pubmed arxiv billsum)
if [[ "$DATASETS_FILTER" == "all" ]]; then
  DATASETS=("${ALL_DATASETS[@]}")
else
  DATASETS=("$DATASETS_FILTER")
fi

if [[ -n "${NFES_OVERRIDE:-}" ]]; then
  read -ra NFES <<< "$NFES_OVERRIDE"
else
  NFES=(8 16 32 64)
fi
LAUNCHER="bash dsl_llada/scripts/eval_sum_shard_launch.sh"
SUM_DIR="eval_results/summarization"
PROG_LOG="logs/summarization/_sweep_progress.log"
mkdir -p logs/summarization
echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep start  datasets=${DATASETS_FILTER}  NFES=${NFES[*]}  skip_sde=${SKIP_SDE:-0}  skip_llada=${SKIP_LLADA:-0}  skip_block=${SKIP_BLOCK:-0}" | tee -a "$PROG_LOG"

is_done() {
  local base="$1"
  local f="$SUM_DIR/${base}.json"
  [[ -f "$f" ]] || return 1
  # Verify file is a real 1000-sample run with valid generations (>10 valid samples).
  ${PY:-python} - "$f" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("n_samples_total", 0) >= 1000, "not 1000 samples"
assert d.get("valid", 0) >= 10, f"only {d.get('valid',0)} valid samples"
PY
}

run_config() {
  local dataset="$1"; shift
  local tag="$1"; shift
  local nfe="$1"; shift
  local base="${dataset}_${tag}_nfe${nfe}"

  if is_done "$base"; then
    echo "[skip] $base already exists" | tee -a "$PROG_LOG"
    return 0
  fi
  echo "[$(date '+%H:%M:%S')] START $base  args: $*" | tee -a "$PROG_LOG"
  local t0=$(date +%s)
  if ! $LAUNCHER "$dataset" 1000 "$@" --nfe "$nfe"; then
    echo "[$(date '+%H:%M:%S')] FAIL $base" | tee -a "$PROG_LOG"
    return 1
  fi
  ${PY:-python} dsl_llada/eval/merge_summarization_shards.py \
    --dataset "$dataset" --method_tag "$tag" --nfe "$nfe" || echo "[warn] merge failed for $base"
  local t1=$(date +%s)
  local dt=$(( t1 - t0 ))
  echo "[$(date '+%H:%M:%S')] DONE  $base  elapsed=${dt}s" | tee -a "$PROG_LOG"
}

for dataset in "${DATASETS[@]}"; do
  for nfe in "${NFES[@]}"; do
    if [[ "${SKIP_SDE:-0}" != "1" ]]; then
      run_config "$dataset" "b1_sde" "$nfe" --method sde --model_key b1 || true
    fi
    if [[ "${SKIP_LLADA:-0}" != "1" ]]; then
      run_config "$dataset" "original_remask" "$nfe" --method remask --model_key original --block_length 256 || true
    fi
    if [[ "${SKIP_BLOCK:-0}" != "1" ]]; then
      run_config "$dataset" "original_remask_eosInf_b32" "$nfe" \
        --method remask --model_key original --eos_inf --block_length 32 || true
    fi
  done
done

echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] Full sweep done"
echo "=============================================="
ls -la $SUM_DIR/*_1000.json $SUM_DIR/*nfe*.json 2>/dev/null | grep -v shard | tail -20
