#!/bin/bash
# ============================================================
# Original LLaDA baselines: 3 settings × NFE curve
# ============================================================
# Setting 1: Original (default) — already done
# Setting 2: Original + logits_eos_inf (EOS patch)
# Setting 3: Original + b32 + suppress_eos (semi-AR + noEOS)
#
# Usage:
#   bash dsl_llada/eval_original_baselines.sh [gpu]

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SCRIPT="dsl_llada/eval_sde_gen_formal.py"
PROMPTS="eval_data/sde_prompts_200.json"
LOGDIR="eval_results/sde_gen_formal/logs"
mkdir -p "$LOGDIR"

GPU="${1:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# Setting 2: Original + logits_eos_inf
# ============================================================
log "========== Original + eos_inf =========="
for nfe in 8 16 32 64 128; do
    log "START: original eos_inf nfe=$nfe"
    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
        --model_key original --method remask --nfe $nfe \
        --tag nfe_curve --gpu 0 --eos_inf \
        --prompts "$PROMPTS" \
        2>&1 | tee -a "$LOGDIR/original_eosInf_${TIMESTAMP}.log"
    log "DONE: original eos_inf nfe=$nfe"
done

# ============================================================
# Setting 3: Original + b32 + suppress_eos
# ============================================================
log "========== Original + b32 + noEOS =========="
for nfe in 8 16 32 64 128; do
    log "START: original b32+noEOS nfe=$nfe"
    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
        --model_key original --method remask --nfe $nfe \
        --tag nfe_curve --gpu 0 --block_length 32 --suppress_eos \
        --prompts "$PROMPTS" \
        2>&1 | tee -a "$LOGDIR/original_b32noEOS_${TIMESTAMP}.log"
    log "DONE: original b32+noEOS nfe=$nfe"
done

log "========== ALL DONE =========="
