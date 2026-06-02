#!/bin/bash
# ============================================================
# Formal SDE vs Remask Generation Quality Evaluation
# ============================================================
# Usage:
#   bash dsl_llada/eval_sde_gen_formal.sh part1     # Standard benchmarks
#   bash dsl_llada/eval_sde_gen_formal.sh part2a    # NFE curve
#   bash dsl_llada/eval_sde_gen_formal.sh part2b    # Long-form gen
#   bash dsl_llada/eval_sde_gen_formal.sh part2c    # Diversity (Self-BLEU)
#   bash dsl_llada/eval_sde_gen_formal.sh part2d    # Speed-quality
#   bash dsl_llada/eval_sde_gen_formal.sh metrics   # Compute all metrics
#   bash dsl_llada/eval_sde_gen_formal.sh all       # Everything
#
# Logs saved to eval_results/sde_gen_formal/logs/

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SCRIPT="dsl_llada/eval_sde_gen_formal.py"
PROMPTS="eval_data/sde_prompts_200.json"
PREFIXES="eval_data/wikitext_prefix_200.json"
LOGDIR="eval_results/sde_gen_formal/logs"
mkdir -p "$LOGDIR"

GPU0=6
GPU1=7

PART="${1:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_gen() {
    local model=$1 method=$2 nfe=$3 tag=$4 gpu=$5
    shift 5
    local extra="$*"
    local logfile="$LOGDIR/${tag}_${model}_${method}_nfe${nfe}_${TIMESTAMP}.log"
    log "START: model=$model method=$method nfe=$nfe tag=$tag gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
        --model_key "$model" --method "$method" --nfe "$nfe" \
        --tag "$tag" --gpu 0 $extra \
        2>&1 | tee "$logfile"
    log "DONE:  model=$model method=$method nfe=$nfe tag=$tag"
}

# ============================================================
# Part 1: Standard benchmarks (5 models × remask + 2 SDE, NFE=64)
# ============================================================
run_part1() {
    log "========== Part 1a: Prompted Generation (200 prompts, NFE=64) =========="

    # Remask: all 5 models (serial on GPU1)
    for model in original mdm_cpt xdlm b1 sm_b2; do
        run_gen $model remask 64 prompted $GPU1 --prompts "$PROMPTS"
    done

    # SDE: b1 and sm_b2 (serial on GPU0)
    for model in b1 sm_b2; do
        run_gen $model sde 64 prompted $GPU0 --prompts "$PROMPTS"
    done

    log "========== Part 1b: Prefix Continuation (200 prefixes, NFE=64) =========="

    for model in original mdm_cpt xdlm b1 sm_b2; do
        run_gen $model remask 64 prefix $GPU1 --prefixes "$PREFIXES"
    done

    for model in b1 sm_b2; do
        run_gen $model sde 64 prefix $GPU0 --prefixes "$PREFIXES"
    done
}

# ============================================================
# Part 2a: NFE-Efficiency Curve (Original remask + b1 remask + b1 SDE)
# ============================================================
run_part2a() {
    log "========== Part 2a: NFE-Efficiency Curve =========="

    for nfe in 4 8 16 32 64 128; do
        # Original remask on GPU1
        run_gen original remask $nfe nfe_curve $GPU1 --prompts "$PROMPTS" &
        PID1=$!

        # b1 remask on GPU1 (wait for original to finish)
        # b1 SDE on GPU0 (parallel with remask)
        run_gen b1 sde $nfe nfe_curve $GPU0 --prompts "$PROMPTS" &
        PID0=$!

        wait $PID1
        run_gen b1 remask $nfe nfe_curve $GPU1 --prompts "$PROMPTS"
        wait $PID0
    done
}

# ============================================================
# Part 2b: Long-form Generation (gen=512, 1024)
# ============================================================
run_part2b() {
    log "========== Part 2b: Long-form Generation =========="

    for gen_len in 512 1024; do
        run_gen original remask 64 longform $GPU1 --prompts "$PROMPTS" --gen_length $gen_len --max_prompts 100 &
        run_gen b1 sde 64 longform $GPU0 --prompts "$PROMPTS" --gen_length $gen_len --max_prompts 100 &
        wait

        run_gen b1 remask 64 longform $GPU1 --prompts "$PROMPTS" --gen_length $gen_len --max_prompts 100
    done
}

# ============================================================
# Part 2c: Diversity (Self-BLEU, 20 prompts × 5 seeds)
# ============================================================
run_part2c() {
    log "========== Part 2c: Diversity Test =========="

    run_gen original remask 64 diversity $GPU1 --prompts "$PROMPTS" --max_prompts 20 --seeds 42,123,456,789,1024 &
    run_gen b1 sde 64 diversity $GPU0 --prompts "$PROMPTS" --max_prompts 20 --seeds 42,123,456,789,1024 &
    wait

    run_gen b1 remask 64 diversity $GPU1 --prompts "$PROMPTS" --max_prompts 20 --seeds 42,123,456,789,1024
}

# ============================================================
# Part 2d: Speed-Quality (same as 2a but we already have timing in trace)
# ============================================================
run_part2d() {
    log "========== Part 2d: Speed data already captured in Part 2a trace =========="
    log "  (avg_time_per_sample in each result JSON)"
}

# ============================================================
# Metrics: compute on all generated result files
# ============================================================
run_metrics() {
    log "========== Computing Metrics =========="
    for f in eval_results/sde_gen_formal/*.json; do
        [ -f "$f" ] || continue
        [[ "$f" == *summary* ]] && continue
        [[ "$f" == *logs* ]] && continue

        # Skip if already has metrics
        if python3 -c "import json; d=json.load(open('$f')); exit(0 if 'metrics' in d else 1)" 2>/dev/null; then
            log "  Skip $f (has metrics)"
            continue
        fi

        log "  Metrics: $(basename $f)"
        CUDA_VISIBLE_DEVICES=$GPU0 python "$SCRIPT" --metrics "$f" --gpu 0 \
            2>&1 | tee -a "$LOGDIR/metrics_${TIMESTAMP}.log"
    done
}

# ============================================================
# Dispatch
# ============================================================
case "$PART" in
    part1)   run_part1 ;;
    part2a)  run_part2a ;;
    part2b)  run_part2b ;;
    part2c)  run_part2c ;;
    part2d)  run_part2d ;;
    metrics) run_metrics ;;
    all)
        run_part1
        run_part2a
        run_part2b
        run_part2c
        run_part2d
        run_metrics
        ;;
    *)
        echo "Usage: $0 {part1|part2a|part2b|part2c|part2d|metrics|all}"
        exit 1
        ;;
esac

log "========== ALL DONE =========="
