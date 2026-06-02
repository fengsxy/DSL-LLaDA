#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="logs/math500_baselines_${TIMESTAMP}"
OUTDIR="${OUTDIR:-eval_results/block_sde_full_${TIMESTAMP}}"
CKPT="${CKPT:-checkpoints/unit_uniform_paper_b1_trainable_1k_20260508_0320/checkpoint-1000}"

mkdir -p "$LOGDIR" "$OUTDIR"

run_job() {
    local gpu="$1"
    local start="$2"
    local end="$3"
    local only="$4"
    local name="$5"

    echo "[$(date)] launch ${name} on GPU ${gpu}: math ${start}:${end}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 dsl_llada/eval_block_sde.py \
        --gpu 0 \
        --ckpt "$CKPT" \
        --dataset math \
        --data_file eval_data/math_500.json \
        --start_index "$start" \
        --end_index "$end" \
        --n_problems 0 \
        --gen_length 512 \
        --only "$only" \
        --bi 2.0 \
        --ns 0.01 \
        --output "${OUTDIR}/${name}.json" \
        > "${LOGDIR}/${name}.log" 2>&1
    echo "[$(date)] done ${name}"
}

run_stage() {
    local only="$1"
    local prefix="$2"
    run_job 0 0 125 "$only" "${prefix}_s00" &
    PID0=$!
    run_job 1 125 250 "$only" "${prefix}_s01" &
    PID1=$!
    run_job 2 250 375 "$only" "${prefix}_s02" &
    PID2=$!
    run_job 3 375 500 "$only" "${prefix}_s03" &
    PID3=$!
    wait "$PID0" "$PID1" "$PID2" "$PID3"
}

echo "[$(date)] Logs: ${LOGDIR}"
echo "[$(date)] Outputs: ${OUTDIR}"

run_stage remask_64_b32 math500_remask64_b32
run_stage pure_sde_32 math500_pure_sde32

python3 dsl_llada/aggregate_block_sde_shards.py \
    --input_dir "$OUTDIR" \
    --output "${OUTDIR}/math500_baselines_merged.json"

echo "[$(date)] MATH-500 baselines complete"
