#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="logs/block_sde_full_${TIMESTAMP}"
OUTDIR="eval_results/block_sde_full_${TIMESTAMP}"
CKPT="${CKPT:-checkpoints/unit_uniform_paper_b1_trainable_1k_20260508_0320/checkpoint-1000}"

mkdir -p "$LOGDIR" "$OUTDIR"

run_job() {
    local gpu="$1"
    local dataset="$2"
    local data_file="$3"
    local start="$4"
    local end="$5"
    local gen_length="$6"
    local block_size="$7"
    local steps_per_block="$8"
    local name="$9"

    echo "[$(date)] launch ${name} on GPU ${gpu}: ${dataset} ${start}:${end}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 dsl_llada/eval_block_sde.py \
        --gpu 0 \
        --ckpt "$CKPT" \
        --dataset "$dataset" \
        --data_file "$data_file" \
        --start_index "$start" \
        --end_index "$end" \
        --n_problems 0 \
        --gen_length "$gen_length" \
        --block_sizes "$block_size" \
        --steps_per_block "$steps_per_block" \
        --skip_baselines \
        --only "block_${block_size}_${steps_per_block}" \
        --bi 2.0 \
        --ns 0.01 \
        --output "${OUTDIR}/${name}.json" \
        > "${LOGDIR}/${name}.log" 2>&1
    echo "[$(date)] done ${name}"
}

run_job 0 gsm8k eval_data/gsm8k_full.json 0 330 256 8 4 gsm8k_full_s00 &
PID0=$!
run_job 1 gsm8k eval_data/gsm8k_full.json 330 660 256 8 4 gsm8k_full_s01 &
PID1=$!
run_job 2 gsm8k eval_data/gsm8k_full.json 660 990 256 8 4 gsm8k_full_s02 &
PID2=$!
run_job 3 gsm8k eval_data/gsm8k_full.json 990 1319 256 8 4 gsm8k_full_s03 &
PID3=$!

run_job 4 math eval_data/math_500.json 0 250 512 32 16 math500_s00 &
PID4=$!
run_job 5 math eval_data/math_500.json 250 500 512 32 16 math500_s01 &
PID5=$!

echo "[$(date)] Logs: ${LOGDIR}"
echo "[$(date)] Outputs: ${OUTDIR}"

wait "$PID0" "$PID1" "$PID2" "$PID3" "$PID4" "$PID5"

python3 dsl_llada/aggregate_block_sde_shards.py \
    --input_dir "$OUTDIR" \
    --output "${OUTDIR}/block_sde_full_merged.json"

echo "[$(date)] all Block-SDE full jobs complete"
