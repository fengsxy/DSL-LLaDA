#!/bin/bash
# eval_mbpp_sweep.sh — MBPP 32-step sampling method sweep
# 15 experiments in 3 batches of 5-6, 6 GPUs parallel
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

OUT_DIR="eval_results/mbpp_sweep"
LOG_DIR="eval_results/batch_mbpp_sweep_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Model paths
declare -A PATHS=(
    [original]="GSAI-ML/LLaDA-8B-Instruct"
    [sm_b2]="checkpoints/pertoken_b2_d100_1k/checkpoint-1000"
    [b1]="checkpoints/pertoken_b1_d100_1k/checkpoint-1000"
    [sem_b05]="checkpoints/semantic_b05_d100_10k/checkpoint-3000"
)

run_one() {
    local gpu=$1 name=$2 model_path=$3 remasking=$4 block=$5
    local tag="${name}_${remasking}_b${block}"
    local out="$OUT_DIR/${tag}"
    local log="$LOG_DIR/${tag}.log"

    if [ -d "$out" ]; then
        echo "  SKIP $tag (exists)"
        return 0
    fi

    echo "[GPU $gpu] $tag"
    CUDA_VISIBLE_DEVICES=$gpu python dsl_llada/eval_mbpp.py \
        --model llada_dsl \
        --model_args "model_path=$model_path,steps=32,gen_length=256,block_length=$block,remasking=$remasking,confidence_eos_eot_inf=True" \
        --tasks mbpp --confirm_run_unsafe_code \
        --output_path "$out" \
        --device cuda:0 --batch_size 1 \
        > "$log" 2>&1
}

print_results() {
    echo ""
    echo "=== MBPP Sweep Results (32 steps) ==="
    python3 -c "
import json, glob, os
configs = [
    ('low_confidence', 256), ('low_confidence', 32), ('low_confidence', 8),
    ('d3im', 256), ('d3im', 32),
]
models = ['original', 'sm_b2', 'b1', 'sem_b05']
header = f\"{'Config':>20} |\" + ''.join(f' {m:>10} |' for m in models)
print(header)
print('-' * len(header))
for remask, block in configs:
    row = f'{remask}_b{block}:>20} |'
    for m in models:
        tag = f'{m}_{remask}_b{block}'
        files = glob.glob(f'$OUT_DIR/{tag}/*/results_*.json')
        if files:
            d = json.load(open(files[0]))
            acc = d.get('results',{}).get('mbpp',{}).get('pass_at_1,none',-1)
            row += f' {acc*100:8.1f}% |'
        else:
            row += '        - |'
    print(row)
"
}

# ============================================================
# Batch 1: low_confidence b32 + b8 (4 models each, 8 GPUs)
# ============================================================
echo "=== Batch 1: block diffusion sweep (b32 + b8) ==="
PIDS=()

run_one 0 original "${PATHS[original]}" low_confidence 32 &
PIDS+=($!)
run_one 1 sm_b2 "${PATHS[sm_b2]}" low_confidence 32 &
PIDS+=($!)
run_one 2 b1 "${PATHS[b1]}" low_confidence 32 &
PIDS+=($!)
run_one 3 sem_b05 "${PATHS[sem_b05]}" low_confidence 32 &
PIDS+=($!)
run_one 4 original "${PATHS[original]}" low_confidence 8 &
PIDS+=($!)
run_one 5 sm_b2 "${PATHS[sm_b2]}" low_confidence 8 &
PIDS+=($!)
run_one 6 b1 "${PATHS[b1]}" low_confidence 8 &
PIDS+=($!)
run_one 7 sem_b05 "${PATHS[sem_b05]}" low_confidence 8 &
PIDS+=($!)

for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "--- Batch 1 done ---"
print_results

# ============================================================
# Batch 2: d3im b256 + d3im b32 (4 models each, 8 GPUs)
# ============================================================
echo ""
echo "=== Batch 2: d3im sweep (b256 + b32) ==="
PIDS=()

run_one 0 original "${PATHS[original]}" d3im 256 &
PIDS+=($!)
run_one 1 sm_b2 "${PATHS[sm_b2]}" d3im 256 &
PIDS+=($!)
run_one 2 b1 "${PATHS[b1]}" d3im 256 &
PIDS+=($!)
run_one 3 sem_b05 "${PATHS[sem_b05]}" d3im 256 &
PIDS+=($!)
run_one 4 original "${PATHS[original]}" d3im 32 &
PIDS+=($!)
run_one 5 sm_b2 "${PATHS[sm_b2]}" d3im 32 &
PIDS+=($!)
run_one 6 b1 "${PATHS[b1]}" d3im 32 &
PIDS+=($!)
run_one 7 sem_b05 "${PATHS[sem_b05]}" d3im 32 &
PIDS+=($!)

for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "--- Batch 2 done ---"
print_results

echo ""
echo "=== ALL DONE ==="
print_results
