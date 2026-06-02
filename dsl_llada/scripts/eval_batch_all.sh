#!/bin/bash
# eval_batch_all.sh — Run all pending experiments in parallel across 8 GPUs
# T5 trip planning (remask + SDE) + MBPP at multiple step counts
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

LOG_DIR="eval_results/batch_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

PIDS=()
NAMES=()

launch() {
    local gpu=$1 name=$2; shift 2
    local log="$LOG_DIR/${name}.log"
    echo "[GPU $gpu] $name → $log"
    CUDA_VISIBLE_DEVICES=$gpu "$@" > "$log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
}

# --- GPU 0: b1 T5 (remask + SDE) ---
launch 0 "b1_t5" python dsl_llada/eval/eval_unified.py \
    --model_key b1 --gpu 0 --tests t5

# --- GPU 1: sm_b2 T5 (remask + SDE, overwrite previous remask-only) ---
launch 1 "sm_b2_t5" python dsl_llada/eval/eval_unified.py \
    --model_key sm_b2 --gpu 0 --tests t5

# --- GPU 2: original T5 (remask only, already done but re-run for consistency) ---
# Skip — already have results. Use GPU 2 for MBPP instead.

# --- GPU 2: original MBPP 32/64/128/256 steps ---
launch 2 "original_mbpp" bash -c '
    for s in 32 64 128 256; do
        echo "=== original steps=$s ==="
        python dsl_llada/eval/eval_mbpp.py \
            --model llada_dsl \
            --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,steps=$s,gen_length=256,block_length=256" \
            --tasks mbpp --confirm_run_unsafe_code --num_fewshot 0 \
            --output_path "eval_results/mbpp/original_s${s}.json" \
            --device cuda:0 --batch_size 1
    done
'

# --- GPU 3: sm_b2 MBPP ---
launch 3 "sm_b2_mbpp" bash -c '
    for s in 32 64 128 256; do
        echo "=== sm_b2 steps=$s ==="
        python dsl_llada/eval/eval_mbpp.py \
            --model llada_dsl \
            --model_args "model_path=checkpoints/pertoken_b2_d100_1k/checkpoint-1000,steps=$s,gen_length=256,block_length=256" \
            --tasks mbpp --confirm_run_unsafe_code --num_fewshot 0 \
            --output_path "eval_results/mbpp/sm_b2_s${s}.json" \
            --device cuda:0 --batch_size 1
    done
'

# --- GPU 4: b1 MBPP ---
launch 4 "b1_mbpp" bash -c '
    for s in 32 64 128 256; do
        echo "=== b1 steps=$s ==="
        python dsl_llada/eval/eval_mbpp.py \
            --model llada_dsl \
            --model_args "model_path=checkpoints/pertoken_b1_d100_1k/checkpoint-1000,steps=$s,gen_length=256,block_length=256" \
            --tasks mbpp --confirm_run_unsafe_code --num_fewshot 0 \
            --output_path "eval_results/mbpp/b1_s${s}.json" \
            --device cuda:0 --batch_size 1
    done
'

# --- GPU 5: xdlm MBPP ---
launch 5 "xdlm_mbpp" bash -c '
    for s in 32 64 128 256; do
        echo "=== xdlm steps=$s ==="
        python dsl_llada/eval/eval_mbpp.py \
            --model llada_dsl \
            --model_args "model_path=checkpoints/xdlm_k01_1k/checkpoint-1000,steps=$s,gen_length=256,block_length=256" \
            --tasks mbpp --confirm_run_unsafe_code --num_fewshot 0 \
            --output_path "eval_results/mbpp/xdlm_s${s}.json" \
            --device cuda:0 --batch_size 1
    done
'

# --- GPU 6: mdm_cpt MBPP ---
launch 6 "mdm_cpt_mbpp" bash -c '
    for s in 32 64 128 256; do
        echo "=== mdm_cpt steps=$s ==="
        python dsl_llada/eval/eval_mbpp.py \
            --model llada_dsl \
            --model_args "model_path=checkpoints/mdm_cpt_1k/checkpoint-1000,steps=$s,gen_length=256,block_length=256" \
            --tasks mbpp --confirm_run_unsafe_code --num_fewshot 0 \
            --output_path "eval_results/mbpp/mdm_cpt_s${s}.json" \
            --device cuda:0 --batch_size 1
    done
'

# --- GPU 7: sem_b05_3k MBPP ---
launch 7 "sem_b05_mbpp" bash -c '
    for s in 32 64 128 256; do
        echo "=== sem_b05_3k steps=$s ==="
        python dsl_llada/eval/eval_mbpp.py \
            --model llada_dsl \
            --model_args "model_path=checkpoints/sem_b05_3k/checkpoint-3000,steps=$s,gen_length=256,block_length=256" \
            --tasks mbpp --confirm_run_unsafe_code --num_fewshot 0 \
            --output_path "eval_results/mbpp/sem_b05_3k_s${s}.json" \
            --device cuda:0 --batch_size 1
    done
'

echo ""
echo "=== ${#PIDS[@]} jobs launched on GPUs 0-7 ==="
echo "Monitor: tail -f $LOG_DIR/*.log"
echo "Waiting for all to finish..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${NAMES[$i]} — see $LOG_DIR/${NAMES[$i]}.log"
        FAILED=$((FAILED+1))
    else
        echo "DONE: ${NAMES[$i]}"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All ${#PIDS[@]} jobs complete ==="
else
    echo "=== $FAILED job(s) failed. Check logs in $LOG_DIR ==="
    exit 1
fi
