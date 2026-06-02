#!/bin/bash
# dsl_llada/eval_after_training.sh
# Automated evaluation pipeline: waits for training to finish, then runs all evals
# Usage: nohup bash dsl_llada/eval_after_training.sh > logs/eval_auto.log 2>&1 &
#
# Phases:
#   1. GSM8K 64-step + Corruption probe (6 jobs, GPU 0-5)
#   2. Calibration (2 GPU per run, sequential)
#   3. MATH-500 64-step + GSM8K 256-step (6 jobs, GPU 0-5)

set -e
source /home/ubuntu/efs/Retrieval_head/retrieval_head/.venv/bin/activate

TRAIN_LOG="logs/train_5k_unfreeze3.log"
CKPT_BASE="./checkpoints/dsl_5k_unfreeze_20260310_231823"
CKPT_1K="${CKPT_BASE}/checkpoint-1000"
CKPT_5K="${CKPT_BASE}/checkpoint-5000"
ORIGINAL="GSAI-ML/LLaDA-8B-Instruct"
RESULTS_DIR="./results/eval_5k_unfreeze"

mkdir -p "${RESULTS_DIR}" logs

echo "=== Eval Pipeline: waiting for training to finish ==="
echo "Watching: ${TRAIN_LOG}"
echo "Checkpoints: ${CKPT_1K}, ${CKPT_5K}"

# ---- Wait for training to complete ----
while true; do
    if ! pgrep -f "llada_cpt_dsl.py" > /dev/null 2>&1; then
        echo "[$(date)] Training processes not found, checking log..."
        if grep -q "Training completed" "${TRAIN_LOG}" 2>/dev/null || \
           grep -q "5000/5000" "${TRAIN_LOG}" 2>/dev/null || \
           grep -q "train_runtime" "${TRAIN_LOG}" 2>/dev/null; then
            echo "[$(date)] Training completed!"
            break
        fi
        if [ -d "${CKPT_5K}" ]; then
            echo "[$(date)] checkpoint-5000 exists, training likely done"
            break
        fi
        echo "[$(date)] No training process and no completion signal, waiting..."
    else
        LAST_STEP=$(grep "'loss'" "${TRAIN_LOG}" 2>/dev/null | tail -1 | grep -oP '\d+/5000' | head -1 || echo "?")
        echo "[$(date)] Training in progress... step ${LAST_STEP}"
    fi
    sleep 300
done

echo ""
echo "============================================"
echo "=== Starting evaluations at $(date) ==="
echo "============================================"
echo ""

# Verify checkpoints exist
for ckpt in "${CKPT_1K}" "${CKPT_5K}"; do
    if [ ! -d "${ckpt}" ]; then
        echo "WARNING: ${ckpt} not found, skipping its evals"
    fi
done

# ======================================================================
# Phase 1: GSM8K 64-step + Corruption Probe (6 parallel jobs, GPU 0-5)
# ======================================================================
echo "=== Phase 1: GSM8K 64-step + Corruption Probe (6 parallel jobs) ==="

# GSM8K 64-step: GPU 0-2
echo "[$(date)] Starting GSM8K 64-step: original"
CUDA_VISIBLE_DEVICES=0 python dsl_llada/eval_reasoning.py \
    --checkpoint "${ORIGINAL}" \
    --method standard --steps 64 --dataset gsm8k \
    --gpu 0 --gen_length 512 \
    --output "${RESULTS_DIR}/gsm8k_64step_original.json" \
    > logs/eval_gsm8k_64_original.log 2>&1 &
PID1=$!

if [ -d "${CKPT_1K}" ]; then
    echo "[$(date)] Starting GSM8K 64-step: ckpt-1000"
    CUDA_VISIBLE_DEVICES=1 python dsl_llada/eval_reasoning.py \
        --checkpoint "${CKPT_1K}" \
        --method standard --steps 64 --dataset gsm8k \
        --gpu 0 --gen_length 512 \
        --output "${RESULTS_DIR}/gsm8k_64step_ckpt1000.json" \
        > logs/eval_gsm8k_64_ckpt1000.log 2>&1 &
    PID2=$!
fi

if [ -d "${CKPT_5K}" ]; then
    echo "[$(date)] Starting GSM8K 64-step: ckpt-5000"
    CUDA_VISIBLE_DEVICES=2 python dsl_llada/eval_reasoning.py \
        --checkpoint "${CKPT_5K}" \
        --method standard --steps 64 --dataset gsm8k \
        --gpu 0 --gen_length 512 \
        --output "${RESULTS_DIR}/gsm8k_64step_ckpt5000.json" \
        > logs/eval_gsm8k_64_ckpt5000.log 2>&1 &
    PID3=$!
fi

# Corruption Probe: GPU 3-5
echo "[$(date)] Starting Corruption: original"
CUDA_VISIBLE_DEVICES=3 python dsl_llada/test_corruption_probe.py \
    --checkpoint "${ORIGINAL}" --gpu 0 \
    --output "${RESULTS_DIR}/corruption_original.json" \
    > logs/eval_corruption_original.log 2>&1 &
PID4=$!

if [ -d "${CKPT_1K}" ]; then
    echo "[$(date)] Starting Corruption: ckpt-1000"
    CUDA_VISIBLE_DEVICES=4 python dsl_llada/test_corruption_probe.py \
        --checkpoint "${CKPT_1K}" --gpu 0 \
        --output "${RESULTS_DIR}/corruption_ckpt1000.json" \
        > logs/eval_corruption_ckpt1000.log 2>&1 &
    PID5=$!
fi

if [ -d "${CKPT_5K}" ]; then
    echo "[$(date)] Starting Corruption: ckpt-5000"
    CUDA_VISIBLE_DEVICES=5 python dsl_llada/test_corruption_probe.py \
        --checkpoint "${CKPT_5K}" --gpu 0 \
        --output "${RESULTS_DIR}/corruption_ckpt5000.json" \
        > logs/eval_corruption_ckpt5000.log 2>&1 &
    PID6=$!
fi

echo "[$(date)] Waiting for Phase 1..."
for pid_var in PID1 PID2 PID3 PID4 PID5 PID6; do
    pid=${!pid_var:-0}
    if [ "$pid" -ne 0 ]; then
        wait $pid 2>/dev/null && echo "  ${pid_var} done" || echo "  ${pid_var} failed"
    fi
done

echo ""
echo "=== Phase 1 complete at $(date) ==="

# ======================================================================
# Phase 2: Calibration (sequential, needs 2 GPUs per run)
# ======================================================================
echo ""
echo "=== Phase 2: Calibration ==="

if [ -d "${CKPT_1K}" ]; then
    echo "[$(date)] Calibration: original vs ckpt-1000"
    CUDA_VISIBLE_DEVICES=0,1 python dsl_llada/test_calibration_100.py \
        --ckpt_path "${CKPT_1K}" \
        --output_prefix "${RESULTS_DIR}/calibration_ckpt1000" \
        > logs/eval_calibration_ckpt1000.log 2>&1
    echo "  Calibration ckpt-1000 done"
fi

if [ -d "${CKPT_5K}" ]; then
    echo "[$(date)] Calibration: original vs ckpt-5000"
    CUDA_VISIBLE_DEVICES=0,1 python dsl_llada/test_calibration_100.py \
        --ckpt_path "${CKPT_5K}" \
        --output_prefix "${RESULTS_DIR}/calibration_ckpt5000" \
        > logs/eval_calibration_ckpt5000.log 2>&1
    echo "  Calibration ckpt-5000 done"
fi

echo ""
echo "=== Phase 2 complete at $(date) ==="

# ======================================================================
# Phase 3: MATH-500 64-step + GSM8K 256-step (6 parallel jobs, GPU 0-5)
# ======================================================================
echo ""
echo "=== Phase 3: MATH-500 64-step + GSM8K 256-step (6 parallel jobs) ==="

# MATH-500 64-step: GPU 0-2
echo "[$(date)] Starting MATH-500 64-step: original"
CUDA_VISIBLE_DEVICES=0 python dsl_llada/eval_reasoning.py \
    --checkpoint "${ORIGINAL}" \
    --method standard --steps 64 --dataset math500 \
    --gpu 0 --gen_length 512 \
    --output "${RESULTS_DIR}/math500_64step_original.json" \
    > logs/eval_math500_64_original.log 2>&1 &
PID1=$!

if [ -d "${CKPT_1K}" ]; then
    echo "[$(date)] Starting MATH-500 64-step: ckpt-1000"
    CUDA_VISIBLE_DEVICES=1 python dsl_llada/eval_reasoning.py \
        --checkpoint "${CKPT_1K}" \
        --method standard --steps 64 --dataset math500 \
        --gpu 0 --gen_length 512 \
        --output "${RESULTS_DIR}/math500_64step_ckpt1000.json" \
        > logs/eval_math500_64_ckpt1000.log 2>&1 &
    PID2=$!
fi

if [ -d "${CKPT_5K}" ]; then
    echo "[$(date)] Starting MATH-500 64-step: ckpt-5000"
    CUDA_VISIBLE_DEVICES=2 python dsl_llada/eval_reasoning.py \
        --checkpoint "${CKPT_5K}" \
        --method standard --steps 64 --dataset math500 \
        --gpu 0 --gen_length 512 \
        --output "${RESULTS_DIR}/math500_64step_ckpt5000.json" \
        > logs/eval_math500_64_ckpt5000.log 2>&1 &
    PID3=$!
fi

# GSM8K 256-step: GPU 3-5
echo "[$(date)] Starting GSM8K 256-step: original"
CUDA_VISIBLE_DEVICES=3 python dsl_llada/eval_reasoning.py \
    --checkpoint "${ORIGINAL}" \
    --method standard --steps 256 --dataset gsm8k \
    --gpu 0 --gen_length 512 \
    --output "${RESULTS_DIR}/gsm8k_256step_original.json" \
    > logs/eval_gsm8k_256_original.log 2>&1 &
PID4=$!

if [ -d "${CKPT_1K}" ]; then
    echo "[$(date)] Starting GSM8K 256-step: ckpt-1000"
    CUDA_VISIBLE_DEVICES=4 python dsl_llada/eval_reasoning.py \
        --checkpoint "${CKPT_1K}" \
        --method standard --steps 256 --dataset gsm8k \
        --gpu 0 --gen_length 512 \
        --output "${RESULTS_DIR}/gsm8k_256step_ckpt1000.json" \
        > logs/eval_gsm8k_256_ckpt1000.log 2>&1 &
    PID5=$!
fi

if [ -d "${CKPT_5K}" ]; then
    echo "[$(date)] Starting GSM8K 256-step: ckpt-5000"
    CUDA_VISIBLE_DEVICES=5 python dsl_llada/eval_reasoning.py \
        --checkpoint "${CKPT_5K}" \
        --method standard --steps 256 --dataset gsm8k \
        --gpu 0 --gen_length 512 \
        --output "${RESULTS_DIR}/gsm8k_256step_ckpt5000.json" \
        > logs/eval_gsm8k_256_ckpt5000.log 2>&1 &
    PID6=$!
fi

echo "[$(date)] Waiting for Phase 3..."
for pid_var in PID1 PID2 PID3 PID4 PID5 PID6; do
    pid=${!pid_var:-0}
    if [ "$pid" -ne 0 ]; then
        wait $pid 2>/dev/null && echo "  ${pid_var} done" || echo "  ${pid_var} failed"
    fi
done

echo ""
echo "=== Phase 3 complete at $(date) ==="

# ======================================================================
# Summary
# ======================================================================
echo ""
echo "=========================================="
echo "=== ALL EVALUATIONS COMPLETE at $(date) ==="
echo "=========================================="
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""

# GSM8K results
echo "--- GSM8K Accuracy ---"
echo "  Model                  | 64-step  | 256-step"
echo "  -----------------------|----------|----------"
for tag in original ckpt1000 ckpt5000; do
    acc_64=$(python -c "import json; d=json.load(open('${RESULTS_DIR}/gsm8k_64step_${tag}.json')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
    acc_256=$(python -c "import json; d=json.load(open('${RESULTS_DIR}/gsm8k_256step_${tag}.json')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
    printf "  %-23s| %-9s| %s\n" "${tag}" "${acc_64}" "${acc_256}"
done

echo ""
echo "--- MATH-500 Accuracy (64-step) ---"
for tag in original ckpt1000 ckpt5000; do
    acc=$(python -c "import json; d=json.load(open('${RESULTS_DIR}/math500_64step_${tag}.json')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
    echo "  ${tag}: ${acc}"
done

echo ""
echo "--- Corruption Robustness ---"
for tag in original ckpt1000 ckpt5000; do
    f="${RESULTS_DIR}/corruption_${tag}.json"
    if [ -f "$f" ]; then
        echo "  ${tag}:"
        python -c "
import json
d = json.load(open('$f'))
if 'results' in d:
    for r in d['results']:
        rate = r.get('corruption_rate', r.get('rate', '?'))
        fix = r.get('fix_rate', r.get('corrupted_fixed_rate', '?'))
        if isinstance(fix, float): fix = f'{fix:.1%}'
        print(f'    corruption={rate}: fix_rate={fix}')
" 2>/dev/null || echo "    parse error"
    fi
done

echo ""
echo "--- Calibration ECE ---"
for tag in ckpt1000 ckpt5000; do
    f="${RESULTS_DIR}/calibration_${tag}_token_100.json"
    if [ -f "$f" ]; then
        echo "  ${tag}:"
        python -c "
import json
d = json.load(open('$f'))
for model_name, data in d.items():
    if model_name == 'n_texts': continue
    for rate, vals in data.items():
        if isinstance(vals, dict) and 'ece' in vals:
            print(f'    {model_name} mask={rate}: ECE={vals[\"ece\"]:.4f}')
" 2>/dev/null || echo "    parse error"
    fi
done

echo ""
echo "All results saved to: ${RESULTS_DIR}/"
echo "Done!"
