#!/bin/bash
# eval_all.sh — Run unified eval for all models in parallel (1 GPU per model)
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

MODELS=(original sm_b2 sem_b05_2k sem_b05_3k mdm_cpt xdlm frozen_embed)
PIDS=()

mkdir -p eval_results

for i in "${!MODELS[@]}"; do
    gpu=$i
    model=${MODELS[$i]}
    log="eval_results/${model}_eval.log"
    echo "[GPU $gpu] Starting eval for $model → $log"
    python dsl_llada/eval/eval_unified.py \
        --model_key "$model" --gpu "$gpu" --skip_existing \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#MODELS[@]} models launched on GPUs 0-$((${#MODELS[@]}-1))"
echo "Monitor: tail -f eval_results/*_eval.log"
echo "Waiting for all to finish..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${MODELS[$i]} (GPU $i) — see eval_results/${MODELS[$i]}_eval.log"
        FAILED=$((FAILED+1))
    else
        echo "DONE: ${MODELS[$i]} (GPU $i)"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All models complete. Generating summary ==="
    python dsl_llada/eval/eval_unified.py --aggregate
else
    echo "=== $FAILED model(s) failed. Check logs. ==="
    # Still generate partial summary
    python dsl_llada/eval/eval_unified.py --aggregate
    exit 1
fi
