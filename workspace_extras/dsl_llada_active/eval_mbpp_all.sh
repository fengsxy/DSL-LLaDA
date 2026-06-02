#!/bin/bash
# eval_mbpp_all.sh — Run MBPP eval at multiple step counts for all key models
# Uses 1 GPU per model, multiple steps sequentially per GPU
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

STEPS=(32 64 128 256)
OUT_DIR="eval_results/mbpp"
mkdir -p "$OUT_DIR"

# Model configs: "name|model_path|gpu"
MODELS=(
    "original|GSAI-ML/LLaDA-8B-Instruct|0"
    "sm_b2|checkpoints/pertoken_b2_d100_1k/checkpoint-1000|1"
    "xdlm|checkpoints/xdlm_k01_1k/checkpoint-1000|2"
    "mdm_cpt|checkpoints/mdm_cpt_1k/checkpoint-1000|3"
)

PIDS=()

for model_cfg in "${MODELS[@]}"; do
    IFS='|' read -r name model_path gpu <<< "$model_cfg"
    log="$OUT_DIR/${name}_eval.log"
    echo "[GPU $gpu] Starting MBPP eval for $name → $log"

    (
        for s in "${STEPS[@]}"; do
            out_file="$OUT_DIR/${name}_s${s}.json"
            if [ -f "$out_file" ]; then
                echo "  Skipping $name steps=$s (exists)"
                continue
            fi
            echo "  Running $name steps=$s ..."
            CUDA_VISIBLE_DEVICES=$gpu python dsl_llada/eval_mbpp.py \
                --model llada_dsl \
                --model_args "model_path=$model_path,steps=$s,gen_length=256,block_length=256" \
                --tasks mbpp \
                --confirm_run_unsafe_code \
                --num_fewshot 0 \
                --output_path "$out_file" \
                --device cuda:0 \
                --batch_size 1
        done
    ) > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#MODELS[@]} models launched. Monitor: tail -f $OUT_DIR/*_eval.log"
echo "Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    IFS='|' read -r name _ _ <<< "${MODELS[$i]}"
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: $name — see $OUT_DIR/${name}_eval.log"
        FAILED=$((FAILED+1))
    else
        echo "DONE: $name"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All MBPP evals complete ==="
else
    echo "=== $FAILED model(s) failed ==="
    exit 1
fi
