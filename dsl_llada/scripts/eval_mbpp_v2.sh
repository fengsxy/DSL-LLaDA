#!/bin/bash
# eval_mbpp_v2.sh — MBPP eval with correct settings (confidence_eos_eot_inf=True)
# Matches LLaDA official: gen_length=256, block_length=256, confidence_eos_eot_inf=True
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

OUT_DIR="eval_results/mbpp_v2"
LOG_DIR="eval_results/batch_mbpp_v2_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_DIR" "$LOG_DIR"

PIDS=()
NAMES=()

launch() {
    local gpu=$1 name=$2 model_path=$3; shift 3
    local log="$LOG_DIR/${name}.log"
    echo "[GPU $gpu] $name → $log"
    (
        for s in 32 64 128 256; do
            out="$OUT_DIR/${name}_s${s}"
            if [ -d "$out" ]; then
                echo "  Skip $name s=$s (exists)"
                continue
            fi
            echo "=== $name steps=$s ==="
            CUDA_VISIBLE_DEVICES=$gpu python dsl_llada/eval/eval_mbpp.py \
                --model llada_dsl \
                --model_args "model_path=$model_path,steps=$s,gen_length=256,block_length=256,confidence_eos_eot_inf=True" \
                --tasks mbpp --confirm_run_unsafe_code \
                --output_path "$out" \
                --device cuda:0 --batch_size 1
        done
    ) > "$log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
}

# 6 models on 6 GPUs
launch 0 original   "GSAI-ML/LLaDA-8B-Instruct"
launch 1 sm_b2      "checkpoints/pertoken_b2_d100_1k/checkpoint-1000"
launch 2 b1         "checkpoints/pertoken_b1_d100_1k/checkpoint-1000"
launch 3 xdlm       "checkpoints/xdlm_baseline_1k/checkpoint-1000"
launch 4 mdm_cpt    "checkpoints/mdm_baseline_1k/checkpoint-1000"
launch 5 sem_b05_3k "checkpoints/semantic_b05_d100_10k/checkpoint-3000"

echo ""
echo "=== ${#PIDS[@]} MBPP v2 jobs launched ==="
echo "Monitor: tail -f $LOG_DIR/*.log"
echo "Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${NAMES[$i]}"
        FAILED=$((FAILED+1))
    else
        echo "DONE: ${NAMES[$i]}"
    fi
done

echo ""
[ $FAILED -eq 0 ] && echo "=== All complete ===" || echo "=== $FAILED failed ==="

# Print results
echo ""
echo "=== MBPP v2 Results ==="
python3 -c "
import json, glob
models = ['original','sm_b2','b1','xdlm','mdm_cpt','sem_b05_3k']
steps = ['32','64','128','256']
print(f\"{'Model':>12} | \" + ' | '.join(f's={s:>3}' for s in steps))
print('-'*60)
for m in models:
    row = f'{m:>12} |'
    for s in steps:
        files = glob.glob(f'$OUT_DIR/{m}_s{s}/*/results_*.json')
        if files:
            d = json.load(open(files[0]))
            acc = d.get('results',{}).get('mbpp',{}).get('pass_at_1,none',-1)
            row += f' {acc*100:5.1f} |' if acc >= 0 else '   ??? |'
        else:
            row += '     - |'
    print(row)
"
