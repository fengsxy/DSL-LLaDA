#!/bin/bash
# Summarization benchmark: 4 datasets × 2 methods = 8 runs on 8 GPUs
# Datasets: xsum, cnn_dailymail, billsum, aeslc
# Methods: β=1 SDE (NFE=64), Original+eos_inf+b32 (NFE=64)

set -e
cd /home/ubuntu/efs/RMDM
source .venv/bin/activate

SCRIPT="dsl_llada/eval_summarization.py"
LOG_DIR="logs/summarization"
mkdir -p "$LOG_DIR"

echo "=== Starting summarization benchmark: $(date) ==="

# GPU 0: XSum SDE
python $SCRIPT --dataset xsum --method sde --model_key b1 --gpu 0 --nfe 64 \
    > "$LOG_DIR/xsum_sde.log" 2>&1 &

# GPU 1: XSum Orig+eos+b32
python $SCRIPT --dataset xsum --method remask --model_key original --gpu 1 --nfe 64 \
    --eos_inf --block_length 32 \
    > "$LOG_DIR/xsum_orig.log" 2>&1 &

# GPU 2: CNN/DailyMail SDE
python $SCRIPT --dataset cnn_dailymail --method sde --model_key b1 --gpu 2 --nfe 64 \
    > "$LOG_DIR/cnn_sde.log" 2>&1 &

# GPU 3: CNN/DailyMail Orig+eos+b32
python $SCRIPT --dataset cnn_dailymail --method remask --model_key original --gpu 3 --nfe 64 \
    --eos_inf --block_length 32 \
    > "$LOG_DIR/cnn_orig.log" 2>&1 &

# GPU 4: BillSum SDE
python $SCRIPT --dataset billsum --method sde --model_key b1 --gpu 4 --nfe 64 \
    > "$LOG_DIR/billsum_sde.log" 2>&1 &

# GPU 5: BillSum Orig+eos+b32
python $SCRIPT --dataset billsum --method remask --model_key original --gpu 5 --nfe 64 \
    --eos_inf --block_length 32 \
    > "$LOG_DIR/billsum_orig.log" 2>&1 &

# GPU 6: AESLC SDE
python $SCRIPT --dataset aeslc --method sde --model_key b1 --gpu 6 --nfe 64 \
    > "$LOG_DIR/aeslc_sde.log" 2>&1 &

# GPU 7: AESLC Orig+eos+b32
python $SCRIPT --dataset aeslc --method remask --model_key original --gpu 7 --nfe 64 \
    --eos_inf --block_length 32 \
    > "$LOG_DIR/aeslc_orig.log" 2>&1 &

echo "All 8 jobs launched. Waiting..."
wait
echo "=== All done: $(date) ==="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for f in eval_results/summarization/*.json; do
    echo ""
    python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f\"{d['dataset']:15s} {d['method']:35s} R1={d['rouge1']:5.1f}  R2={d['rouge2']:5.1f}  RL={d['rougeL']:5.1f}  valid={d['valid']}/{d['n_samples']}  words={d['avg_words']:.0f}  degen={d['degenerate_pct']:.0f}%\")
"
done
