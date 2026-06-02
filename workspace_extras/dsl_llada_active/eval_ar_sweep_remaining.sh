#!/usr/bin/env bash
# Run Qwen2.5-7B AR baseline on CNN/DM, PubMed, arXiv (XSum already done).
# Uses 7 shards across GPUs 0,1,3,4,5,6,7 (skips GPU 2 due to other research).
set -euo pipefail
cd /home/ubuntu/efs/RMDM
PY=/home/ubuntu/efs/RMDM/.venv/bin/python
LOG_DIR="logs/summarization"
mkdir -p "$LOG_DIR"

GPUS=(0 1 3 4 5 6 7)  # 7 GPUs
NSHARD=7

for DS in cnn_dailymail pubmed arxiv; do
  MERGED="eval_results/summarization/${DS}_qwen25_7b_ar.json"
  if [[ -f "$MERGED" ]]; then
    N=$($PY -c "import json;print(json.load(open('$MERGED'))['n_samples_total'])" 2>/dev/null || echo 0)
    if [[ "$N" -ge 1000 ]]; then
      echo "[skip] $DS already merged"; continue
    fi
  fi
  echo "=== $(date +%H:%M:%S) $DS AR (1000 samples / $NSHARD shards) ==="
  PIDS=()
  for k in 0 1 2 3 4 5 6; do
    GPU=${GPUS[$k]}
    LOG="$LOG_DIR/qwen_${DS}_shard${k}.log"
    $PY dsl_llada/eval_summarization_ar.py \
      --dataset "$DS" --gpu "$GPU" --shard_id "$k" --num_shards "$NSHARD" --seed 42 \
      >"$LOG" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p" || echo "  shard pid=$p failed"; done

  echo "$(date +%H:%M:%S) $DS shards done -> merging"
  $PY <<PYEOF
import json, glob, numpy as np
files = sorted(glob.glob('/home/ubuntu/efs/RMDM/eval_results/summarization/${DS}_qwen25_7b_ar_shard*of${NSHARD}.json'))
print(f"  shards: {len(files)}")
samples = []; seen = set()
for f in files:
    d = json.load(open(f))
    for s in d['samples']:
        if s['id'] not in seen: seen.add(s['id']); samples.append(s)
samples.sort(key=lambda s: s['id'])
r1 = round(np.mean([s['rouge1'] for s in samples]), 2)
r2 = round(np.mean([s['rouge2'] for s in samples]), 2)
rl = round(np.mean([s['rougeL'] for s in samples]), 2)
ln = round(np.mean([s['gen_words'] for s in samples]), 2)
out = {
    'dataset': '${DS}', 'method': 'qwen25_7b_instruct_ar', 'model_key': 'qwen25_7b_instruct',
    'gen_method': 'ar', 'nfe': 'AR', 'seed': 42,
    'n_samples_total': 1000, 'n_samples_here': len(samples),
    'valid': sum(1 for s in samples if s['generated'].strip()),
    'avg_words': ln, 'degenerate_pct': 0.0,
    'rouge1': r1, 'rouge2': r2, 'rougeL': rl, 'samples': samples,
}
json.dump(out, open('/home/ubuntu/efs/RMDM/eval_results/summarization/${DS}_qwen25_7b_ar.json', 'w'), ensure_ascii=False, indent=2)
print(f"  ${DS} Qwen2.5-7B AR: R1={r1} R2={r2} RL={rl} len={ln}w n={len(samples)}/1000")
PYEOF
done

echo ""
echo "=== Final summary ==="
$PY -c "
import json
for ds in ['xsum','cnn_dailymail','pubmed','arxiv']:
    f=f'/home/ubuntu/efs/RMDM/eval_results/summarization/{ds}_qwen25_7b_ar.json'
    try:
        d=json.load(open(f))
        print(f'  {ds:15s} R1={d[\"rouge1\"]:5.2f} R2={d[\"rouge2\"]:5.2f} RL={d[\"rougeL\"]:5.2f} len={d[\"avg_words\"]:.0f}w n={d[\"n_samples_here\"]}/1000')
    except Exception as e: print(f'  {ds}: missing ({e})')
"
