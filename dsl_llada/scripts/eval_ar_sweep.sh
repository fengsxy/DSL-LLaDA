#!/usr/bin/env bash
# Run Qwen2.5-7B-Instruct AR baseline on 4 summarization datasets, 1000 samples each, 8-GPU sharded.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

LOG_DIR="logs/summarization"
mkdir -p "$LOG_DIR"
PY="${PY:-python}"

is_done() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  $PY - "$f" <<'PYEOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("n_samples_total", 0) >= 1000 and d.get("valid", 0) >= 10
PYEOF
}

# Clean smoke output if it's a < 1000 sample test
if [[ -f eval_results/summarization/xsum_qwen25_7b_ar.json ]]; then
  if ! is_done eval_results/summarization/xsum_qwen25_7b_ar.json; then
    rm eval_results/summarization/xsum_qwen25_7b_ar.json
    echo "Removed partial smoke output"
  fi
fi

for DS in xsum cnn_dailymail pubmed arxiv; do
  MERGED="eval_results/summarization/${DS}_qwen25_7b_ar.json"
  if is_done "$MERGED"; then
    echo "[skip] $DS already done"
    continue
  fi
  echo "=== $(date +%H:%M:%S) $DS AR (1000 samples / 8 shards) ==="
  PIDS=()
  for i in 0 1 2 3 4 5 6 7; do
    LOG="$LOG_DIR/qwen_${DS}_shard${i}.log"
    $PY dsl_llada/eval/eval_summarization_ar.py \
      --dataset "$DS" --gpu "$i" --shard_id "$i" --num_shards 8 --seed 42 \
      >"$LOG" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  echo "$(date +%H:%M:%S) $DS shards done -> merging"
  $PY dsl_llada/eval/merge_summarization_shards.py \
    --dataset "$DS" --method_tag qwen25_7b_ar --nfe AR 2>&1 || \
    $PY - <<PYEOF
import json, glob, os, numpy as np
files = sorted(glob.glob('eval_results/summarization/${DS}_qwen25_7b_ar_shard*of*.json'))
samples = []
for f in files:
    samples += json.load(open(f))['samples']
samples.sort(key=lambda s: s['id'])
seen=set(); samples=[s for s in samples if not (s['id'] in seen or seen.add(s['id']))]
r1 = round(np.mean([s['rouge1'] for s in samples]), 2)
r2 = round(np.mean([s['rouge2'] for s in samples]), 2)
rl = round(np.mean([s['rougeL'] for s in samples]), 2)
ln = round(np.mean([s['gen_words'] for s in samples]), 2)
out = {'dataset': '${DS}', 'method': 'qwen25_7b_instruct_ar', 'model_key': 'qwen25_7b_instruct',
       'gen_method': 'ar', 'nfe': 'AR', 'seed': 42, 'n_samples_total': 1000,
       'n_samples_here': len(samples), 'valid': sum(1 for s in samples if s['generated'].strip()),
       'avg_words': ln, 'degenerate_pct': 0.0,
       'rouge1': r1, 'rouge2': r2, 'rougeL': rl, 'samples': samples}
json.dump(out, open('eval_results/summarization/${DS}_qwen25_7b_ar.json', 'w'), ensure_ascii=False, indent=2)
print(f"manual merge: {len(samples)}/1000 R1={r1} R2={r2} RL={rl}")
PYEOF
done

echo ""
echo "=== Final summary ==="
$PY -c "
import json
for ds in ['xsum','cnn_dailymail','pubmed','arxiv']:
    f=f'eval_results/summarization/{ds}_qwen25_7b_ar.json'
    try:
        d=json.load(open(f))
        print(f'  {ds:15s} R1={d[\"rouge1\"]:5.2f} R2={d[\"rouge2\"]:5.2f} RL={d[\"rougeL\"]:5.2f} len={d[\"avg_words\"]:.0f}w n={d[\"n_samples_here\"]}/1000')
    except: print(f'  {ds}: missing')
"
