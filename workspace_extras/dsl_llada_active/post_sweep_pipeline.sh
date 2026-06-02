#!/usr/bin/env bash
# After Stage 1 sweep completes, run BERTScore + case analysis + LLM judge
# for every (dataset, NFE=64) × (SDE, LLaDA, LLaDA+EOS+Block) combination.
#
# Idempotent: BERTScore skips already-scored files, judge resumes.
set -euo pipefail
cd /home/ubuntu/efs/RMDM
set -a; source .env.local; set +a

DATASETS=(xsum cnn_dailymail pubmed arxiv billsum)
NFE=${NFE:-64}
SUM_DIR="eval_results/summarization"
ANA_DIR="$SUM_DIR/analysis"
mkdir -p "$ANA_DIR"

PY=/home/ubuntu/efs/RMDM/.venv/bin/python
STAMP="[$(date '+%H:%M:%S')]"

# -------- 1. BERTScore for all merged files at this NFE --------
echo "$STAMP === 1/4  BERTScore on all NFE=${NFE} files ==="
FILES=()
for d in "${DATASETS[@]}"; do
  for tag in b1_sde original_remask original_remask_eosInf_b32; do
    f="$SUM_DIR/${d}_${tag}_nfe${NFE}.json"
    [[ -f "$f" ]] && FILES+=("$f")
  done
done
echo "  files: ${#FILES[@]}"
if [[ ${#FILES[@]} -gt 0 ]]; then
  $PY dsl_llada/compute_bertscore_summarization.py --files "${FILES[@]}" --gpu 0 2>&1 \
    | grep -vE "UserWarning|not writable|TRAIN this model|newly initialized" | tail -40
fi

# -------- 2. Case analysis: SDE vs {default, block} per dataset --------
echo ""
echo "$STAMP === 2/4  Case analysis ==="
for d in "${DATASETS[@]}"; do
  main="$SUM_DIR/${d}_b1_sde_nfe${NFE}.json"
  [[ ! -f "$main" ]] && { echo "  skip $d (no SDE result)"; continue; }
  for base_tag in original_remask original_remask_eosInf_b32; do
    base="$SUM_DIR/${d}_${base_tag}_nfe${NFE}.json"
    [[ ! -f "$base" ]] && continue
    out="$ANA_DIR/${d}_nfe${NFE}__b1_sde_vs_${base_tag}.md"
    if [[ -f "$out" ]]; then
      echo "  [skip] $(basename "$out")"
      continue
    fi
    echo "  analyzing $d vs $base_tag..."
    $PY dsl_llada/analyze_summarization.py \
      --dataset "$d" --nfe "$NFE" \
      --main b1_sde --baseline "$base_tag" \
      --topk 3 2>&1 | tail -3
  done
done

# -------- 3. LLM-as-judge: SDE vs {default, block} per dataset --------
echo ""
echo "$STAMP === 3/4  LLM-as-judge (GPT-5.4) ==="
JUDGE_N=${JUDGE_N:-100}
JUDGE_C=${JUDGE_C:-8}
for d in "${DATASETS[@]}"; do
  main_file="$SUM_DIR/${d}_b1_sde_nfe${NFE}.json"
  [[ ! -f "$main_file" ]] && { echo "  skip $d (no SDE result)"; continue; }
  for base_tag in original_remask original_remask_eosInf_b32; do
    base_file="$SUM_DIR/${d}_${base_tag}_nfe${NFE}.json"
    [[ ! -f "$base_file" ]] && continue
    out="$ANA_DIR/judge__${d}_nfe${NFE}__b1_sde_vs_${base_tag}.json"
    done_n=0
    if [[ -f "$out" ]]; then
      done_n=$($PY -c "import json;d=json.load(open('$out'));print(d['aggregate']['n_parsed'])" 2>/dev/null || echo 0)
    fi
    if [[ "$done_n" -ge "$JUDGE_N" ]]; then
      echo "  [skip] judge $d vs $base_tag already has $done_n judgments"
      continue
    fi
    echo "  judging $d vs $base_tag (n=$JUDGE_N, conc=$JUDGE_C) ..."
    $PY dsl_llada/judge_summarization.py \
      --dataset "$d" --nfe "$NFE" \
      --a b1_sde --b "$base_tag" \
      --n "$JUDGE_N" --concurrency "$JUDGE_C" --seed 42 2>&1 | tail -5
  done
done

# -------- 4. Summary table --------
echo ""
echo "$STAMP === 4/4  Aggregate table ==="
$PY dsl_llada/summarize_results_table.py \
  --csv "$SUM_DIR/stage1_nfe${NFE}_summary.csv" 2>&1 | tail -80
echo ""
echo "$STAMP DONE. CSV -> $SUM_DIR/stage1_nfe${NFE}_summary.csv"
