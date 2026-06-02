#!/usr/bin/env bash
set -u

export HF_HOME=/data2/ylong030/huggingface
export TRANSFORMERS_CACHE=/data2/ylong030/huggingface/hub

mkdir -p logs/sum_variant_data1000

get_data_file() {
  case "$1" in
    xsum) echo "eval_data/xsum_1000.json" ;;
    cnn_dailymail) echo "eval_data/cnn_dailymail_1000.json" ;;
    pubmed) echo "eval_data/pubmed_1000.json" ;;
    arxiv) echo "eval_data/arxiv_1000.json" ;;
  esac
}

is_complete_json() {
  local out="$1"
  [ -f "$out" ] || return 1
  python - "$out" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
raise SystemExit(0 if d.get("n_samples_total") == 1000 and d.get("n_samples_here") == 1000 else 1)
PY
}

run_sharded_sum() {
  local ds="$1"
  local nfe="$2"
  local method_tag="$3"
  shift 3

  local out="eval_results/summarization/${ds}_${method_tag}_nfe${nfe}.json"
  if is_complete_json "$out"; then
    echo "SKIP ${ds} ${method_tag} nfe${nfe}"
    return 0
  fi

  local data_file
  data_file="$(get_data_file "$ds")"
  local logdir="logs/sum_variant_data1000/${method_tag}/${ds}/nfe${nfe}"
  mkdir -p "$logdir"
  echo "START ${ds} ${method_tag} nfe${nfe} data=${data_file}"

  local pids=""
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$gpu" python dsl_llada/eval_summarization.py \
      --dataset "$ds" \
      --data_file "$data_file" \
      --nfe "$nfe" \
      --gpu 0 \
      --limit 1000 \
      --shard_id "$gpu" \
      --num_shards 8 \
      "$@" > "$logdir/shard${gpu}.log" 2>&1 &
    pids="$pids $!"
  done

  local fail=0
  local pid
  for pid in $pids; do
    wait "$pid" || fail=1
  done
  if [ "$fail" -ne 0 ]; then
    echo "FAIL ${ds} ${method_tag} nfe${nfe}; see $logdir"
    exit 1
  fi

  python dsl_llada/merge_summarization_shards.py \
    --dataset "$ds" \
    --method_tag "$method_tag" \
    --nfe "$nfe"
}

for ds in xsum cnn_dailymail pubmed arxiv; do
  for nfe in 8 16 32 64; do
    run_sharded_sum "$ds" "$nfe" "hf_beta1_sde_beta1_snrScaled_noNoiseScale_data1000" \
      --model_key hf_beta1 \
      --method sde \
      --out_tag beta1_snrScaled_noNoiseScale_data1000 \
      --sde_beta_infer 1.0 \
      --sde_snr_min 0.01984375 \
      --sde_sensitive_low 13.890625 \
      --sde_sensitive_high 146.84375 \
      --sde_snr_max 198.4375
  done
done

for ds in xsum cnn_dailymail pubmed arxiv; do
  for nfe in 8 16 32 64; do
    run_sharded_sum "$ds" "$nfe" "original_rmdm_data1000" \
      --model_key original \
      --method rmdm \
      --out_tag data1000
  done
done
