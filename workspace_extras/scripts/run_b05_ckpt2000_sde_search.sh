#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
VENV=${VENV:-${REPO_ROOT}/.venv}
MODEL_KEY=${MODEL_KEY:-probe_unit_uniform_b05_mu322_sig06_snrln80_1k_beta_snr_matched_20260512_225231_ckpt2000}
LIMIT=${LIMIT:-8}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/sde_param_search}
OUT_SUMMARY=${OUT_SUMMARY:-${REPO_ROOT}/eval_results/summarization/b05_ckpt2000_sde_search_summary.json}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results/summarization"
cd "${REPO_ROOT}"

run_eval() {
  local tag=$1
  local gpu=$2
  local beta=$3
  local snr_min=$4
  local snr_max=$5
  local gen_len=$6
  local nfe=$7
  local noise=$8
  local log_file="${LOG_DIR}/b05_ckpt2000_${tag}.log"

  echo "[b05-2k] gpu=${gpu} tag=${tag} beta=${beta} snr=${snr_min}..${snr_max} gen=${gen_len} nfe=${nfe} noise=${noise}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${VENV}/bin/python" dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "${MODEL_KEY}" \
    --gpu 0 \
    --nfe "${nfe}" \
    --gen_length "${gen_len}" \
    --limit "${LIMIT}" \
    --sde_beta_infer "${beta}" \
    --sde_noise_scale "${noise}" \
    --sde_schedule sensitive \
    --sde_snr_min "${snr_min}" \
    --sde_snr_max "${snr_max}" \
    --sde_sensitive_low 7 \
    --sde_sensitive_high 74 \
    --sde_top_k 512 \
    --out_tag "b05_ckpt2000_${tag}" \
    > "${log_file}" 2>&1
}

run_batch4() {
  run_eval "$1" 4 "$2" "$3" "$4" "$5" "$6" "$7" &
  run_eval "$8" 5 "$9" "${10}" "${11}" "${12}" "${13}" "${14}" &
  run_eval "${15}" 6 "${16}" "${17}" "${18}" "${19}" "${20}" "${21}" &
  run_eval "${22}" 7 "${23}" "${24}" "${25}" "${26}" "${27}" "${28}" &
  wait
}

run_batch4 \
  b0p5_snr1_100_g96_n32_ns0   0.5 1 100 96 32 0.0 \
  b1_snr1_100_g96_n32_ns0     1.0 1 100 96 32 0.0 \
  b1p5_snr1_100_g96_n32_ns0   1.5 1 100 96 32 0.0 \
  b2_snr1_100_g96_n32_ns0     2.0 1 100 96 32 0.0

run_batch4 \
  b0p5_snr0p5_80_g96_n32_ns0  0.5 0.5 80 96 32 0.0 \
  b1_snr0p5_80_g96_n32_ns0    1.0 0.5 80 96 32 0.0 \
  b1_snr2_100_g96_n32_ns0     1.0 2 100 96 32 0.0 \
  b1p5_snr2_100_g96_n32_ns0   1.5 2 100 96 32 0.0

run_batch4 \
  b1_snr1_100_g64_n32_ns0     1.0 1 100 64 32 0.0 \
  b1_snr1_100_g128_n32_ns0    1.0 1 100 128 32 0.0 \
  b1_snr1_100_g96_n16_ns0     1.0 1 100 96 16 0.0 \
  b1_snr1_100_g96_n64_ns0     1.0 1 100 96 64 0.0

"${VENV}/bin/python" - <<'PY'
import glob
import json
from pathlib import Path

rows = []
for fname in glob.glob("eval_results/summarization/xsum_*b05_ckpt2000_*.json"):
    path = Path(fname)
    obj = json.load(open(path))
    rows.append({
        "file": path.name,
        "rouge1": obj.get("rouge1"),
        "rouge2": obj.get("rouge2"),
        "rougeL": obj.get("rougeL"),
        "avg_words": obj.get("avg_words"),
        "degenerate_pct": obj.get("degenerate_pct"),
        "sde_params": obj.get("sde_params"),
        "gen_length": obj.get("gen_length"),
        "nfe": obj.get("nfe"),
    })
rows.sort(key=lambda r: (r["rouge1"] or 0), reverse=True)
out = Path("eval_results/summarization/b05_ckpt2000_sde_search_summary.json")
out.write_text(json.dumps(rows, indent=2) + "\n")
print(f"[b05-2k] wrote {out}")
for r in rows[:12]:
    s = r.get("sde_params") or {}
    print(
        f"{r['rouge1']:.2f}\t{r['rouge2']:.2f}\t{r['rougeL']:.2f}\t"
        f"words={r['avg_words']:.1f}\tdeg={r['degenerate_pct']}\t"
        f"beta={s.get('beta_infer')} snr={s.get('snr_min')}..{s.get('snr_max')} "
        f"gen={r['gen_length']} nfe={r['nfe']} noise={s.get('noise_scale')}\t{r['file']}"
    )
PY

echo "[b05-2k] complete"
