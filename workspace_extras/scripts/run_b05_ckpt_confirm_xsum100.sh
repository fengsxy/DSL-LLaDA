#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
VENV=${VENV:-${REPO_ROOT}/.venv}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/sde_param_search}
SUMMARY=${SUMMARY:-${REPO_ROOT}/eval_results/summarization/b05_ckpt_confirm_xsum100_summary.json}
LIMIT=${LIMIT:-100}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results/summarization"
cd "${REPO_ROOT}"

BASE_KEY=probe_unit_uniform_b05_mu322_sig06_snrln80_1k_beta_snr_matched_20260512_225231

run_eval() {
  local tag=$1
  local key=$2
  local gpu=$3
  local beta=$4
  local snr_min=$5
  local snr_max=$6
  local gen_len=$7
  local nfe=$8
  local log_file="${LOG_DIR}/b05_confirm_${tag}.log"

  echo "[b05-confirm] gpu=${gpu} tag=${tag} key=${key} beta=${beta} snr=${snr_min}..${snr_max} gen=${gen_len} nfe=${nfe}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${VENV}/bin/python" dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "${key}" \
    --gpu 0 \
    --nfe "${nfe}" \
    --gen_length "${gen_len}" \
    --limit "${LIMIT}" \
    --sde_beta_infer "${beta}" \
    --sde_noise_scale 0.0 \
    --sde_schedule sensitive \
    --sde_snr_min "${snr_min}" \
    --sde_snr_max "${snr_max}" \
    --sde_sensitive_low 7 \
    --sde_sensitive_high 74 \
    --sde_top_k 512 \
    --out_tag "b05_confirm_${tag}" \
    > "${log_file}" 2>&1
}

run_eval ckpt1000_b1_snr1_100_g96_n32 "${BASE_KEY}" 0 1.0 1 100 96 32 &
run_eval ckpt2000_b1_snr1_100_g96_n32 "${BASE_KEY}_ckpt2000" 1 1.0 1 100 96 32 &
run_eval ckpt4000_b1_snr1_100_g96_n32 "${BASE_KEY}_ckpt4000" 2 1.0 1 100 96 32 &
run_eval ckpt5000_b2_snr3_100_g96_n64 "${BASE_KEY}_ckpt5000" 3 2.0 3 100 96 64 &
wait

"${VENV}/bin/python" - <<'PY'
import glob
import json
from pathlib import Path

rows = []
for fname in glob.glob("eval_results/summarization/xsum_*b05_confirm_*.json"):
    path = Path(fname)
    obj = json.load(open(path))
    rows.append({
        "file": path.name,
        "model_key": obj.get("model_key"),
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
out = Path("eval_results/summarization/b05_ckpt_confirm_xsum100_summary.json")
out.write_text(json.dumps(rows, indent=2) + "\n")
print(f"[b05-confirm] wrote {out}")
for r in rows:
    s = r.get("sde_params") or {}
    print(
        f"{r['model_key']}\tR1={r['rouge1']:.2f}\tR2={r['rouge2']:.2f}\t"
        f"RL={r['rougeL']:.2f}\twords={r['avg_words']:.1f}\tdeg={r['degenerate_pct']}\t"
        f"beta={s.get('beta_infer')} snr={s.get('snr_min')}..{s.get('snr_max')} "
        f"gen={r['gen_length']} nfe={r['nfe']}"
    )
PY
