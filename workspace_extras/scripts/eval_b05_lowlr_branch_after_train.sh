#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_b05_mu322_sig06_snrln80_from2k_lowlr5e6_cscale5_to3k_20260513_1848}
SCREEN_NAME=${SCREEN_NAME:-b05_lowlr_from2k_to3k_fastskip}
LIMIT=${LIMIT:-8}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/sde_param_search}
SUMMARY=${SUMMARY:-${REPO_ROOT}/eval_results/summarization/${RUN_ID}_xsum_sde_summary.json}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results/summarization"
cd "${REPO_ROOT}"

echo "[b05-lowlr-eval] waiting for training screen ${SCREEN_NAME}"
while screen -ls | grep -q "${SCREEN_NAME}"; do
  sleep 60
done

echo "[b05-lowlr-eval] training screen ended; registering checkpoints"
"${VENV}/bin/python" - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
reg_path = Path("eval_results/registry.json")
reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
for step in (2500, 3000):
    ckpt = Path("checkpoints") / run_id / f"checkpoint-{step}"
    if not ckpt.exists():
        print(f"[registry] missing {ckpt}; skip")
        continue
    reg[f"probe_{run_id}_ckpt{step}"] = {
        "path": f"checkpoints/{run_id}/checkpoint-{step}",
        "type": "local",
        "description": "b05 checkpoint-2000 conservative continuation branch",
        "dsl": True,
        "dsl_config": {
            "beta_init": 0.5,
            "noise_dim": 100,
            "noise_init": "random_unit_uniform",
            "train_noise_embed": True,
            "snr_mu": 3.22,
            "snr_sigma": 0.6,
            "snr_max": 100.0,
            "snr_max_ln": 80.0,
            "converter_lr_scale": 5.0,
        },
    }
reg_path.write_text(json.dumps(reg, indent=2) + "\n")
print("[registry] updated conservative continuation checkpoints")
PY

run_eval() {
  local step=$1
  local nfe=$2
  local gpu=$3
  local key="probe_${RUN_ID}_ckpt${step}"
  local tag="${RUN_ID}_ckpt${step}_b1_snr1_100_g96_n${nfe}"
  local log_file="${LOG_DIR}/${tag}.log"
  echo "[b05-lowlr-eval] step=${step} nfe=${nfe} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${VENV}/bin/python" dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "${key}" \
    --gpu 0 \
    --nfe "${nfe}" \
    --gen_length 96 \
    --limit "${LIMIT}" \
    --sde_beta_infer 1.0 \
    --sde_noise_scale 0.0 \
    --sde_schedule sensitive \
    --sde_snr_min 1 \
    --sde_snr_max 100 \
    --sde_sensitive_low 7 \
    --sde_sensitive_high 74 \
    --sde_top_k 512 \
    --out_tag "${tag}" \
    > "${log_file}" 2>&1
}

run_eval 2500 32 0 &
run_eval 3000 32 1 &
run_eval 3000 64 2 &
wait

"${VENV}/bin/python" - <<PY
import glob
import json
from pathlib import Path

run_id = "${RUN_ID}"
rows = []
for fname in glob.glob(f"eval_results/summarization/xsum_*{run_id}_ckpt*_b1_snr1_100_g96_n*.json"):
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
out = Path("${SUMMARY}")
out.write_text(json.dumps(rows, indent=2) + "\n")
print(f"[b05-lowlr-eval] wrote {out}")
for r in rows:
    s = r.get("sde_params") or {}
    print(
        f"{r['model_key']} R1={r['rouge1']:.2f} R2={r['rouge2']:.2f} "
        f"RL={r['rougeL']:.2f} words={r['avg_words']:.1f} "
        f"deg={r['degenerate_pct']} beta={s.get('beta_infer')} "
        f"snr={s.get('snr_min')}..{s.get('snr_max')} nfe={r['nfe']}"
    )
PY

echo "[b05-lowlr-eval] complete"
