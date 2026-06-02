#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
VENV=${VENV:-${REPO_ROOT}/.venv}
PAIR_ID=${PAIR_ID:-beta_snr_matched_20260512_225231}
RUN_ID=${RUN_ID:-unit_uniform_b05_mu322_sig06_snrln80_1k_${PAIR_ID}}
STEPS=${STEPS:-5000}
MODEL_KEY=${MODEL_KEY:-probe_${RUN_ID}_ckpt${STEPS}}
CKPT_DIR=${CKPT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}/checkpoint-${STEPS}}
LIMIT=${LIMIT:-8}
NFE=${NFE:-32}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/sde_param_search}
WAIT_SCREEN=${WAIT_SCREEN:-b05_scale_5k}

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results" "${REPO_ROOT}/eval_results/summarization"

echo "[b05-search] waiting for ${CKPT_DIR}"
until [ -f "${CKPT_DIR}/model.safetensors.index.json" ] || [ -f "${CKPT_DIR}/pytorch_model.bin" ]; do
  sleep 60
done

if command -v screen >/dev/null 2>&1 && command -v rg >/dev/null 2>&1; then
  echo "[b05-search] checkpoint exists; waiting for screen ${WAIT_SCREEN} to finish its built-in sanity eval"
  while screen -ls | rg -q "${WAIT_SCREEN}"; do
    sleep 60
  done
fi

cd "${REPO_ROOT}"

"${VENV}/bin/python" - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
steps = int("${STEPS}")
key = "${MODEL_KEY}"
path = Path("eval_results/registry.json")
reg = json.loads(path.read_text()) if path.exists() else {}
reg[key] = {
    "path": f"checkpoints/{run_id}/checkpoint-{steps}",
    "type": "local",
    "description": "b05 beta/SNR-matched unit-uniform trainable DSL scale checkpoint",
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
    },
}
path.write_text(json.dumps(reg, indent=2) + "\n")
print(f"[registry] updated {key}")
PY

run_eval() {
  local tag=$1
  local beta=$2
  local snr_min=$3
  local snr_max=$4
  local gen_len=$5
  local gpu=$6
  local log_file="${LOG_DIR}/b05_ckpt${STEPS}_${tag}.log"

  echo "[b05-search] gpu=${gpu} tag=${tag} beta=${beta} snr=${snr_min}..${snr_max} gen=${gen_len}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${VENV}/bin/python" dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "${MODEL_KEY}" \
    --gpu 0 \
    --nfe "${NFE}" \
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
    --out_tag "b05_ckpt${STEPS}_${tag}" \
    2>&1 | tee "${log_file}"
}

run_eval b0p75_snr1_80_g128 0.75 1 80 128 0 &
run_eval b1p0_snr1_80_g128 1.0 1 80 128 1 &
run_eval b1p0_snr1_100_g128 1.0 1 100 128 2 &
run_eval b1p25_snr1_100_g128 1.25 1 100 128 3 &
run_eval b1p5_snr1_100_g128 1.5 1 100 128 4 &
run_eval b2p0_snr1_100_g128 2.0 1 100 128 5 &
run_eval b1p0_snr0p5_100_g128 1.0 0.5 100 128 6 &
run_eval b1p0_snr1_150_g128 1.0 1 150 128 7 &
wait

run_eval b1p25_snr1_150_g128 1.25 1 150 128 0 &
run_eval b1p0_snr1_100_g96 1.0 1 100 96 1 &
wait

"${VENV}/bin/python" - <<'PY'
import glob
import json
from pathlib import Path

rows = []
for fname in glob.glob("eval_results/summarization/xsum_*b05_ckpt5000_*.json"):
    path = Path(fname)
    with path.open() as f:
        obj = json.load(f)
    rows.append({
        "file": path.name,
        "rouge1": obj.get("rouge1"),
        "rouge2": obj.get("rouge2"),
        "rougeL": obj.get("rougeL"),
        "avg_words": obj.get("avg_words"),
        "degenerate_pct": obj.get("degenerate_pct"),
        "sde_params": obj.get("sde_params"),
        "gen_length": obj.get("gen_length"),
    })
rows.sort(key=lambda r: (r["rouge1"] or 0), reverse=True)
out = Path("eval_results/summarization/b05_ckpt5000_sde_search_summary.json")
out.write_text(json.dumps(rows, indent=2) + "\n")
print(f"[b05-search] wrote {out}")
for r in rows[:10]:
    params = r["sde_params"] or {}
    print(
        f"{r['rouge1']:.2f}\t{r['rouge2']:.2f}\t{r['rougeL']:.2f}\t"
        f"words={r['avg_words']:.1f}\tdeg={r['degenerate_pct']}\t"
        f"beta={params.get('beta_infer')} snr={params.get('snr_min')}..{params.get('snr_max')} "
        f"gen={r['gen_length']}\t{r['file']}"
    )
PY

echo "[b05-search] complete"
