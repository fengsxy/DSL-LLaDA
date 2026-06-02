#!/usr/bin/env bash
# Wait for the beta=0.1, mu=1.69/sigma=0.9 10k run to finish, then run
# the first agreed checks: XSum SDE and DSL embedding structure.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_b01_mu169_sig09_trainable_10k_8gpu_20260511_2337}
MODEL_KEY=${MODEL_KEY:-uu_b01_mu169_sig09_10k}
TRAIN_SCREEN=${TRAIN_SCREEN:-uu_b01_mu169_10k_8gpu}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_ID}}
CKPT="${OUTPUT_DIR}/checkpoint-10000"
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}
ANALYSIS_DIR=${ANALYSIS_DIR:-${REPO_ROOT}/eval_results/embedding_analysis}
BASELINE_CKPT=${BASELINE_CKPT:-${REPO_ROOT}/checkpoints/unit_uniform_paper_b1_trainable_1k_20260508_0320/checkpoint-1000}
EVAL_GPU=${EVAL_GPU:-0}

mkdir -p "${LOG_DIR}" "${ANALYSIS_DIR}" "${REPO_ROOT}/eval_results"

WATCH_LOG="${LOG_DIR}/${RUN_ID}_post10k_watch.log"
XSUM_LOG="${LOG_DIR}/${RUN_ID}_checkpoint10000_xsum_sde_nfe64.log"
EMBED_LOG="${LOG_DIR}/${RUN_ID}_checkpoint10000_embedding_analysis.log"

{
  echo "[watch] run_id=${RUN_ID}"
  echo "[watch] checkpoint=${CKPT}"
  echo "[watch] training_screen=${TRAIN_SCREEN}"
  echo "[watch] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${WATCH_LOG}"

while true; do
  if [ -d "${CKPT}" ] && ! screen -ls | grep -q "${TRAIN_SCREEN}"; then
    break
  fi
  date -u "+[watch] %Y-%m-%dT%H:%M:%SZ waiting for checkpoint/train exit" | tee -a "${WATCH_LOG}"
  sleep 300
done

date -u "+[watch] %Y-%m-%dT%H:%M:%SZ checkpoint ready; starting eval" | tee -a "${WATCH_LOG}"

source "${VENV}/bin/activate"
cd "${REPO_ROOT}"

python - <<PY
import json
from pathlib import Path

path = Path("eval_results/registry.json")
reg = json.loads(path.read_text()) if path.exists() else {}
reg["${MODEL_KEY}"] = {
    "path": "checkpoints/${RUN_ID}/checkpoint-10000",
    "type": "local",
    "description": "Remote DSL unit-uniform beta_init=0.1, trainable unit-norm noise embedding, d=100, mu=1.69 sigma=0.9, 10K steps, 8 GPU",
    "dsl": True,
    "dsl_config": {
        "beta_init": 0.1,
        "noise_dim": 100,
        "noise_init": "random_unit_uniform",
        "train_noise_embed": True,
        "snr_mu": 1.69,
        "snr_sigma": 0.9,
        "snr_max": 100,
        "snr_max_ln": 40,
    },
    "sde_config": {
        "beta_infer": 2.0,
        "noise_scale": 0.05,
        "schedule": [3, 100],
        "steps": 32,
        "solver": "heun",
    },
    "gen_methods": ["sde", "remask_free", "remask_suppress_block32"],
}
path.write_text(json.dumps(reg, indent=2) + "\\n")
print("[registry] updated ${MODEL_KEY} -> checkpoints/${RUN_ID}/checkpoint-10000")
PY

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${VENV}/bin/python" dsl_llada/eval_summarization.py \
  --dataset xsum \
  --method sde \
  --model_key "${MODEL_KEY}" \
  --nfe 64 \
  --gpu 0 \
  --out_tag "${RUN_ID}_checkpoint10000" \
  2>&1 | tee "${XSUM_LOG}"

"${VENV}/bin/python" dsl_llada/analyze_dsl_embedding_structure.py \
  --checkpoint "${CKPT}" \
  --baseline-checkpoint "${BASELINE_CKPT}" \
  --output "${ANALYSIS_DIR}/${RUN_ID}_checkpoint10000_embedding.json" \
  2>&1 | tee "${EMBED_LOG}"

date -u "+[watch] %Y-%m-%dT%H:%M:%SZ done" | tee -a "${WATCH_LOG}"
