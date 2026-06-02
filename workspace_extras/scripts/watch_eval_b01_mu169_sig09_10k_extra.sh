#!/usr/bin/env bash
# Extra post-10k checks. This waits for the primary post-10k watcher to finish,
# then runs a small XSum NFE sweep and writes a compact comparison summary.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_b01_mu169_sig09_trainable_10k_8gpu_20260511_2337}
MODEL_KEY=${MODEL_KEY:-uu_b01_mu169_sig09_10k}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}
EVAL_GPU=${EVAL_GPU:-0}
PRIMARY_WATCH_LOG="${LOG_DIR}/${RUN_ID}_post10k_watch.log"
EXTRA_LOG="${LOG_DIR}/${RUN_ID}_post10k_extra_eval.log"
OUT_TAG="${RUN_ID}_checkpoint10000_sweep100"
SUMMARY="${REPO_ROOT}/eval_results/summarization/${RUN_ID}_checkpoint10000_xsum_summary.md"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/eval_results/summarization"

{
  echo "[extra] run_id=${RUN_ID}"
  echo "[extra] waiting for primary watcher done marker in ${PRIMARY_WATCH_LOG}"
  echo "[extra] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${EXTRA_LOG}"

while true; do
  if [ -f "${PRIMARY_WATCH_LOG}" ] && grep -q " done$" "${PRIMARY_WATCH_LOG}"; then
    break
  fi
  date -u "+[extra] %Y-%m-%dT%H:%M:%SZ waiting for primary eval" | tee -a "${EXTRA_LOG}"
  sleep 300
done

source "${VENV}/bin/activate"
cd "${REPO_ROOT}"

for nfe in 8 16 32; do
  date -u "+[extra] %Y-%m-%dT%H:%M:%SZ xsum sde nfe=${nfe}" | tee -a "${EXTRA_LOG}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${VENV}/bin/python" dsl_llada/eval_summarization.py \
    --dataset xsum \
    --method sde \
    --model_key "${MODEL_KEY}" \
    --nfe "${nfe}" \
    --gpu 0 \
    --out_tag "${OUT_TAG}" \
    2>&1 | tee -a "${EXTRA_LOG}"
done

"${VENV}/bin/python" - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
model_key = "${MODEL_KEY}"
out_tag = "${OUT_TAG}"
summary = Path("${SUMMARY}")
root = Path("eval_results/summarization")

rows = []
for nfe in [8, 16, 32, 64]:
    if nfe == 64:
        path = root / f"xsum_{model_key}_sde_{run_id}_checkpoint10000_nfe64.json"
    else:
        path = root / f"xsum_{model_key}_sde_{out_tag}_nfe{nfe}.json"
    if not path.exists():
        rows.append((nfe, None, None, None, None, None, str(path)))
        continue
    d = json.loads(path.read_text())
    rows.append((
        nfe,
        d.get("n_samples_here"),
        d.get("rouge1"),
        d.get("rouge2"),
        d.get("rougeL"),
        d.get("avg_words"),
        str(path),
    ))

base_path = root / "xsum_uu_trainable_1k_sde_cmp_train_nfe64.json"
base = json.loads(base_path.read_text()) if base_path.exists() else None

lines = [
    f"# XSum Post-10k Summary: {run_id}",
    "",
    "## 10k XSum SDE Sweep",
    "",
    "| NFE | n | R-1 | R-2 | R-L | avg words | artifact |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for nfe, n, r1, r2, rl, words, path in rows:
    if n is None:
        lines.append(f"| {nfe} | missing |  |  |  |  | `{path}` |")
    else:
        lines.append(f"| {nfe} | {n} | {r1:.2f} | {r2:.2f} | {rl:.2f} | {words:.2f} | `{path}` |")

lines += ["", "## Comparable 100-sample Baseline", ""]
if base:
    lines += [
        "| Model | NFE | n | R-1 | R-2 | R-L | avg words | artifact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| uu_trainable_1k | 64 | {base.get('n_samples_here')} | {base.get('rouge1'):.2f} | {base.get('rouge2'):.2f} | {base.get('rougeL'):.2f} | {base.get('avg_words'):.2f} | `{base_path}` |",
    ]
else:
    lines.append(f"Missing comparable baseline: `{base_path}`")

summary.write_text("\\n".join(lines) + "\\n")
print(f"[summary] wrote {summary}")
PY

date -u "+[extra] %Y-%m-%dT%H:%M:%SZ done" | tee -a "${EXTRA_LOG}"

