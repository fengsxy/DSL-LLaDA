#!/usr/bin/env bash
# Wait for all post-10k checks, then write a compact report with metrics and
# the next hyperparameter decision.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/efs/RMDM}
VENV=${VENV:-${REPO_ROOT}/.venv}
RUN_ID=${RUN_ID:-unit_uniform_b01_mu169_sig09_trainable_10k_8gpu_20260511_2337}
MODEL_KEY=${MODEL_KEY:-uu_b01_mu169_sig09_10k}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/unit_uniform}
ANALYSIS_DIR=${ANALYSIS_DIR:-${REPO_ROOT}/eval_results/embedding_analysis}
EXTRA_LOG="${LOG_DIR}/${RUN_ID}_post10k_extra_eval.log"
TRAIN_LOG="${LOG_DIR}/${RUN_ID}_train.log"
EMBED_JSON="${ANALYSIS_DIR}/${RUN_ID}_checkpoint10000_embedding.json"
REPORT="${REPO_ROOT}/docs/plans/${RUN_ID}_post10k_eval_report.md"
WATCH_LOG="${LOG_DIR}/${RUN_ID}_post10k_report_watch.log"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/docs/plans"

{
  echo "[report] run_id=${RUN_ID}"
  echo "[report] waiting for ${EXTRA_LOG}"
  echo "[report] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${WATCH_LOG}"

while true; do
  if [ -f "${EXTRA_LOG}" ] && grep -q " done$" "${EXTRA_LOG}"; then
    break
  fi
  date -u "+[report] %Y-%m-%dT%H:%M:%SZ waiting for extra eval" | tee -a "${WATCH_LOG}"
  sleep 300
done

cd "${REPO_ROOT}"

"${VENV}/bin/python" - <<PY
import json
import re
from pathlib import Path

run_id = "${RUN_ID}"
model_key = "${MODEL_KEY}"
train_log = Path("${TRAIN_LOG}")
embed_json = Path("${EMBED_JSON}")
report = Path("${REPORT}")
sum_root = Path("eval_results/summarization")

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None

rows = []
for nfe in [8, 16, 32, 64]:
    if nfe == 64:
        path = sum_root / f"xsum_{model_key}_sde_{run_id}_checkpoint10000_nfe64.json"
    else:
        path = sum_root / f"xsum_{model_key}_sde_{run_id}_checkpoint10000_sweep100_nfe{nfe}.json"
    d = load_json(path)
    rows.append((nfe, path, d))

baseline_path = sum_root / "xsum_uu_trainable_1k_sde_cmp_train_nfe64.json"
baseline = load_json(baseline_path)
embed = load_json(embed_json)

last_dsl = None
last_health = None
if train_log.exists():
    text = train_log.read_text(errors="replace")
    dsl_matches = list(re.finditer(
        r"\\[DSL step=(\\d+)\\] acc=([0-9.]+), cos\\(h,wte\\)=([\\-0-9.]+), beta=([0-9.]+), mask_bias=([0-9.]+), snr_max=([0-9.]+)",
        text,
    ))
    if dsl_matches:
        m = dsl_matches[-1]
        last_dsl = {
            "step": int(m.group(1)),
            "acc": float(m.group(2)),
            "cos_h_wte": float(m.group(3)),
            "beta": float(m.group(4)),
            "mask_bias": float(m.group(5)),
            "snr_max": float(m.group(6)),
        }
    health_matches = list(re.finditer(
        r"\\[EmbedHealth step=(\\d+)\\] embed/norm_mean=([0-9.]+), embed/norm_std=([0-9.]+), embed/norm_cv=([0-9.]+).*?embed/effective_rank=([0-9.]+).*?converter/effective_rank=([0-9.]+)",
        text,
    ))
    if health_matches:
        m = health_matches[-1]
        last_health = {
            "step": int(m.group(1)),
            "norm_mean": float(m.group(2)),
            "norm_std": float(m.group(3)),
            "norm_cv": float(m.group(4)),
            "embed_rank": float(m.group(5)),
            "converter_rank": float(m.group(6)),
        }

lines = [
    f"# Post-10k Evaluation Report: {run_id}",
    "",
    "## Status",
    "",
    f"- Model key: `{model_key}`",
    f"- Training log: `{train_log}`",
    f"- Embedding analysis: `{embed_json}`",
    "",
    "## XSum SDE",
    "",
    "| NFE | n | R-1 | R-2 | R-L | avg words | degenerate % | artifact |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for nfe, path, d in rows:
    if d is None:
        lines.append(f"| {nfe} | missing |  |  |  |  |  | `{path}` |")
    else:
        lines.append(
            f"| {nfe} | {d.get('n_samples_here')} | {d.get('rouge1'):.2f} | "
            f"{d.get('rouge2'):.2f} | {d.get('rougeL'):.2f} | "
            f"{d.get('avg_words'):.2f} | {d.get('degenerate_pct'):.2f} | `{path}` |"
        )

lines += ["", "## Comparable 1k Baseline", ""]
if baseline:
    lines += [
        "| Model | NFE | n | R-1 | R-2 | R-L | avg words | artifact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| uu_trainable_1k | 64 | {baseline.get('n_samples_here')} | {baseline.get('rouge1'):.2f} | "
        f"{baseline.get('rouge2'):.2f} | {baseline.get('rougeL'):.2f} | "
        f"{baseline.get('avg_words'):.2f} | `{baseline_path}` |",
    ]
else:
    lines.append(f"Missing baseline: `{baseline_path}`")

lines += ["", "## Embedding Structure", ""]
if embed:
    m = embed.get("metrics", {})
    top10 = float(m.get("top10_overlap_vs_wte", 0.0))
    pair_corr = float(m.get("pair_cos_corr_vs_wte", 0.0))
    semantic_score = 1.0 + 9.0 * max(0.0, min(1.0, top10))
    lines += [
        f"- Effective rank sample: `{m.get('dsl/effective_rank_sample')}`",
        f"- Pair cosine correlation vs WTE: `{pair_corr:.6f}`",
        f"- Top-10 neighbor overlap vs WTE: `{top10:.6f}`",
        f"- Heuristic semantic score, 1-10: `{semantic_score:.2f}`. This is defined as `1 + 9 * top10_overlap_vs_wte`, so values near 1 mean little WTE-like semantic neighborhood structure.",
    ]
else:
    lines.append(f"Missing embedding analysis: `{embed_json}`")

lines += ["", "## Final Training Diagnostics", ""]
if last_dsl:
    lines.append(
        f"- Last DSL diagnostic: step `{last_dsl['step']}`, acc `{last_dsl['acc']:.4f}`, "
        f"cos(h,wte) `{last_dsl['cos_h_wte']:.4f}`, beta `{last_dsl['beta']:.4f}`, "
        f"snr_max `{last_dsl['snr_max']:.1f}`."
    )
else:
    lines.append("- Last DSL diagnostic not found.")
if last_health:
    lines.append(
        f"- Last embedding health: step `{last_health['step']}`, norm CV `{last_health['norm_cv']:.4f}`, "
        f"embedding rank `{last_health['embed_rank']:.2f}`, converter rank `{last_health['converter_rank']:.2f}`."
    )
else:
    lines.append("- Last embedding health diagnostic not found.")

lines += ["", "## Decision Rule", ""]
best64 = rows[-1][2]
if best64 and baseline:
    delta = best64.get("rouge1", 0) - baseline.get("rouge1", 0)
    lines.append(f"- XSum NFE=64 R-1 delta vs 1k 100-sample baseline: `{delta:.2f}`.")
    if delta >= -0.5:
        lines.append("- Recommendation: consider this run viable for broader evaluation; run CNN/DM and inspect samples before scaling further.")
    else:
        lines.append("- Recommendation: do not scale this exact setting yet. Run 100/1000-step probes with larger beta init (`0.5` or `1.0`) and/or a narrower higher-SNR distribution.")
else:
    lines.append("- Recommendation pending because XSum or baseline metrics are missing.")

report.write_text("\\n".join(lines) + "\\n")
print(f"[report] wrote {report}")
PY

date -u "+[report] %Y-%m-%dT%H:%M:%SZ done" | tee -a "${WATCH_LOG}"

