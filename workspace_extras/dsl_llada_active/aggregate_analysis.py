"""Aggregate per-dataset case analysis + LLM-judge results into a single report.

Reads: eval_results/summarization/analysis/{dataset}_nfe{NFE}__{main}_vs_{base}.json
       eval_results/summarization/analysis/judge__{dataset}_nfe{NFE}__{main}_vs_{base}.json

Writes: eval_results/summarization/analysis/_aggregate_nfe{NFE}.md
        eval_results/summarization/analysis/_aggregate_nfe{NFE}.json

Produces two main tables:
  (A) Case-analysis bucket distribution + failure modes per (dataset, comparison)
  (B) LLM-judge win rates + 4-axis means per (dataset, comparison)

Usage:
    python dsl_llada/aggregate_analysis.py --nfe 64
"""
import argparse
import glob
import json
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(_root, "eval_results", "summarization", "analysis")

DATASETS = ["xsum", "cnn_dailymail", "pubmed", "arxiv", "billsum"]
COMPARISONS = [
    ("b1_sde", "original_remask",            "SDE vs LLaDA-default"),
    ("b1_sde", "original_remask_eosInf_b32", "SDE vs LLaDA+EOS+Block"),
]


def load_case(dataset, nfe, main, base):
    p = os.path.join(ANA, f"{dataset}_nfe{nfe}__{main}_vs_{base}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_judge(dataset, nfe, main, base):
    p = os.path.join(ANA, f"judge__{dataset}_nfe{nfe}__{main}_vs_{base}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfe", type=int, default=64)
    args = ap.parse_args()
    nfe = args.nfe

    # ---------- collect ----------
    rows_case = []
    rows_judge = []
    for ds in DATASETS:
        for main, base, label in COMPARISONS:
            c = load_case(ds, nfe, main, base)
            if c:
                s = c["summary"]
                agg = s["aggregate"]
                bp = s["bucket_pct"]
                fm_m = s["failure_modes_main"]
                fm_b = s["failure_modes_base"]
                ln = s["length"]
                rows_case.append({
                    "dataset": ds, "comparison": label,
                    "main_r1": agg["main_r1"], "base_r1": agg["base_r1"],
                    "delta_r1": agg["mean_delta_r1"],
                    "main_r2": agg["main_r2"], "base_r2": agg["base_r2"],
                    "main_rl": agg["main_rl"], "base_rl": agg["base_rl"],
                    "win_big": bp.get("big_win", 0), "win": bp.get("win", 0),
                    "tie": bp.get("tie", 0),
                    "loss": bp.get("loss", 0), "loss_big": bp.get("big_loss", 0),
                    "main_preos": fm_m.get("premature_eos", 0),
                    "main_rep":   fm_m.get("repetition", 0),
                    "main_degen": fm_m.get("degen_ending", 0),
                    "base_preos": fm_b.get("premature_eos", 0),
                    "base_rep":   fm_b.get("repetition", 0),
                    "base_degen": fm_b.get("degen_ending", 0),
                    "ref_len":  ln["ref"]["mean"],
                    "main_len": ln["main"]["mean"],
                    "base_len": ln["base"]["mean"],
                })
            j = load_judge(ds, nfe, main, base)
            if j:
                agg = j["aggregate"]
                rows_judge.append({
                    "dataset": ds, "comparison": label,
                    "n": agg["n_parsed"],
                    "a_win": agg["overall_pct"]["A"],
                    "b_win": agg["overall_pct"]["B"],
                    "tie":   agg["overall_pct"]["tie"],
                    "a_fact": agg["a_means"]["factuality"],
                    "a_cov":  agg["a_means"]["coverage"],
                    "a_flu":  agg["a_means"]["fluency"],
                    "a_conc": agg["a_means"]["conciseness"],
                    "b_fact": agg["b_means"]["factuality"],
                    "b_cov":  agg["b_means"]["coverage"],
                    "b_flu":  agg["b_means"]["fluency"],
                    "b_conc": agg["b_means"]["conciseness"],
                })

    # ---------- write ----------
    out_md = os.path.join(ANA, f"_aggregate_nfe{nfe}.md")
    out_json = os.path.join(ANA, f"_aggregate_nfe{nfe}.json")

    with open(out_md, "w") as f:
        f.write(f"# Summarization analysis aggregate (NFE={nfe}, 1000 samples each)\n\n")

        # ----- Table A: case analysis -----
        f.write("## Table A. Per-sample ΔROUGE-1 buckets + failure modes\n\n")
        f.write("For each (dataset, comparison), the % of samples where SDE wins big "
                "(ΔR-1≥+5), wins (+1..+5), ties, loses, loses big.\n\n")
        f.write("| Dataset | Comparison | ΔR-1 | Win+big | Win | Tie | Loss | Loss-big | SDE len | base len | ref len |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_case:
            f.write(f"| {r['dataset']} | {r['comparison']} | {r['delta_r1']:+.2f} | "
                    f"{r['win_big']}% | {r['win']}% | {r['tie']}% | {r['loss']}% | {r['loss_big']}% | "
                    f"{r['main_len']} | {r['base_len']} | {r['ref_len']} |\n")

        f.write("\n### Failure-mode rates (% of samples)\n\n")
        f.write("| Dataset | Comparison | SDE preEOS | SDE rep | SDE degen | base preEOS | base rep | base degen |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_case:
            f.write(f"| {r['dataset']} | {r['comparison']} | "
                    f"{r['main_preos']}% | {r['main_rep']}% | {r['main_degen']}% | "
                    f"{r['base_preos']}% | {r['base_rep']}% | {r['base_degen']}% |\n")

        # ----- Table B: LLM-judge -----
        f.write("\n## Table B. LLM-as-judge (GPT-5.4) pairwise\n\n")
        f.write("Overall preference win-rates (%) and per-axis means (1-5).\n\n")
        f.write("| Dataset | Comparison | SDE win | base win | Tie | "
                "SDE fact | SDE cov | SDE flu | SDE conc | "
                "base fact | base cov | base flu | base conc |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_judge:
            f.write(f"| {r['dataset']} | {r['comparison']} | "
                    f"**{r['a_win']}%** | {r['b_win']}% | {r['tie']}% | "
                    f"{r['a_fact']} | {r['a_cov']} | {r['a_flu']} | {r['a_conc']} | "
                    f"{r['b_fact']} | {r['b_cov']} | {r['b_flu']} | {r['b_conc']} |\n")

        # ----- Dataset-averaged summary -----
        f.write("\n## Dataset-averaged summary (across 5 datasets)\n\n")
        for label in [c[2] for c in COMPARISONS]:
            j_rows = [r for r in rows_judge if r["comparison"] == label]
            c_rows = [r for r in rows_case if r["comparison"] == label]
            if not j_rows and not c_rows:
                continue
            f.write(f"### {label}\n\n")
            if c_rows:
                def mean(k): return round(sum(r[k] for r in c_rows) / len(c_rows), 2)
                f.write(f"- ΔROUGE-1 (mean across datasets): **{mean('delta_r1'):+.2f}**\n")
                f.write(f"- Win+big bucket: **{mean('win_big')}%**,  Win: {mean('win')}%,  "
                        f"Tie: {mean('tie')}%,  Loss: {mean('loss')}%,  Loss-big: {mean('loss_big')}%\n")
                f.write(f"- SDE failure rates: preEOS={mean('main_preos')}%, "
                        f"rep={mean('main_rep')}%, degen={mean('main_degen')}%\n")
                f.write(f"- Baseline failure rates: preEOS={mean('base_preos')}%, "
                        f"rep={mean('base_rep')}%, degen={mean('base_degen')}%\n")
            if j_rows:
                def mj(k): return round(sum(r[k] for r in j_rows) / len(j_rows), 2)
                f.write(f"- **LLM-judge SDE-win: {mj('a_win')}%**,  base-win: {mj('b_win')}%,  tie: {mj('tie')}%\n")
                f.write(f"- Axes (SDE / base):  "
                        f"fact {mj('a_fact')}/{mj('b_fact')}  "
                        f"cov {mj('a_cov')}/{mj('b_cov')}  "
                        f"flu {mj('a_flu')}/{mj('b_flu')}  "
                        f"conc {mj('a_conc')}/{mj('b_conc')}\n")
            f.write("\n")

    with open(out_json, "w") as f:
        json.dump({"case": rows_case, "judge": rows_judge}, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print(f"case rows: {len(rows_case)}, judge rows: {len(rows_judge)}")


if __name__ == "__main__":
    main()
