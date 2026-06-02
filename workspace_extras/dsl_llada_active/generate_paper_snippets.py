"""Generate LaTeX snippets from Stage-1 results ready to paste into paper.tex.

Produces:
  (1) Updated hero table right-half  @ NFE=64  (R-1, R-2, R-L, Len, Degen, BERTScore)
  (2) LLM-as-judge 4-axis table
  (3) Case-analysis bucket table
  (4) Failure-mode table

Reads:
  - eval_results/summarization/{dataset}_{tag}_nfe64.json  (generated + scored)
  - eval_results/summarization/analysis/_aggregate_nfe64.json  (case + judge agg)

Usage:
    python dsl_llada/generate_paper_snippets.py --nfe 64 > snippets_nfe64.tex
"""
import argparse
import json
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM = os.path.join(_root, "eval_results", "summarization")
ANA = os.path.join(SUM, "analysis")

DATASETS = [
    ("xsum", "XSum"),
    ("cnn_dailymail", "CNN/DM"),
    ("pubmed", "PubMed"),
    ("arxiv", "arXiv"),
    ("billsum", "BillSum"),
]


def get(path, default=None):
    if os.path.exists(path):
        try: return json.load(open(path))
        except Exception: return default
    return default


def load_cfg(dataset, tag, nfe):
    return get(os.path.join(SUM, f"{dataset}_{tag}_nfe{nfe}.json"))


def fmt(v, prec=1, best=False):
    if v is None: return "---"
    s = f"{v:.{prec}f}"
    return f"\\best{{{s}}}" if best else s


def section_hero(nfe):
    """NFE=64 right-half: R-1/R-2/R-L, Len, Degen, BERTScore, len."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quality at NFE=64 on 1{,}000 samples per benchmark (seed=42). "
                 r"DSL-LLaDA-SDE matches reference length without length control and achieves the highest BERTScore on every dataset.}")
    lines.append(r"\label{tab:sum_nfe64_1000}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{ll rrr rrr}")
    lines.append(r"\toprule")
    lines.append(r" & & R-1 & R-2 & R-L & BERT-F1 & Len(w) & Degen\% \\")
    lines.append(r"\midrule")
    for ds, nice in DATASETS:
        rows = []
        for tag, pretty in [
            ("b1_sde",                     r"\textbf{DSL-LLaDA-SDE}"),
            ("original_remask",            r"LLaDA"),
            ("original_remask_eosInf_b32", r"LLaDA+EOS+Block"),
        ]:
            d = load_cfg(ds, tag, nfe)
            rows.append((tag, pretty, d))

        # pick best per column among the 3
        vals = {
            "r1": [(tag, (d or {}).get("rouge1")) for tag, _, d in rows],
            "r2": [(tag, (d or {}).get("rouge2")) for tag, _, d in rows],
            "rl": [(tag, (d or {}).get("rougeL")) for tag, _, d in rows],
            "bs": [(tag, (d or {}).get("bertscore_f1")) for tag, _, d in rows],
        }
        best = {k: max((v for _, v in vals[k] if v is not None), default=None) for k in vals}

        for j, (tag, pretty, d) in enumerate(rows):
            if d is None:
                row = f"  & {pretty} & --- & --- & --- & --- & --- & --- \\\\"
            else:
                r1 = fmt(d.get("rouge1"), best=(d.get("rouge1") == best["r1"]))
                r2 = fmt(d.get("rouge2"), best=(d.get("rouge2") == best["r2"]))
                rl = fmt(d.get("rougeL"), best=(d.get("rougeL") == best["rl"]))
                bs = fmt(d.get("bertscore_f1"), prec=2,
                         best=(d.get("bertscore_f1") == best["bs"]))
                ln = fmt(d.get("avg_words"), prec=0)
                dg = fmt(d.get("degenerate_pct"), prec=1)
                ds_label = r"\multirow{3}{*}{" + nice + r"}" if j == 0 else ""
                row = f"  {ds_label} & {pretty} & {r1} & {r2} & {rl} & {bs} & {ln} & {dg} \\\\"
            lines.append(row)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def section_judge(nfe):
    agg = get(os.path.join(ANA, f"_aggregate_nfe{nfe}.json"))
    if not agg or not agg.get("judge"): return ""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LLM-as-judge (GPT-5.4) pairwise preference. "
                 r"SDE vs LLaDA+EOS+Block @ NFE=64, 100 samples/dataset. "
                 r"Scores 1--5 averaged across samples; "
                 r"``SDE win'' is overall pairwise preference rate.}")
    lines.append(r"\label{tab:judge_nfe64}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{ll rrr rrrr rrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Comparison & SDE-win\% & Base-win\% & Tie\% "
                 r"& \multicolumn{4}{c}{DSL-LLaDA-SDE (1-5)} "
                 r"& \multicolumn{4}{c}{Baseline (1-5)} \\")
    lines.append(r"\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    lines.append(r" & & & & & fact & cov & flu & conc & fact & cov & flu & conc \\")
    lines.append(r"\midrule")
    # prefer SDE vs LLaDA+EOS+Block
    for row in agg["judge"]:
        if row["comparison"] != "SDE vs LLaDA+EOS+Block": continue
        lines.append(
            f"{row['dataset']} & {row['comparison']} & "
            f"\\textbf{{{row['a_win']:.1f}}} & {row['b_win']:.1f} & {row['tie']:.1f} & "
            f"{row['a_fact']:.2f} & {row['a_cov']:.2f} & {row['a_flu']:.2f} & {row['a_conc']:.2f} & "
            f"{row['b_fact']:.2f} & {row['b_cov']:.2f} & {row['b_flu']:.2f} & {row['b_conc']:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def section_buckets(nfe):
    agg = get(os.path.join(ANA, f"_aggregate_nfe{nfe}.json"))
    if not agg or not agg.get("case"): return ""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-sample winners. Fraction of the 1{,}000 test "
                 r"samples on which DSL-LLaDA-SDE beats LLaDA+EOS+Block by "
                 r"$\geq$+5 R-1 (Win-big), +1..+5 (Win), ties, or loses. "
                 r"SDE wins on the majority of samples in every benchmark.}")
    lines.append(r"\label{tab:buckets_nfe64}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l rrrrr rr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Win-big & Win & Tie & Loss & Loss-big & $\Delta$R-1 avg & SDE-total \\")
    lines.append(r"\midrule")
    for row in agg["case"]:
        if row["comparison"] != "SDE vs LLaDA+EOS+Block": continue
        sde_total = row["win_big"] + row["win"]
        lines.append(
            f"{row['dataset']} & "
            f"\\textbf{{{row['win_big']}\\%}} & {row['win']}\\% & {row['tie']}\\% & "
            f"{row['loss']}\\% & {row['loss_big']}\\% & "
            f"{row['delta_r1']:+.2f} & \\textbf{{{sde_total}\\%}} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def section_failures(nfe):
    agg = get(os.path.join(ANA, f"_aggregate_nfe{nfe}.json"))
    if not agg or not agg.get("case"): return ""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Automated failure-mode rates @ NFE=64 on 1{,}000 samples. "
                 r"\textbf{degen}=dot/comma spam, \textbf{rep}=$\geq$15\% 4-gram repetition, "
                 r"\textbf{preEOS}=output $<$40\% of reference length.}")
    lines.append(r"\label{tab:failures_nfe64}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l rrr rrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{DSL-LLaDA-SDE} & \multicolumn{3}{c}{LLaDA+EOS+Block} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append(r"Dataset & preEOS & rep & degen & preEOS & rep & degen \\")
    lines.append(r"\midrule")
    for row in agg["case"]:
        if row["comparison"] != "SDE vs LLaDA+EOS+Block": continue
        lines.append(
            f"{row['dataset']} & "
            f"{row['main_preos']}\\% & {row['main_rep']}\\% & {row['main_degen']}\\% & "
            f"{row['base_preos']}\\% & {row['base_rep']}\\% & {row['base_degen']}\\% \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfe", type=int, default=64)
    args = ap.parse_args()
    parts = [
        "% === HERO TABLE @ NFE=64 (1000 samples + BERTScore) ===\n",
        section_hero(args.nfe), "\n\n",
        "% === CASE-ANALYSIS BUCKET TABLE ===\n",
        section_buckets(args.nfe), "\n\n",
        "% === FAILURE-MODE TABLE ===\n",
        section_failures(args.nfe), "\n\n",
        "% === LLM-AS-JUDGE TABLE ===\n",
        section_judge(args.nfe), "\n",
    ]
    out = "\n".join(parts)
    print(out)


if __name__ == "__main__":
    main()
