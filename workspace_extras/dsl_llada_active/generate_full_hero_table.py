"""Generate the full Table 1 (summarization hero) using 1000-sample numbers
across all NFEs and all three methods.

Output goes to stdout, ready to paste into paper.tex.
"""
import json, os, sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM = os.path.join(_root, "eval_results", "summarization")

DATASETS = [
    ("xsum", "XSum", 21),
    ("cnn_dailymail", "CNN/DM", 55),
    ("pubmed", "PubMed", 205),
    ("arxiv", "arXiv", 163),
    ("billsum", "BillSum", 180),
]

METHODS = [
    ("b1_sde",                     r"\textbf{DSL-LLaDA-SDE} (ours)"),
    ("original_remask",            r"LLaDA"),
    ("original_remask_eosInf_b32", r"LLaDA+EOS+Block"),
]

NFES = [8, 16, 32, 64]


def load(ds, tag, nfe):
    p = os.path.join(SUM, f"{ds}_{tag}_nfe{nfe}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def f(v, prec=1):
    if v is None: return "---"
    return f"{v:.{prec}f}"


def best_marker(values, value):
    """Mark with \best{} if value is the max of values (None ignored)."""
    if value is None: return "---"
    valid = [x for x in values if x is not None]
    if not valid: return f(value)
    is_best = (value == max(valid))
    s = f(value)
    return r"\best{" + s + "}" if is_best else s


print(r"""\begin{table}[t]
\centering
\caption{Zero-shot summarization on five benchmarks (\textbf{1{,}000 samples}, seed=42). \textbf{Left:} ROUGE-1 across NFE budgets. \textbf{Right:} quality detail at NFE=64 (R-2/R-L F1, BERTScore F1, output length, automated degenerate-ending rate). DSL-LLaDA-SDE leads ROUGE on short-summary tasks (XSum/CNN/DM) at every NFE. Vanilla LLaDA wins ROUGE on long-reference tasks (PubMed/arXiv/BillSum) at NFE=64 because EOS cascade does not fire there; on short-summary tasks its average output is 12--17 words. LLaDA+EOS+Block trades shorter outputs for $\geq$5\% degenerate endings (dot/comma spam) at all NFE on short-summary tasks (24.6\% on XSum at NFE=64).}
\label{tab:summarization_hero}
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{ll cccc cccc}
\toprule
& & \multicolumn{4}{c}{\textbf{ROUGE-1 vs.\ NFE}} & \multicolumn{4}{c}{\textbf{Quality @ NFE=64}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Dataset (ref) & Method & 8 & 16 & 32 & 64 & R-2 & R-L & BERT & Len/Deg \\
\midrule""")

for ds, nice, ref_w in DATASETS:
    # Per dataset, gather all method R-1 values to compute "best" per NFE
    r1_per_nfe = {nfe: [] for nfe in NFES}
    r2_64 = []
    rl_64 = []
    bert_64 = []
    rows = []
    for tag, pretty in METHODS:
        per_method = {}
        for nfe in NFES:
            d = load(ds, tag, nfe)
            r1 = d["rouge1"] if d else None
            per_method[nfe] = r1
            r1_per_nfe[nfe].append(r1)
        d64 = load(ds, tag, 64)
        per_method["r2"] = d64["rouge2"] if d64 else None
        per_method["rl"] = d64["rougeL"] if d64 else None
        per_method["bert"] = d64.get("bertscore_f1") if d64 else None
        per_method["len"] = d64["avg_words"] if d64 else None
        per_method["deg"] = d64["degenerate_pct"] if d64 else None
        rows.append((tag, pretty, per_method))
        r2_64.append(per_method["r2"])
        rl_64.append(per_method["rl"])
        bert_64.append(per_method["bert"])

    print(rf"\multirow{{3}}{{*}}{{{nice} ({ref_w}w)}}")
    for j, (tag, pretty, pm) in enumerate(rows):
        cols = []
        for nfe in NFES:
            v = pm[nfe]
            cols.append(best_marker(r1_per_nfe[nfe], v))
        cols.append(best_marker(r2_64, pm["r2"]))
        cols.append(best_marker(rl_64, pm["rl"]))
        cols.append(best_marker(bert_64, pm["bert"], ) if pm["bert"] is not None else "---")
        # length / degen combined column
        ln = f"{pm['len']:.0f}w" if pm["len"] is not None else "---"
        dg = f"{pm['deg']:.1f}\\%" if pm["deg"] is not None else "---"
        cols.append(f"{ln}/{dg}")
        line = " & ".join(cols)
        prefix = "" if j > 0 else ""
        print(f"& {pretty} & {line} \\\\")
    print(r"\midrule")

# replace last \midrule with \bottomrule
import sys
print(r"""% (replace last \midrule above with \bottomrule)
\bottomrule
\end{tabular}
\end{table}""")
