"""Aggregate merged summarization JSONs into a comparison table (paper/readable).

Reads all eval_results/summarization/{dataset}_{tag}_nfe{N}.json (non-shard),
and prints a table per dataset: method tag x NFE -> R-1, R-2, R-L, BERTScore-F1,
avg length, degen %.

Usage:
    python dsl_llada/summarize_results_table.py
    python dsl_llada/summarize_results_table.py --datasets xsum,cnn_dailymail --csv out.csv
"""
import argparse
import csv
import glob
import json
import os
import re

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(_root, "eval_results", "summarization")
PAT = re.compile(r"^(?P<dataset>[a-z_]+)_(?P<tag>[A-Za-z0-9_]+?)_nfe(?P<nfe>\d+)\.json$")


def gather(datasets):
    rows = []
    for fn in sorted(os.listdir(DIR)):
        if "_shard" in fn or "analysis" in fn:
            continue
        m = PAT.match(fn)
        if not m:
            continue
        ds = m["dataset"]
        if datasets and ds not in datasets:
            continue
        path = os.path.join(DIR, fn)
        try:
            d = json.load(open(path))
        except Exception:
            continue
        rows.append({
            "dataset": ds,
            "tag": m["tag"],
            "nfe": int(m["nfe"]),
            "n": d.get("n_samples_total", d.get("n_samples_here", 0)),
            "R1": d.get("rouge1"),
            "R2": d.get("rouge2"),
            "RL": d.get("rougeL"),
            "BERT": d.get("bertscore_f1"),
            "len": d.get("avg_words"),
            "degen%": d.get("degenerate_pct"),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="xsum,cnn_dailymail,pubmed,arxiv,billsum")
    p.add_argument("--csv", default=None)
    p.add_argument("--sort", default="tag,nfe")
    args = p.parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows = gather(datasets)
    # group by dataset
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds in datasets:
        rs = by_ds.get(ds, [])
        if not rs:
            print(f"\n=== {ds} ===\n  (no rows)"); continue
        rs.sort(key=lambda r: (r["tag"], r["nfe"]))
        print(f"\n=== {ds} (n up to {max((r['n'] or 0) for r in rs)}) ===")
        print(f"{'tag':<32} {'NFE':>4} {'n':>5} {'R1':>6} {'R2':>6} {'RL':>6} {'BERT':>6} {'len':>6} {'deg%':>6}")
        for r in rs:
            def s(v): return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
            print(f"{r['tag']:<32} {r['nfe']:>4} {r['n']:>5} "
                  f"{s(r['R1']):>6} {s(r['R2']):>6} {s(r['RL']):>6} "
                  f"{s(r['BERT']):>6} {s(r['len']):>6} {s(r['degen%']):>6}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset","tag","nfe","n","R1","R2","RL","BERT","len","degen%"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    main()
