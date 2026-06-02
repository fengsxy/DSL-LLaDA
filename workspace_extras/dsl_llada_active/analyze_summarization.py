"""Case analysis for summarization eval outputs.

Compares per-sample results across methods on a fixed dataset, producing:
  - ROUGE-1 delta buckets (big-win/win/tie/loss/big-loss)
  - length distribution stats vs reference
  - automated failure-mode tags per sample (premature_eos / repetition /
    degen_ending / empty / off_topic)
  - top-k qualitative cases per bucket

Output: JSON summary + markdown report.

Usage:
    python dsl_llada/analyze_summarization.py \
        --dataset xsum --nfe 64 \
        --main  b1_sde \
        --baseline original_remask_eosInf_b32 \
        --outdir eval_results/summarization/analysis
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(_root, "eval_results", "summarization")


# ---------- failure-mode heuristics ----------

def tag_failures(gen, ref, gen_words):
    tags = []
    if not gen or not gen.strip():
        tags.append("empty"); return tags
    ref_w = max(1, len(ref.split()))
    # premature EOS: <40% of reference or <15 words total
    if gen_words < 15 or gen_words < 0.4 * ref_w:
        tags.append("premature_eos")
    # 4-gram repetition rate
    toks = gen.split()
    if len(toks) >= 8:
        ngrams = [" ".join(toks[i:i+4]) for i in range(len(toks)-3)]
        if ngrams:
            top = Counter(ngrams).most_common(1)[0][1]
            rep_rate = top / len(ngrams)
            if rep_rate > 0.15:
                tags.append("repetition")
    # dot/comma spam ending
    tail = gen[-40:]
    if tail.count(".") > 8 or tail.count(",") > 8 or "......." in gen:
        tags.append("degen_ending")
    # off-topic heuristic (very weak but ROUGE-based; only if single-token overlap almost zero)
    return tags


# ---------- load ----------

def load_config(dataset, method_tag, nfe):
    path = os.path.join(DIR, f"{dataset}_{method_tag}_nfe{nfe}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing: {path}")
    return json.load(open(path))


def index_by_id(data):
    return {s["id"]: s for s in data["samples"]}


# ---------- analysis ----------

def bucket_delta(d1):
    if d1 >= 5: return "big_win"
    if d1 >= 1: return "win"
    if d1 > -1: return "tie"
    if d1 > -5: return "loss"
    return "big_loss"


def analyze(dataset, nfe, main_tag, base_tag, outdir, topk=5):
    os.makedirs(outdir, exist_ok=True)
    data_m = load_config(dataset, main_tag, nfe)
    data_b = load_config(dataset, base_tag, nfe)
    idx_m = index_by_id(data_m)
    idx_b = index_by_id(data_b)
    common = sorted(set(idx_m) & set(idx_b))
    print(f"[{dataset}/nfe={nfe}] main={main_tag} base={base_tag} common={len(common)}")

    # per-sample delta + tags
    records = []
    for sid in common:
        sm, sb = idx_m[sid], idx_b[sid]
        ref = sm["reference"]
        rec = {
            "id": sid,
            "reference": ref,
            "ref_words": len(ref.split()),
            "main_gen": sm["generated"],
            "main_words": sm["gen_words"],
            "main_r1": sm["rouge1"],
            "main_r2": sm["rouge2"],
            "main_rl": sm["rougeL"],
            "base_gen": sb["generated"],
            "base_words": sb["gen_words"],
            "base_r1": sb["rouge1"],
            "base_r2": sb["rouge2"],
            "base_rl": sb["rougeL"],
            "delta_r1": round(sm["rouge1"] - sb["rouge1"], 2),
            "delta_r2": round(sm["rouge2"] - sb["rouge2"], 2),
            "delta_rl": round(sm["rougeL"] - sb["rougeL"], 2),
            "main_tags": tag_failures(sm["generated"], ref, sm["gen_words"]),
            "base_tags": tag_failures(sb["generated"], ref, sb["gen_words"]),
        }
        rec["bucket"] = bucket_delta(rec["delta_r1"])
        records.append(rec)

    # bucket counts
    buckets = Counter(r["bucket"] for r in records)
    n = len(records)
    bucket_pct = {b: round(buckets.get(b, 0) / n * 100, 1)
                  for b in ["big_win", "win", "tie", "loss", "big_loss"]}

    # failure-mode rates
    def fm_rates(key):
        counts = Counter()
        for r in records:
            for t in r[key]:
                counts[t] += 1
        return {t: round(c / n * 100, 1) for t, c in counts.items()}

    main_fm = fm_rates("main_tags")
    base_fm = fm_rates("base_tags")

    # length stats
    def len_stats(key):
        xs = np.array([r[key] for r in records])
        return {"mean": round(float(xs.mean()), 1),
                "median": int(np.median(xs)),
                "p10": int(np.percentile(xs, 10)),
                "p90": int(np.percentile(xs, 90))}
    length = {
        "ref": len_stats("ref_words"),
        "main": len_stats("main_words"),
        "base": len_stats("base_words"),
    }

    # aggregate ROUGE (sanity check vs stored)
    agg = {
        "main_r1": round(float(np.mean([r["main_r1"] for r in records])), 2),
        "base_r1": round(float(np.mean([r["base_r1"] for r in records])), 2),
        "main_r2": round(float(np.mean([r["main_r2"] for r in records])), 2),
        "base_r2": round(float(np.mean([r["base_r2"] for r in records])), 2),
        "main_rl": round(float(np.mean([r["main_rl"] for r in records])), 2),
        "base_rl": round(float(np.mean([r["base_rl"] for r in records])), 2),
        "mean_delta_r1": round(float(np.mean([r["delta_r1"] for r in records])), 2),
    }

    # top-k examples per bucket (by |delta_r1|)
    examples = {}
    for b in ["big_win", "win", "tie", "loss", "big_loss"]:
        subset = [r for r in records if r["bucket"] == b]
        key_fn = (lambda r: -r["delta_r1"]) if b in ("big_win", "win") \
                 else ((lambda r: r["delta_r1"]) if b in ("big_loss", "loss")
                       else (lambda r: abs(r["delta_r1"])))
        subset.sort(key=key_fn)
        examples[b] = subset[:topk]

    summary = {
        "dataset": dataset,
        "nfe": nfe,
        "main_tag": main_tag,
        "base_tag": base_tag,
        "n": n,
        "aggregate": agg,
        "bucket_counts": dict(buckets),
        "bucket_pct": bucket_pct,
        "failure_modes_main": main_fm,
        "failure_modes_base": base_fm,
        "length": length,
    }

    out_json = os.path.join(outdir, f"{dataset}_nfe{nfe}__{main_tag}_vs_{base_tag}.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records, "examples": examples},
                  f, ensure_ascii=False, indent=2)

    out_md = os.path.join(outdir, f"{dataset}_nfe{nfe}__{main_tag}_vs_{base_tag}.md")
    with open(out_md, "w") as f:
        f.write(f"# {dataset} NFE={nfe}: {main_tag} vs {base_tag}\n\n")
        f.write(f"n={n}\n\n")
        f.write(f"## Aggregate\n\n")
        for k, v in agg.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## Bucket distribution (by ΔR-1)\n\n")
        for b in ["big_win", "win", "tie", "loss", "big_loss"]:
            f.write(f"- **{b}**: {buckets.get(b,0)} ({bucket_pct[b]}%)\n")
        f.write(f"\n## Length (words)\n\n")
        f.write("| | mean | median | p10 | p90 |\n|--|--|--|--|--|\n")
        for k in ("ref", "main", "base"):
            s = length[k]
            f.write(f"| {k} | {s['mean']} | {s['median']} | {s['p10']} | {s['p90']} |\n")
        f.write(f"\n## Failure modes (% of samples)\n\n")
        f.write("| tag | main ({m}) | base ({b}) |\n".format(m=main_tag, b=base_tag))
        f.write("|--|--|--|\n")
        all_tags = sorted(set(main_fm) | set(base_fm))
        for t in all_tags:
            f.write(f"| {t} | {main_fm.get(t, 0)}% | {base_fm.get(t, 0)}% |\n")
        f.write(f"\n## Example cases\n\n")
        for b in ["big_win", "big_loss", "win", "loss", "tie"]:
            subset = examples[b]
            if not subset:
                continue
            f.write(f"### {b} (top {len(subset)})\n\n")
            for r in subset:
                f.write(f"**id={r['id']}**  ΔR1={r['delta_r1']:+.1f}  "
                        f"(main={r['main_r1']:.1f}, base={r['base_r1']:.1f})  "
                        f"main_tags={r['main_tags'] or '-'}  "
                        f"base_tags={r['base_tags'] or '-'}\n\n")
                f.write(f"- REF ({r['ref_words']}w): {r['reference']}\n")
                f.write(f"- MAIN ({r['main_words']}w): {r['main_gen']}\n")
                f.write(f"- BASE ({r['base_words']}w): {r['base_gen']}\n\n")
    print(f"  -> {out_md}")
    print(f"  -> {out_json}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--nfe", type=int, required=True)
    p.add_argument("--main", required=True, help="main method tag, e.g. b1_sde")
    p.add_argument("--baseline", required=True,
                   help="baseline method tag, e.g. original_remask_eosInf_b32")
    p.add_argument("--outdir", default=os.path.join(DIR, "analysis"))
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()
    analyze(args.dataset, args.nfe, args.main, args.baseline, args.outdir, args.topk)


if __name__ == "__main__":
    main()
