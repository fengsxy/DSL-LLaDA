"""Merge sharded summarization eval outputs into a single per-config file.

Input:  eval_results/summarization/{dataset}_{method_tag}_nfe{N}_shard{i}of{K}.json
Output: eval_results/summarization/{dataset}_{method_tag}_nfe{N}.json  (merged)

Re-computes aggregate ROUGE/word/degen stats from per-sample records so the
merged file is directly usable by downstream analysis scripts.

Usage:
    python dsl_llada/merge_summarization_shards.py \
        --dataset xsum --method_tag b1_sde --nfe 64 --num_shards 8
    # or match multiple configs at once:
    python dsl_llada/merge_summarization_shards.py --auto
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(_root, "eval_results", "summarization")

SHARD_RE = re.compile(r"^(?P<base>.+?)_shard(?P<i>\d+)of(?P<k>\d+)\.json$")


def _agg(samples, num_total):
    n = len(samples)
    valid = sum(1 for s in samples if s["generated"].strip())
    word_counts = [s["gen_words"] for s in samples]
    avg_w = float(np.mean(word_counts)) if n else 0.0
    r1 = float(np.mean([s["rouge1"] for s in samples])) if n else 0.0
    r2 = float(np.mean([s["rouge2"] for s in samples])) if n else 0.0
    rl = float(np.mean([s["rougeL"] for s in samples])) if n else 0.0
    degen = sum(
        1 for s in samples
        if any(s["generated"].count(p) > 3 for p in [':::','...','{}','""','quest','​'])
    )
    return {
        "n_samples_total": num_total,
        "n_samples_here": n,
        "valid": valid,
        "avg_words": round(avg_w, 2),
        "degenerate_pct": round(degen / max(n, 1) * 100, 2),
        "rouge1": round(r1, 2),
        "rouge2": round(r2, 2),
        "rougeL": round(rl, 2),
    }


def merge_base(base):
    """Merge all shards matching {base}_shard{i}of{k}.json."""
    shard_files = sorted(glob.glob(os.path.join(DIR, f"{base}_shard*of*.json")))
    if not shard_files:
        return None
    first = json.load(open(shard_files[0]))
    num_shards = first["num_shards"]
    if len(shard_files) != num_shards:
        print(f"[{base}] WARNING: found {len(shard_files)}/{num_shards} shards, skipping")
        return None
    samples = []
    time_s = 0.0
    seen = set()
    for sf in shard_files:
        d = json.load(open(sf))
        for s in d["samples"]:
            if s["id"] in seen:
                continue
            seen.add(s["id"])
            samples.append(s)
        time_s += d.get("time_s", 0.0)
    samples.sort(key=lambda s: s["id"])
    merged = {k: v for k, v in first.items() if k not in
              ("samples", "shard_id", "num_shards", "n_samples_here",
               "valid", "avg_words", "degenerate_pct",
               "rouge1", "rouge2", "rougeL", "time_s")}
    agg = _agg(samples, first["n_samples_total"])
    merged.update(agg)
    merged["time_s"] = round(time_s, 1)
    merged["samples"] = samples
    out = os.path.join(DIR, f"{base}.json")
    with open(out, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[{base}] merged {len(samples)}/{first['n_samples_total']} samples -> {out}  "
          f"(R1={agg['rouge1']}, R2={agg['rouge2']}, RL={agg['rougeL']}, "
          f"len={agg['avg_words']}w, degen={agg['degenerate_pct']}%)")
    return out


def auto_merge():
    """Find all sharded configs in DIR and merge each."""
    files = os.listdir(DIR)
    bases = defaultdict(set)
    for fn in files:
        m = SHARD_RE.match(fn)
        if not m:
            continue
        bases[m["base"]].add(int(m["i"]))
    print(f"Found {len(bases)} sharded configs")
    for base, idxs in sorted(bases.items()):
        merge_base(base)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auto", action="store_true",
                   help="auto-detect all sharded configs and merge")
    p.add_argument("--dataset")
    p.add_argument("--method_tag",
                   help="e.g. b1_sde or original_remask_eosInf_b32")
    p.add_argument("--nfe", type=int)
    args = p.parse_args()
    if args.auto:
        auto_merge()
    else:
        assert args.dataset and args.method_tag and args.nfe, \
            "provide --dataset/--method_tag/--nfe or use --auto"
        base = f"{args.dataset}_{args.method_tag}_nfe{args.nfe}"
        merge_base(base)


if __name__ == "__main__":
    main()
