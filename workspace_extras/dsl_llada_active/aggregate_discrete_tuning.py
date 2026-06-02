"""Aggregate discrete baseline tuning outputs."""
import argparse
import glob
import json
import os
import re

import numpy as np


def rep_rate(texts):
    total = rep = 0
    for text in texts:
        words = text.split()
        for i in range(1, len(words)):
            total += 1
            rep += int(words[i] == words[i - 1])
    return rep / total if total else 0.0


def distinct2(texts):
    grams = []
    for text in texts:
        words = text.split()
        grams.extend(zip(words, words[1:]))
    return len(set(grams)) / len(grams) if grams else 0.0


def parse_config(name):
    cfg = {"block": 256, "eos": "default", "temperature": 0.0}
    m = re.search(r"_b(\d+)", name)
    if m:
        cfg["block"] = int(m.group(1))
    if "_eosInf" in name:
        cfg["eos"] = "eosInf"
    elif "_noEOS" in name:
        cfg["eos"] = "suppress"
    m = re.search(r"_t([0-9p]+)", name)
    if m:
        cfg["temperature"] = float(m.group(1).replace("p", "."))
    return cfg


def load_open(paths):
    rows = []
    for path in paths:
        d = json.load(open(path))
        texts = d.get("texts", [])
        cfg = parse_config(os.path.basename(path))
        rows.append({
            "task": "open",
            "path": path,
            **cfg,
            "n": len(texts),
            "avg_len": round(float(np.mean([len(t.split()) for t in texts])) if texts else 0.0, 2),
            "rep": round(rep_rate(texts), 4),
            "d2": round(distinct2(texts), 4),
            "time_per_sample": d.get("avg_time_per_sample"),
        })
    return rows


def load_sum(paths):
    rows = []
    for path in paths:
        d = json.load(open(path))
        cfg = parse_config(os.path.basename(path))
        rows.append({
            "task": d["dataset"],
            "path": path,
            **cfg,
            "n": d.get("n_samples_here"),
            "r1": d.get("rouge1"),
            "r2": d.get("rouge2"),
            "rl": d.get("rougeL"),
            "avg_len": d.get("avg_words"),
            "degen_pct": d.get("degenerate_pct"),
            "time_per_sample": round(d.get("time_s", 0) / max(d.get("n_samples_here", 1), 1), 3),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/plans/discrete_tuning_phase1_results.md")
    parser.add_argument("--json_out", default="eval_results/discrete_tuning_phase1_summary.json")
    args = parser.parse_args()

    open_paths = glob.glob("eval_results/sde_gen_formal/tune_open_original_remask*_nfe16_gen256.json")
    sum_paths = glob.glob("eval_results/summarization/xsum_original_remask*_tune_nfe16.json")
    sum_paths += glob.glob("eval_results/summarization/cnn_dailymail_original_remask*_tune_nfe16.json")

    rows = load_open(sorted(open_paths)) + load_sum(sorted(sum_paths))
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Discrete Tuning Results\n\n")
        f.write(
            "NFE=16 dev sweep for LLaDA discrete remasking. "
            "Phase 1 sweeps block size and EOS handling at temperature 0; "
            "phase 2 adds a small temperature sweep for representative settings.\n\n"
        )
        for task in ["open", "xsum", "cnn_dailymail"]:
            task_rows = [r for r in rows if r["task"] == task]
            f.write(f"## {task}\n\n")
            if task == "open":
                f.write("| block | eos | temp | n | len | rep | d2 | sec/sample |\n")
                f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
                task_rows.sort(key=lambda r: (r["rep"], -r["d2"], -r["avg_len"]))
                for r in task_rows:
                    f.write(
                        f"| {r['block']} | {r['eos']} | {r['temperature']} | "
                        f"{r['n']} | {r['avg_len']} | {r['rep']:.3f} | "
                        f"{r['d2']:.3f} | {r['time_per_sample']} |\n"
                    )
            else:
                f.write("| block | eos | temp | n | R1 | R2 | RL | len | degen | sec/sample |\n")
                f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                task_rows.sort(key=lambda r: (-(r.get("r1") or 0), r.get("degen_pct") or 0))
                for r in task_rows:
                    f.write(
                        f"| {r['block']} | {r['eos']} | {r['temperature']} | "
                        f"{r['n']} | {r.get('r1')} | {r.get('r2')} | {r.get('rl')} | "
                        f"{r.get('avg_len')} | {r.get('degen_pct')} | {r['time_per_sample']} |\n"
                    )
            f.write("\n")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
