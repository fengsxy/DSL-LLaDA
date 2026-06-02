"""Bootstrap CIs and position-bias diagnostics for summarization judge files."""
import argparse
import glob
import json
import os

import numpy as np


def bootstrap_ci(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def summarize_file(path, n_boot, seed):
    data = json.load(open(path))
    records = [r for r in data.get("records", []) if r.get("parsed") and r.get("overall")]
    n = len(records)

    a_win = np.array([1.0 if r["overall"] == "A" else 0.0 for r in records])
    b_win = np.array([1.0 if r["overall"] == "B" else 0.0 for r in records])
    tie = np.array([1.0 if r["overall"] == "tie" else 0.0 for r in records])
    non_tie = [r for r in records if r["overall"] in ("A", "B")]

    shown_a_wins = []
    shown_b_wins = []
    for r in records:
        # `overall` is already un-swapped to original A/B. Recover which visible
        # position won to check positional preference.
        if r["overall"] == "tie":
            continue
        if not r.get("swap", False):
            shown_a_wins.append(1.0 if r["overall"] == "A" else 0.0)
            shown_b_wins.append(1.0 if r["overall"] == "B" else 0.0)
        else:
            shown_a_wins.append(1.0 if r["overall"] == "B" else 0.0)
            shown_b_wins.append(1.0 if r["overall"] == "A" else 0.0)

    shown_a_rate = float(np.mean(shown_a_wins)) if shown_a_wins else 0.0
    swap_true = [r for r in records if r.get("swap", False)]
    swap_false = [r for r in records if not r.get("swap", False)]

    def a_rate(rows):
        if not rows:
            return 0.0
        return sum(1 for r in rows if r["overall"] == "A") / len(rows)

    return {
        "file": path,
        "dataset": data.get("dataset"),
        "nfe": data.get("nfe"),
        "tag_a": data.get("tag_a"),
        "tag_b": data.get("tag_b"),
        "n": n,
        "a_win": round(float(a_win.mean()) if n else 0.0, 4),
        "a_win_ci": bootstrap_ci(a_win, n_boot=n_boot, seed=seed),
        "b_win": round(float(b_win.mean()) if n else 0.0, 4),
        "b_win_ci": bootstrap_ci(b_win, n_boot=n_boot, seed=seed + 1),
        "tie": round(float(tie.mean()) if n else 0.0, 4),
        "tie_ci": bootstrap_ci(tie, n_boot=n_boot, seed=seed + 2),
        "shown_a_win_non_tie": round(shown_a_rate, 4),
        "shown_a_minus_0p5": round(shown_a_rate - 0.5, 4),
        "swap_true_n": len(swap_true),
        "swap_false_n": len(swap_false),
        "orig_a_win_when_swapped": round(a_rate(swap_true), 4),
        "orig_a_win_when_not_swapped": round(a_rate(swap_false), 4),
        "n_non_tie": len(non_tie),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="eval_results/summarization/analysis/judge__*.json")
    parser.add_argument("--out", default="eval_results/summarization/analysis/judge_reliability.json")
    parser.add_argument("--md", default="docs/plans/judge_reliability_20260510.md")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    rows = [summarize_file(p, args.n_boot, args.seed) for p in paths]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_files": len(rows), "rows": rows}, f, indent=2)

    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    with open(args.md, "w") as f:
        f.write("# Judge Reliability Summary (2026-05-10)\n\n")
        f.write(f"Analyzed `{len(rows)}` judge files. CIs are bootstrap 95% intervals over judged samples.\n\n")
        f.write("| Dataset | NFE | A | B | n | A win | 95% CI | B win | 95% CI | shown-A win | pos bias |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['nfe']} | {r['tag_a']} | {r['tag_b']} | {r['n']} | "
                f"{100*r['a_win']:.1f} | [{100*r['a_win_ci'][0]:.1f}, {100*r['a_win_ci'][1]:.1f}] | "
                f"{100*r['b_win']:.1f} | [{100*r['b_win_ci'][0]:.1f}, {100*r['b_win_ci'][1]:.1f}] | "
                f"{100*r['shown_a_win_non_tie']:.1f} | {100*r['shown_a_minus_0p5']:+.1f} |\n"
            )
        f.write("\n`shown-A win` is computed after recovering which visible position the judge picked. "
                "Large deviations from 50% indicate possible position bias, but they are also affected by "
                "finite sample size and ties.\n")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
