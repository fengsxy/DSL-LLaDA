#!/usr/bin/env python3
"""Plot mask-rate calibration from eval_results/*/t3_mask.json."""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_result(root, model):
    path = Path(root) / model / "t3_mask.json"
    with path.open() as f:
        data = json.load(f)
    return path, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--results_root", default="eval_results")
    parser.add_argument("--out", default="eval_results/mask_calibration.png")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    ax_rate, ax_rel = axes
    summary = {}

    for model in args.models:
        path, data = load_result(args.results_root, model)
        rates = []
        accs = []
        confs = []
        eces = []
        rel_conf_by_bin = []
        rel_acc_by_bin = []

        for rate_key, vals in sorted(data["mask_rates"].items(), key=lambda kv: int(kv[0])):
            rates.append(int(rate_key))
            accs.append(vals["accuracy"])
            confs.append(vals["avg_confidence"])
            eces.append(vals["ece"])
            bins = vals.get("bins", [])
            if bins:
                rel_conf_by_bin.append([b["avg_confidence"] for b in bins])
                rel_acc_by_bin.append([b["accuracy"] for b in bins])

        ax_rate.plot(rates, accs, marker="o", label=f"{model} acc")
        ax_rate.plot(rates, confs, marker="x", linestyle="--", label=f"{model} conf")

        rel = None
        if rel_conf_by_bin and rel_acc_by_bin:
            rel_conf = np.array(rel_conf_by_bin, dtype=float)
            rel_acc = np.array(rel_acc_by_bin, dtype=float)
            valid = rel_conf.sum(axis=0) > 0
            rel = {
                "confidence": rel_conf[:, valid].mean(axis=0).round(4).tolist(),
                "accuracy": rel_acc[:, valid].mean(axis=0).round(4).tolist(),
            }
            ax_rel.plot(rel["confidence"], rel["accuracy"], marker="o", label=model)

        summary[model] = {
            "source": str(path),
            "mask_rates": rates,
            "accuracy": accs,
            "avg_confidence": confs,
            "ece": eces,
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "mean_confidence": round(float(np.mean(confs)), 4),
            "mean_ece": round(float(np.mean(eces)), 4),
            "mean_reliability": rel,
        }

    ax_rate.set_xlabel("Mask rate (%)")
    ax_rate.set_ylabel("Token-level value")
    ax_rate.set_title("Accuracy vs. confidence by mask rate")
    ax_rate.set_ylim(0, 1)
    ax_rate.grid(alpha=0.25)
    ax_rate.legend(fontsize=7)

    ax_rel.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
    ax_rel.set_xlabel("Mean confidence bin")
    ax_rel.set_ylabel("Mean accuracy bin")
    ax_rel.set_title("Reliability curve averaged over mask rates")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.grid(alpha=0.25)
    ax_rel.legend(fontsize=7)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)

    json_out = Path(args.json_out) if args.json_out else out.with_suffix(".json")
    with json_out.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {out}")
    print(f"Saved {json_out}")


if __name__ == "__main__":
    main()
