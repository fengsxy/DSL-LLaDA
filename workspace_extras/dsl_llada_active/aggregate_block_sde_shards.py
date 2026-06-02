"""Aggregate sharded eval_block_sde.py outputs."""
import argparse
import glob
import json
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    output_abs = os.path.abspath(args.output)
    paths = [
        p for p in paths
        if os.path.abspath(p) != output_abs
        and not os.path.basename(p).endswith("_merged.json")
    ]
    if not paths:
        raise SystemExit(f"No JSON shards found in {args.input_dir}")

    merged = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": args.input_dir,
        "shards": [],
        "results": {},
    }

    for path in paths:
        data = json.load(open(path))
        shard_meta = {
            "path": path,
            "dataset": data.get("dataset"),
            "data_file": data.get("data_file"),
            "data_start": data.get("data_start"),
            "data_end": data.get("data_end"),
            "checkpoint": data.get("checkpoint"),
        }
        merged["shards"].append(shard_meta)

        for name, result in data.get("results", {}).items():
            dataset = data.get("dataset", "unknown")
            key = f"{dataset}_{name}"
            out = merged["results"].setdefault(key, {
                "dataset": dataset,
                "config_name": name,
                "correct": 0,
                "n": 0,
                "time_s": 0.0,
                "nfe": result.get("nfe"),
                "cases": [],
            })
            out["correct"] += int(result.get("correct", 0))
            out["n"] += int(result.get("n", 0))
            out["time_s"] += float(result.get("time_s", 0.0))
            out["cases"].extend(result.get("cases", []))

    for result in merged["results"].values():
        result["accuracy"] = round(result["correct"] / result["n"], 4) if result["n"] else 0.0
        result["time_s"] = round(result["time_s"], 1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"{'Result':<28} {'Correct':>10} {'Acc':>8} {'NFE':>6} {'Time(s)':>9}")
    print("-" * 68)
    for key, result in sorted(merged["results"].items()):
        print(
            f"{key:<28} {result['correct']:>4}/{result['n']:<5} "
            f"{result['accuracy'] * 100:>7.2f}% {result.get('nfe', '?'):>6} "
            f"{result['time_s']:>9.1f}"
        )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
