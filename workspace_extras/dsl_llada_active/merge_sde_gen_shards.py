"""Merge sharded eval_sde_gen_formal.py outputs.

The generation script shards by global sample index modulo num_shards and stores
the original indices in selected_indices/trace. This merger restores the
original sample order so metrics can be computed on the merged file.
"""
import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True, help="Glob for shard JSON files")
    parser.add_argument("--out", required=True, help="Merged output JSON path")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No shard files match: {args.pattern}")

    shards = [json.load(open(p)) for p in paths]
    first = shards[0]

    by_idx = {}
    trace_by_idx = {}
    ref_by_idx = {}
    for shard, path in zip(shards, paths):
        indices = shard.get("selected_indices")
        if indices is None:
            raise ValueError(f"{path} has no selected_indices; rerun with sharding support")
        texts = shard.get("texts", [])
        if len(indices) != len(texts):
            raise ValueError(f"{path}: selected_indices/texts length mismatch")
        refs = shard.get("ref_texts")
        for i, text in zip(indices, texts):
            by_idx[int(i)] = text
        for item in shard.get("trace", []):
            trace_by_idx[int(item["idx"])] = item
        if refs is not None:
            for i, ref in zip(indices, refs):
                ref_by_idx[int(i)] = ref

    expected_total = first.get("n_samples_total", len(by_idx))
    missing = [i for i in range(expected_total) if i not in by_idx]
    if missing:
        raise ValueError(f"Missing {len(missing)} indices, first few: {missing[:10]}")

    ordered_indices = list(range(expected_total))
    merged = dict(first)
    merged["texts"] = [by_idx[i] for i in ordered_indices]
    merged["trace"] = [trace_by_idx.get(i, {"idx": i}) for i in ordered_indices]
    if ref_by_idx:
        merged["ref_texts"] = [ref_by_idx[i] for i in ordered_indices]
    merged["n_samples"] = expected_total
    merged["n_samples_total"] = expected_total
    merged["shard_id"] = None
    merged["num_shards"] = len(shards)
    merged["selected_indices"] = ordered_indices
    merged["merged_from"] = [os.path.abspath(p) for p in paths]
    merged["total_time_s"] = max(float(s.get("total_time_s", 0.0)) for s in shards)
    merged["sum_shard_time_s"] = round(sum(float(s.get("total_time_s", 0.0)) for s in shards), 1)
    merged["avg_time_per_sample"] = round(merged["total_time_s"] / max(expected_total, 1), 2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(shards)} shards, {expected_total} samples -> {args.out}")


if __name__ == "__main__":
    main()
