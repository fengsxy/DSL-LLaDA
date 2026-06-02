"""Build N-sample summarization eval datasets (seed=42).

Mirrors the schema of existing eval_data/*_100.json:
    {id, source, reference, prompt}

Usage:
    python dsl_llada/build_summarization_data.py --n 1000
    python dsl_llada/build_summarization_data.py --n 1000 --datasets xsum
"""
import argparse
import json
import os
import random
import sys

from datasets import load_dataset

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_root, "eval_data")

# (hf_name, hf_config, split, text_key, summary_key, prompt_template, source_tag)
DATASETS = {
    "xsum": (
        "EdinburghNLP/xsum", None, "test",
        "document", "summary",
        "Summarize the following text in one sentence:\n\n{text}",
        "xsum",
    ),
    "cnn_dailymail": (
        "cnn_dailymail", "3.0.0", "test",
        "article", "highlights",
        "Summarize the following news article in 2-3 sentences:\n\n{text}",
        "cnn_dailymail",
    ),
    "pubmed": (
        "ccdv/pubmed-summarization", None, "test",
        "article", "abstract",
        "Summarize the following scientific paper in a concise abstract:\n\n{text}",
        "pubmed",
    ),
    "arxiv": (
        "ccdv/arxiv-summarization", None, "test",
        "article", "abstract",
        "Summarize the following research paper in a concise abstract:\n\n{text}",
        "arxiv",
    ),
    "billsum": (
        "FiscalNote/billsum", None, "test",
        "text", "summary",
        "Summarize the following legislation:\n\n{text}",
        "billsum",
    ),
}


def truncate_words(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def build(name, n, seed, max_input_words):
    hf_name, hf_cfg, split, tkey, skey, tmpl, src_tag = DATASETS[name]
    print(f"[{name}] loading {hf_name} ({hf_cfg or 'default'}, split={split}) ...", flush=True)
    ds = load_dataset(hf_name, hf_cfg, split=split) if hf_cfg else load_dataset(hf_name, split=split)
    total = len(ds)
    print(f"[{name}] total examples: {total}")
    rng = random.Random(seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    kept = []
    for i in idxs:
        if len(kept) >= n:
            break
        row = ds[i]
        text = (row.get(tkey) or "").strip()
        summ = (row.get(skey) or "").strip()
        if not text or not summ:
            continue
        text_trunc = truncate_words(text, max_input_words)
        kept.append({
            "id": row.get("id", i) if row.get("id") is not None else i,
            "source": src_tag,
            "reference": summ,
            "prompt": tmpl.format(text=text_trunc),
        })
    out_file = os.path.join(OUT_DIR, f"{name}_{n}.json")
    with open(out_file, "w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    avg_prompt = sum(len(x["prompt"].split()) for x in kept) / len(kept)
    avg_ref = sum(len(x["reference"].split()) for x in kept) / len(kept)
    print(f"[{name}] wrote {out_file}  n={len(kept)}  avg_prompt={avg_prompt:.0f}w  avg_ref={avg_ref:.0f}w")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--datasets", default="xsum,cnn_dailymail,pubmed,arxiv,billsum",
        help="comma-separated"
    )
    p.add_argument("--max_input_words", type=int, default=1500,
                   help="truncate very long source docs so prompt fits in context")
    args = p.parse_args()
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for name in names:
        if name not in DATASETS:
            print(f"unknown dataset: {name}", file=sys.stderr); sys.exit(1)
        build(name, args.n, args.seed, args.max_input_words)


if __name__ == "__main__":
    main()
