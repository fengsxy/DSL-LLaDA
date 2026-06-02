"""AR baseline summarization eval (Qwen2.5-7B-Instruct).

Same 1000-sample test sets, same prompt template, greedy decoding (temp=0).
Adds a single AR reference point to the hero table; not the focus of the paper.

Usage:
    python dsl_llada/eval/eval_summarization_ar.py \
        --dataset xsum --gpu 0 --shard_id 0 --num_shards 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(_root, "eval_results", "summarization")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_FILES = {
    "xsum": "eval_data/xsum_1000.json",
    "cnn_dailymail": "eval_data/cnn_dailymail_1000.json",
    "pubmed": "eval_data/pubmed_1000.json",
    "arxiv": "eval_data/arxiv_1000.json",
}

GEN_LENGTHS = {
    "xsum": 128,
    "cnn_dailymail": 256,
    "pubmed": 256,
    "arxiv": 256,
}

MODELS = {
    "qwen25_7b":  "Qwen/Qwen2.5-7B-Instruct",
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
}
MODEL_TAGS = {
    "qwen25_7b":  "qwen25_7b_ar",
    "llama31_8b": "llama31_8b_ar",
}


def load_model(device, model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[model_key]
    print(f"[AR/{model_key}] Loading {name}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    return model, tok


def generate_ar(model, tok, prompt, device, gen_length=256):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=gen_length,
            do_sample=False, temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
    new = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def compute_rouge(generated, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    per = []
    r1s, r2s, rls = [], [], []
    for gen, ref in zip(generated, references):
        if not gen.strip():
            per.append({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
            r1s.append(0.0); r2s.append(0.0); rls.append(0.0)
            continue
        s = scorer.score(ref, gen)
        r1, r2, rl = s['rouge1'].fmeasure, s['rouge2'].fmeasure, s['rougeL'].fmeasure
        per.append({"rouge1": round(r1*100, 2), "rouge2": round(r2*100, 2), "rougeL": round(rl*100, 2)})
        r1s.append(r1); r2s.append(r2); rls.append(rl)
    agg = {
        "rouge1": round(np.mean(r1s)*100, 2),
        "rouge2": round(np.mean(r2s)*100, 2),
        "rougeL": round(np.mean(rls)*100, 2),
    }
    return agg, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_FILES.keys()))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gen_length", type=int, default=None)
    p.add_argument("--model_key", default="qwen25_7b", choices=list(MODELS.keys()))
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    gen_length = args.gen_length or GEN_LENGTHS[args.dataset]
    data_path = os.path.join(_root, DATASET_FILES[args.dataset])
    data = json.load(open(data_path))
    if args.limit:
        data = data[:args.limit]
    indexed = [(i, item) for i, item in enumerate(data)]
    if args.num_shards > 1:
        indexed = [(i, it) for (i, it) in indexed if i % args.num_shards == args.shard_id]
    print(f"[{args.dataset}] shard {args.shard_id}/{args.num_shards}: {len(indexed)} samples")

    model, tok = load_model(device, args.model_key)
    method_tag = MODEL_TAGS[args.model_key]

    texts, gids = [], []
    t0 = time.time()
    for g, item in tqdm(indexed, desc=f"{args.dataset}/AR"):
        try:
            text = generate_ar(model, tok, item["prompt"], device, gen_length=gen_length)
        except Exception as e:
            print(f"  failed g={g}: {e}")
            text = ""
        texts.append(text); gids.append(g)
    elapsed = time.time() - t0

    refs = [item["reference"] for (_, item) in indexed]
    agg, per = compute_rouge(texts, refs)
    print(f"  AR ROUGE-1={agg['rouge1']} R-2={agg['rouge2']} R-L={agg['rougeL']}  ({elapsed:.0f}s)")

    samples = [
        {"id": g, "reference": r, "generated": t,
         "gen_words": len(t.split()),
         "rouge1": pr["rouge1"], "rouge2": pr["rouge2"], "rougeL": pr["rougeL"]}
        for g, r, t, pr in zip(gids, refs, texts, per)
    ]
    valid = sum(1 for t in texts if t.strip())
    out = {
        "dataset": args.dataset, "method": f"{args.model_key}_ar",
        "model_key": args.model_key, "gen_method": "ar",
        "nfe": "AR", "gen_length": gen_length, "seed": args.seed,
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "n_samples_total": len(data), "n_samples_here": len(texts), "valid": valid,
        "avg_words": round(float(np.mean([len(t.split()) for t in texts])), 2),
        "degenerate_pct": 0.0,
        **agg,
        "time_s": round(elapsed, 1),
        "samples": samples,
    }
    suffix = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    out_file = os.path.join(OUTPUT_DIR, f"{args.dataset}_{method_tag}{suffix}.json")
    with open(out_file, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
