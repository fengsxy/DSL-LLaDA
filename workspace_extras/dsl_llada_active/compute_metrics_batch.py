"""Batch compute GenPPL + MAUVE on all result files that don't have metrics yet.

Usage: CUDA_VISIBLE_DEVICES=7 python dsl_llada/compute_metrics_batch.py --gpu 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(_root, "eval_results", "sde_gen_formal")


def compute_gen_ppl(texts, gpt2_model, gpt2_tok, device):
    nlls = []
    for text in texts:
        words = text.split()[:128]
        truncated = " ".join(words)
        if not truncated.strip():
            continue
        enc = gpt2_tok(truncated, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            nlls.append(gpt2_model(input_ids, labels=input_ids).loss.item())
    return float(np.exp(np.mean(nlls))) if nlls else float("nan")


def compute_distinct_n(texts, n):
    ngrams = []
    for t in texts:
        ws = t.split()
        for i in range(len(ws) - n + 1):
            ngrams.append(tuple(ws[i:i+n]))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def compute_rep_rate(texts):
    total = rep = 0
    for t in texts:
        ws = t.split()
        for i in range(1, len(ws)):
            total += 1
            if ws[i] == ws[i-1]:
                rep += 1
    return rep / total if total else 0.0


def compute_mauve(gen_texts, ref_texts, device_id):
    try:
        import mauve
        gen_valid = [t for t in gen_texts if t.strip()]
        ref_valid = [t for t in ref_texts[:len(gen_valid)] if t.strip()]
        min_n = min(len(gen_valid), len(ref_valid))
        if min_n < 10:
            return None
        result = mauve.compute_mauve(
            p_text=ref_valid[:min_n], q_text=gen_valid[:min_n],
            device_id=device_id, max_text_length=256,
        )
        return round(float(result.mauve), 4)
    except Exception as e:
        print(f"    MAUVE error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    device_id = args.gpu

    # Load GPT-2 once
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print("Loading GPT-2 Large...")
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
    print("  GPT-2 ready.")

    files = sorted([f for f in os.listdir(RESULT_DIR)
                    if f.endswith(".json") and "summary" not in f])

    done = 0
    for fname in files:
        fpath = os.path.join(RESULT_DIR, fname)
        data = json.load(open(fpath))

        if "metrics" in data:
            continue

        texts = data.get("texts", [])
        valid = [t for t in texts if t.strip()]
        if not valid:
            print(f"  SKIP {fname}: no valid texts")
            continue

        print(f"  [{done+1}] {fname} ({len(valid)} texts)...", end="", flush=True)
        t0 = time.time()

        # GenPPL
        gen_ppl = compute_gen_ppl(valid, gpt2_model, gpt2_tok, device)

        # D2, D3, Rep, Len
        d2 = compute_distinct_n(valid, 2)
        d3 = compute_distinct_n(valid, 3)
        rep = compute_rep_rate(valid)
        avg_len = float(np.mean([len(t.split()) for t in valid]))

        metrics = {
            "gen_ppl": round(gen_ppl, 2) if not math.isnan(gen_ppl) else None,
            "d2": round(d2, 4),
            "d3": round(d3, 4),
            "rep_rate": round(rep, 4),
            "avg_len": round(avg_len, 1),
            "n_valid": len(valid),
        }

        # MAUVE for prefix continuation (has ref_texts)
        ref_texts = data.get("ref_texts", None)
        if ref_texts:
            metrics["mauve"] = compute_mauve(valid, ref_texts, device_id)

        data["metrics"] = metrics
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

        elapsed = time.time() - t0
        ppl_str = f"PPL={metrics['gen_ppl']}" if metrics['gen_ppl'] else "PPL=N/A"
        mauve_str = f" MAUVE={metrics.get('mauve', 'N/A')}" if ref_texts else ""
        print(f" {ppl_str} D2={metrics['d2']:.3f} Rep={metrics['rep_rate']:.3f} Len={metrics['avg_len']:.0f}{mauve_str} ({elapsed:.0f}s)")
        done += 1

    print(f"\nDone: {done} files processed.")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'File':55s} | {'PPL':>6} | {'D2':>5} | {'Rep':>5} | {'Len':>5} | {'MAUVE':>6}")
    print("-" * 90)
    for fname in files:
        fpath = os.path.join(RESULT_DIR, fname)
        data = json.load(open(fpath))
        m = data.get("metrics", {})
        if not m:
            continue
        name = fname.replace(".json", "")
        ppl = m.get("gen_ppl", "?")
        mauve = m.get("mauve", "")
        print(f"  {name:53s} | {ppl:>6} | {m.get('d2','?'):>5} | {m.get('rep_rate','?'):>5} | {m.get('avg_len','?'):>5} | {mauve if mauve else '':>6}")


if __name__ == "__main__":
    main()
