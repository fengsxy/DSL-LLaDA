"""Wall-time benchmark: SDE NFE={8,16,32,64} vs Qwen-AR (with/without KV cache).

Uses 20 prompts per dataset × 4 datasets = 80 prompts. Single GPU. Each model
is loaded once and timed across all 80 samples. Reports per-sample median +
mean wall time.

Usage:
    CUDA_VISIBLE_DEVICES=4 python dsl_llada/benchmark_wall_time.py
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "xsum":          ("eval_data/xsum_1000.json",          128),
    "cnn_dailymail": ("eval_data/cnn_dailymail_1000.json", 256),
    "pubmed":        ("eval_data/pubmed_1000.json",        256),
    "arxiv":         ("eval_data/arxiv_1000.json",         256),
}

QWEN = "Qwen/Qwen2.5-7B-Instruct"
LLAMA3 = "meta-llama/Llama-3.1-8B-Instruct"
LLADA = "GSAI-ML/LLaDA-8B-Instruct"
SDE_CKPT = "checkpoints/pertoken_b1_d100_1k/checkpoint-1000"
TOP_K = 512


def adaptive_ns(nfe):
    if nfe <= 8: return 0.1
    elif nfe <= 16: return 0.05
    elif nfe <= 32: return 0.01
    else: return 0.005


# ---- Loading prompts ----
def load_prompts(n_per_ds=20, seed=42):
    rng = np.random.RandomState(seed)
    bench = []
    for ds, (path, gen_len) in DATASETS.items():
        data = json.load(open(os.path.join(_root, path)))
        idxs = rng.choice(len(data), n_per_ds, replace=False)
        for i in idxs:
            bench.append({"dataset": ds, "prompt": data[int(i)]["prompt"], "gen_length": gen_len})
    return bench


# ---- AR (Qwen / LLaMA) ----
def benchmark_ar(prompts, device, use_cache, model_name, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[{label}, KV={'on' if use_cache else 'off'}] loading ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    # warmup with use_cache flag
    msg = [{"role":"user","content":"Say hello."}]
    txt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    enc = tok(txt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=8, do_sample=False, use_cache=use_cache, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for p in prompts:
        msg = [{"role":"user","content":p["prompt"]}]
        txt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=8192).to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(
                **enc, max_new_tokens=p["gen_length"],
                do_sample=False, use_cache=use_cache,
                pad_token_id=tok.eos_token_id,
            )
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    del model
    torch.cuda.empty_cache()
    return times


# ---- SDE (DSL-LLaDA) ----
def benchmark_sde(prompts, device, nfes):
    """Return {nfe: [times]}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    print(f"[SDE] loading b1 ...", flush=True)
    ckpt = os.path.join(_root, SDE_CKPT)
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    # Load SDE components
    sde_comp = None
    for fname in sorted(os.listdir(ckpt)):
        if fname.endswith(".safetensors"):
            st = load_file(os.path.join(ckpt, fname), device=str(device))
            if "noise_embed.weight" in st:
                ne = st["noise_embed.weight"].float()
                sde_comp = {
                    "ne": ne,
                    "lb": st["converter.logit_bias"].float(),
                    "bw": st["converter.backbone_embedding.weight"].float(),
                    "bb": st["converter.backbone_embedding.bias"].float(),
                    "K":  torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0),
                    "beta_train": float(st["converter.beta"]),
                }
            del st
    assert sde_comp is not None
    nd = sde_comp["ne"].shape[1]
    bv = sde_comp["beta_train"] * 2.0

    out = {nfe: [] for nfe in nfes}
    # warmup once at smallest nfe
    p0 = prompts[0]
    _ = generate_sde_once(model, tok, p0["prompt"], device, sde_comp, nfe=8, gen_length=p0["gen_length"], seed=42)
    torch.cuda.synchronize()

    for nfe in nfes:
        for p in prompts:
            torch.cuda.synchronize()
            t0 = time.time()
            _ = generate_sde_once(model, tok, p["prompt"], device, sde_comp,
                                  nfe=nfe, gen_length=p["gen_length"], seed=42)
            torch.cuda.synchronize()
            out[nfe].append(time.time() - t0)
    del model
    torch.cuda.empty_cache()
    return out


def generate_sde_once(model, tok, text, device, sde, nfe, gen_length, seed):
    ne, lb, bw, bb, K = sde["ne"], sde["lb"], sde["bw"], sde["bb"], sde["K"]
    nd = ne.shape[1]
    bv = sde["beta_train"] * 2.0
    ns = adaptive_ns(nfe)
    steps = nfe // 2

    msg = [{"role":"user","content":text}]
    txt = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tok(txt, add_special_tokens=False, return_tensors="pt")
    pid = enc["input_ids"].to(device)
    P = pid.shape[1]
    pe = model.model.transformer.wte(pid).float()
    dummy = torch.zeros(1, P + gen_length, dtype=torch.long, device=device)

    snr_min = 0.01
    n1 = max(1, int(steps * 0.05))
    n2 = max(1, int(steps * 0.90))
    n3 = steps - n1 - n2
    snrs = torch.cat([
        torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
        torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
        torch.exp(torch.linspace(math.log(74), math.log(100), max(n3, 1) + 1))[1:],
    ]).to(device)

    torch.manual_seed(seed)
    y = F.normalize(torch.randn(1, gen_length, nd, device=device), dim=-1)

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, P:, :].float()
        bp = F.softmax(logits, dim=-1)
        tv, ti = bp.topk(min(TOP_K, bp.shape[-1]), dim=-1)
        tv = tv / tv.sum(dim=-1, keepdim=True)
        return (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)

    for i in range(len(snrs) - 1):
        s, s_next = snrs[i], snrs[i+1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s
        g = ns / s
        y_e = y + f*ds + g*dW
        xh_e = get_xhat(y_e, s_next)
        f_e = (xh_e - y_e) / s_next
        g_e = ns / s_next
        y = y + 0.5*(f + f_e)*ds + 0.5*(g + g_e)*dW
    return y  # (don't bother decoding for benchmark)


# ---- main ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_ds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="eval_results/wall_time_benchmark.json")
    p.add_argument("--mode", choices=["all","qwen_kv","qwen_nokv","sde","llama_kv","llama_nokv"], default="all")
    args = p.parse_args()

    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES restricts which GPU
    prompts = load_prompts(n_per_ds=args.n_per_ds, seed=args.seed)
    print(f"Loaded {len(prompts)} prompts ({args.n_per_ds}/dataset × {len(DATASETS)})")

    # Optional resume: load existing partial file
    out_path = os.path.join(_root, args.out)
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        print(f"  resuming from {out_path}: {list(results.keys())}")

    if args.mode in ("all","qwen_kv") and "qwen_kv_on" not in results:
        t = benchmark_ar(prompts, device, True, QWEN, "Qwen")
        results["qwen_kv_on"] = {"per_sample": t, "median": float(np.median(t)), "mean": float(np.mean(t))}
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  Qwen KV-on  : median={results['qwen_kv_on']['median']:.2f}s  mean={results['qwen_kv_on']['mean']:.2f}s")

    if args.mode in ("all","qwen_nokv") and "qwen_kv_off" not in results:
        t = benchmark_ar(prompts, device, False, QWEN, "Qwen")
        results["qwen_kv_off"] = {"per_sample": t, "median": float(np.median(t)), "mean": float(np.mean(t))}
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  Qwen KV-off : median={results['qwen_kv_off']['median']:.2f}s  mean={results['qwen_kv_off']['mean']:.2f}s")

    if args.mode in ("all","llama_kv") and "llama_kv_on" not in results:
        t = benchmark_ar(prompts, device, True, LLAMA3, "Llama-3.1")
        results["llama_kv_on"] = {"per_sample": t, "median": float(np.median(t)), "mean": float(np.mean(t))}
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  Llama-3.1 KV-on  : median={results['llama_kv_on']['median']:.2f}s  mean={results['llama_kv_on']['mean']:.2f}s")

    if args.mode in ("all","llama_nokv") and "llama_kv_off" not in results:
        t = benchmark_ar(prompts, device, False, LLAMA3, "Llama-3.1")
        results["llama_kv_off"] = {"per_sample": t, "median": float(np.median(t)), "mean": float(np.mean(t))}
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  Llama-3.1 KV-off : median={results['llama_kv_off']['median']:.2f}s  mean={results['llama_kv_off']['mean']:.2f}s")

    if args.mode in ("all","sde"):
        nfes = [8, 16, 32, 64]
        nfes = [n for n in nfes if f"sde_nfe{n}" not in results]
        if nfes:
            sde_times = benchmark_sde(prompts, device, nfes)
            for nfe, t in sde_times.items():
                results[f"sde_nfe{nfe}"] = {"per_sample": t,
                                             "median": float(np.median(t)),
                                             "mean":   float(np.mean(t))}
                print(f"  SDE NFE={nfe:3d}  : median={results[f'sde_nfe{nfe}']['median']:.2f}s  mean={results[f'sde_nfe{nfe}']['mean']:.2f}s")
            json.dump(results, open(out_path, "w"), indent=2)

    # also store per-dataset breakdown
    breakdown = results.get("__per_dataset__", {})
    for k, v in results.items():
        if k == "__per_dataset__" or not isinstance(v, dict) or "per_sample" not in v:
            continue
        per = v["per_sample"]
        ds_times = {}
        for i, p in enumerate(prompts):
            ds_times.setdefault(p["dataset"], []).append(per[i])
        breakdown[k] = {ds: {"median": float(np.median(t)), "mean": float(np.mean(t))} for ds, t in ds_times.items()}
    results["__per_dataset__"] = breakdown
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
