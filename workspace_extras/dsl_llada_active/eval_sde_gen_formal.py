"""Formal SDE vs Remask generation quality evaluation.

Single-model, single-config generation + metrics. Shell scripts orchestrate.

Usage:
    # Generate with remask
    python eval_sde_gen_formal.py --model_key original --method remask --nfe 64 \
        --prompts eval_data/sde_prompts_200.json --gpu 0 --tag prompted

    # Generate with SDE
    python eval_sde_gen_formal.py --model_key b1 --method sde --nfe 64 \
        --prompts eval_data/sde_prompts_200.json --gpu 0 --tag prompted

    # Prefix continuation
    python eval_sde_gen_formal.py --model_key b1 --method sde --nfe 64 \
        --prefixes eval_data/wikitext_prefix_200.json --gpu 0 --tag prefix

    # Diversity test (multiple seeds)
    python eval_sde_gen_formal.py --model_key b1 --method sde --nfe 64 \
        --prompts eval_data/sde_prompts_200.json --gpu 0 --tag diversity \
        --seeds 42,123,456,789,1024 --max_prompts 20

    # Compute metrics only (on existing generated texts)
    python eval_sde_gen_formal.py --metrics eval_results/sde_gen_formal/some_result.json --gpu 0
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_root, "LLaDA"))
sys.path.insert(0, _script_dir)
from generate import generate
from safetensors.torch import load_file
from dsl_modules import LoRALinear

MASK_ID = 126336
TOP_K = 512
OUTPUT_DIR = os.path.join(_root, "eval_results", "sde_gen_formal")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_ffout_lora_if_present(model, ckpt, device):
    """Restore ff_out LoRA checkpoints saved by the DSL trainer."""
    if not os.path.isdir(ckpt):
        return False
    tensors = {}
    wanted = {
        "model.transformer.ff_out.base.weight",
        "model.transformer.ff_out.lora_A",
        "model.transformer.ff_out.lora_B",
    }
    for fname in sorted(os.listdir(ckpt)):
        if not fname.endswith(".safetensors"):
            continue
        st = load_file(os.path.join(ckpt, fname), device=str(device))
        for key in wanted:
            if key in st:
                tensors[key] = st[key]
        del st
    if not wanted.issubset(tensors):
        return False

    base = model.model.transformer.ff_out
    r = tensors["model.transformer.ff_out.lora_A"].shape[1]
    lora = LoRALinear(base, r=r).to(device=device, dtype=base.weight.dtype)
    with torch.no_grad():
        lora.base.weight.copy_(
            tensors["model.transformer.ff_out.base.weight"].to(lora.base.weight.dtype)
        )
        lora.lora_A.copy_(tensors["model.transformer.ff_out.lora_A"].to(lora.lora_A.dtype))
        lora.lora_B.copy_(tensors["model.transformer.ff_out.lora_B"].to(lora.lora_B.dtype))
    model.model.transformer.ff_out = lora
    print(f"  Loaded ff_out LoRA: r={r}, scale={lora.scale}", flush=True)
    return True


# ===================================================================
# Model loading
# ===================================================================

def load_model_and_components(model_key, device):
    """Load model, tokenizer, and SDE components if available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    registry = json.load(open(os.path.join(_root, "eval_results", "registry.json")))
    entry = registry[model_key]
    ckpt = entry["path"]
    if entry.get("type") == "local":
        ckpt = os.path.join(_root, ckpt)

    print(f"[{model_key}] Loading from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    load_ffout_lora_if_present(model, ckpt, device)

    # Load SDE components from safetensors
    sde_components = None
    if entry.get("dsl"):
        for fname in sorted(os.listdir(ckpt)):
            if fname.endswith(".safetensors"):
                st = load_file(os.path.join(ckpt, fname), device=str(device))
                if "noise_embed.weight" in st:
                    ne = st["noise_embed.weight"].float()
                    sde_components = {
                        "ne": ne,
                        "lb": st["converter.logit_bias"].float(),
                        "bw": st["converter.backbone_embedding.weight"].float(),
                        "bb": st["converter.backbone_embedding.bias"].float(),
                        "K": torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0),
                        "beta_train": float(st["converter.beta"]),
                    }
                    print(f"  SDE components loaded (beta_train={sde_components['beta_train']:.3f})")
                del st

    return model, tokenizer, entry, sde_components


# ===================================================================
# Generation: Remask
# ===================================================================

def generate_remask(model, tokenizer, text, device, nfe=64, gen_length=256,
                    block_length=256, suppress_eos=False, logits_eos_inf=False,
                    temperature=0.0, remasking="low_confidence"):
    """Discrete remasking generation."""
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        out = generate(
            model, input_ids, torch.ones_like(input_ids),
            steps=nfe, gen_length=gen_length, block_length=block_length,
            temperature=temperature, cfg_scale=0.0, remasking=remasking,
            eos_suppress_ratio=1.0 if suppress_eos else 0.0,
            logits_eos_inf=logits_eos_inf,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def generate_remask_prefix(model, tokenizer, prefix_text, device, nfe=64,
                           gen_length=256, logits_eos_inf=False,
                           temperature=0.0, remasking="low_confidence"):
    """Remask generation from a text prefix (no chat template)."""
    encoded = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        out = generate(
            model, input_ids, torch.ones_like(input_ids),
            steps=nfe, gen_length=gen_length, block_length=256,
            temperature=temperature, cfg_scale=0.0, remasking=remasking,
            logits_eos_inf=logits_eos_inf,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


# ===================================================================
# Generation: SDE (correct formulation from sde_sweep_single.py)
# ===================================================================

def generate_sde(model, tokenizer, text, device, sde_comp, nfe=64,
                 gen_length=256, beta_infer=None, noise_scale=None,
                 schedule="sensitive", seed=42):
    """SDE generation with correct drift/diffusion."""
    ne = sde_comp["ne"]
    lb = sde_comp["lb"]
    bw = sde_comp["bw"]
    bb = sde_comp["bb"]
    K = sde_comp["K"]
    nd = ne.shape[1]

    if beta_infer is None:
        beta_infer = sde_comp["beta_train"] * 2.0
    if noise_scale is None:
        noise_scale = _adaptive_ns(nfe)
    bv = beta_infer

    steps = nfe // 2  # Heun: NFE = steps * 2

    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    P = prompt_ids.shape[1]

    pe = model.model.transformer.wte(prompt_ids).float()
    dummy = torch.zeros(1, P + gen_length, dtype=torch.long, device=device)

    # SNR schedule
    snr_min = 0.01
    if schedule == "sensitive":
        n1 = max(1, int(steps * 0.05))
        n2 = max(1, int(steps * 0.90))
        n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(100), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(
            torch.linspace(math.log(snr_min), math.log(100), steps + 1)
        ).to(device)

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

    ns = noise_scale
    for i in range(len(snrs) - 1):
        s, s_next = snrs[i], snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    # Final decode
    z_f = snrs[-1] * y
    cl_f = bv * (z_f.float() @ K.T) + lb
    pf = F.softmax(cl_f.float(), dim=-1)
    hf = F.linear(pf, bw, bb).to(torch.bfloat16)
    ef = torch.cat([pe.to(torch.bfloat16), hf], dim=1)
    with torch.no_grad():
        lo = model(input_ids=dummy, inputs_embeds=ef).logits[:, P:, :].float()
    return tokenizer.decode(lo.argmax(dim=-1)[0], skip_special_tokens=True)


def generate_sde_prefix(model, tokenizer, prefix_text, device, sde_comp,
                         nfe=64, gen_length=256, beta_infer=None,
                         noise_scale=None, schedule="sensitive", seed=42):
    """SDE generation from text prefix (no chat template)."""
    ne = sde_comp["ne"]
    lb = sde_comp["lb"]
    bw = sde_comp["bw"]
    bb = sde_comp["bb"]
    K = sde_comp["K"]
    nd = ne.shape[1]

    if beta_infer is None:
        beta_infer = sde_comp["beta_train"] * 2.0
    if noise_scale is None:
        noise_scale = _adaptive_ns(nfe)
    bv = beta_infer
    steps = nfe // 2

    encoded = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    P = prompt_ids.shape[1]

    pe = model.model.transformer.wte(prompt_ids).float()
    dummy = torch.zeros(1, P + gen_length, dtype=torch.long, device=device)

    snr_min = 0.01
    if schedule == "sensitive":
        n1 = max(1, int(steps * 0.05))
        n2 = max(1, int(steps * 0.90))
        n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(100), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(
            torch.linspace(math.log(snr_min), math.log(100), steps + 1)
        ).to(device)

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

    ns = noise_scale
    for i in range(len(snrs) - 1):
        s, s_next = snrs[i], snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    z_f = snrs[-1] * y
    cl_f = bv * (z_f.float() @ K.T) + lb
    pf = F.softmax(cl_f.float(), dim=-1)
    hf = F.linear(pf, bw, bb).to(torch.bfloat16)
    ef = torch.cat([pe.to(torch.bfloat16), hf], dim=1)
    with torch.no_grad():
        lo = model(input_ids=dummy, inputs_embeds=ef).logits[:, P:, :].float()
    return tokenizer.decode(lo.argmax(dim=-1)[0], skip_special_tokens=True)


def _adaptive_ns(nfe):
    """Noise scale adaptive to NFE (from validated param rules)."""
    if nfe <= 8:
        return 0.1
    elif nfe <= 16:
        return 0.05
    elif nfe <= 32:
        return 0.01
    else:
        return 0.005


def _sde_schedule(model_key):
    """Schedule type per model (from validated param search)."""
    if model_key in ("b1", "hf_beta1", "uu_frozen_1k", "uu_trainable_1k", "uu_trainable_5k"):
        return "sensitive"
    return "uniform"


# ===================================================================
# Metrics
# ===================================================================

def compute_metrics(texts, device, ref_texts=None, ppl_max_words=128):
    """Compute GenPPL, D2, D3, Rep, Len. Optionally MAUVE if ref_texts given."""
    valid = [t for t in texts if t.strip()]
    if not valid:
        return {"gen_ppl": None, "d2": 0, "d3": 0, "rep_rate": 0, "avg_len": 0,
                "n_valid": 0, "mauve": None}

    # GenPPL
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
    nlls = []
    for text in valid:
        words = text.split()[:ppl_max_words]
        truncated = " ".join(words)
        if not truncated.strip():
            continue
        enc = gpt2_tok(truncated, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            nlls.append(gpt2_model(input_ids, labels=input_ids).loss.item())
    del gpt2_model
    torch.cuda.empty_cache()
    gen_ppl = float(np.exp(np.mean(nlls))) if nlls else None

    # D2, D3
    def distinct_n(txts, n):
        ngrams = []
        for t in txts:
            ws = t.split()
            for i in range(len(ws) - n + 1):
                ngrams.append(tuple(ws[i:i+n]))
        return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    # Rep
    total = rep = 0
    for t in valid:
        ws = t.split()
        for i in range(1, len(ws)):
            total += 1
            if ws[i] == ws[i-1]:
                rep += 1
    rep_rate = rep / total if total > 0 else 0.0

    result = {
        "gen_ppl": round(gen_ppl, 2) if gen_ppl else None,
        "d2": round(distinct_n(valid, 2), 4),
        "d3": round(distinct_n(valid, 3), 4),
        "rep_rate": round(rep_rate, 4),
        "avg_len": round(float(np.mean([len(t.split()) for t in valid])), 1),
        "n_valid": len(valid),
        "ppl_max_words": int(ppl_max_words),
    }

    # MAUVE (optional)
    if ref_texts:
        try:
            import mauve
            ref_valid = [t for t in ref_texts[:len(valid)] if t.strip()]
            mauve_result = mauve.compute_mauve(
                p_text=ref_valid, q_text=valid[:len(ref_valid)],
                device_id=int(str(device).split(":")[-1]) if ":" in str(device) else 0,
                max_text_length=256,
            )
            result["mauve"] = round(float(mauve_result.mauve), 4)
        except Exception as e:
            print(f"  MAUVE failed: {e}")
            result["mauve"] = None
    else:
        result["mauve"] = None

    return result


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Formal SDE vs Remask generation evaluation")
    parser.add_argument("--model_key", type=str, help="Model key from registry")
    parser.add_argument("--method", choices=["remask", "rmdm", "sde"], default="remask")
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", type=str, default="prompted", help="Experiment tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-sep seeds for diversity test")
    parser.add_argument("--max_prompts", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard id for parallel generation")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of generation shards")
    parser.add_argument("--eos_inf", action="store_true", help="Set EOS logits to -inf (LLaDA Appendix B.4)")
    parser.add_argument("--block_length", type=int, default=256, help="Block length for remask (32=semi-AR)")
    parser.add_argument("--suppress_eos", action="store_true", help="Suppress EOS in confidence ranking")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for discrete remasking")
    parser.add_argument("--sde_top_k", type=int, default=None,
                        help="Override top-k tokens used in the SDE denoiser expectation")
    parser.add_argument("--sde_schedule", choices=["sensitive", "uniform"], default=None,
                        help="Override SDE SNR schedule")
    parser.add_argument("--sde_beta_infer", type=float, default=None,
                        help="Override converter beta used at SDE inference")
    parser.add_argument("--sde_noise_scale", type=float, default=None,
                        help="Override stochastic noise scale used at each SDE step")

    # Data sources (mutually exclusive)
    parser.add_argument("--prompts", type=str, help="Path to prompts JSON")
    parser.add_argument("--prefixes", type=str, help="Path to prefix JSON")

    # Metrics only
    parser.add_argument("--metrics", type=str, help="Compute metrics on existing result file")
    parser.add_argument("--ppl_max_words", type=int, default=128,
                        help="Word budget used before GPT-2 GenPPL scoring")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    global TOP_K
    if args.sde_top_k is not None:
        TOP_K = args.sde_top_k

    # === Metrics-only mode ===
    if args.metrics:
        print(f"Computing metrics on {args.metrics}")
        data = json.load(open(args.metrics))
        texts = data.get("texts", data.get("generated_texts", []))
        ref_texts = data.get("ref_texts", None)
        metrics = compute_metrics(texts, device, ref_texts, ppl_max_words=args.ppl_max_words)
        print(f"  Results: {metrics}")
        metrics_key = "metrics" if args.ppl_max_words == 128 else f"metrics_ppl{args.ppl_max_words}"
        data[metrics_key] = metrics
        with open(args.metrics, "w") as f:
            json.dump(data, f, indent=2)
        return

    # === Generation mode ===
    assert args.model_key, "--model_key required for generation"
    assert args.prompts or args.prefixes, "Need --prompts or --prefixes"

    # Load data
    if args.prompts:
        data = json.load(open(args.prompts))
        if args.max_prompts:
            data = data[:args.max_prompts]
        all_input_texts = [item["prompt"] for item in data]
        mode = "prompted"
    else:
        data = json.load(open(args.prefixes))
        if args.max_prompts:
            data = data[:args.max_prompts]
        all_input_texts = [item["prefix"] for item in data]
        all_ref_texts = [item.get("continuation", "") for item in data]
        mode = "prefix"

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    selected = [i for i in range(len(all_input_texts)) if i % args.num_shards == args.shard_id]
    input_texts = [all_input_texts[i] for i in selected]
    if mode == "prefix":
        ref_texts = [all_ref_texts[i] for i in selected]

    # Load model
    model, tokenizer, entry, sde_comp = load_model_and_components(args.model_key, device)

    # Determine seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    # SDE schedule per model
    sde_sch = args.sde_schedule or _sde_schedule(args.model_key)

    for seed in seeds:
        seed_tag = f"_seed{seed}" if len(seeds) > 1 else ""
        eos_tag = "_eosInf" if args.eos_inf else ""
        blk_tag = f"_b{args.block_length}" if args.block_length != 256 else ""
        sup_tag = "_noEOS" if args.suppress_eos else ""
        temp_tag = f"_t{str(args.temperature).replace('.', 'p')}" if args.temperature != 0.0 else ""
        shard_tag = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
        remasking = "remdm_conf" if args.method == "rmdm" else "low_confidence"
        out_name = f"{args.tag}_{args.model_key}_{args.method}{eos_tag}{blk_tag}{sup_tag}{temp_tag}_nfe{args.nfe}_gen{args.gen_length}{seed_tag}{shard_tag}.json"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):
            print(f"  Skipping {out_name} (exists)")
            continue

        print(f"\n{'='*60}")
        print(f"  {out_name}")
        print(f"  model={args.model_key} method={args.method} nfe={args.nfe} "
              f"gen={args.gen_length} seed={seed} n={len(input_texts)}/{len(all_input_texts)} "
              f"shard={args.shard_id}/{args.num_shards}")
        print(f"{'='*60}")

        texts = []
        trace = []  # Per-sample trace for analysis
        t_total = time.time()

        for local_idx, inp in enumerate(tqdm(input_texts, desc=out_name)):
            idx = selected[local_idx]
            t0 = time.time()
            try:
                if args.method == "sde":
                    assert sde_comp, f"Model {args.model_key} has no SDE components"
                    if mode == "prefix":
                        text = generate_sde_prefix(
                            model, tokenizer, inp, device, sde_comp,
                            nfe=args.nfe, gen_length=args.gen_length,
                            beta_infer=args.sde_beta_infer,
                            noise_scale=args.sde_noise_scale,
                            schedule=sde_sch, seed=seed,
                        )
                    else:
                        text = generate_sde(
                            model, tokenizer, inp, device, sde_comp,
                            nfe=args.nfe, gen_length=args.gen_length,
                            beta_infer=args.sde_beta_infer,
                            noise_scale=args.sde_noise_scale,
                            schedule=sde_sch, seed=seed,
                        )
                else:
                    if mode == "prefix":
                        text = generate_remask_prefix(
                            model, tokenizer, inp, device,
                            nfe=args.nfe, gen_length=args.gen_length,
                            logits_eos_inf=args.eos_inf,
                            temperature=args.temperature,
                            remasking=remasking,
                        )
                    else:
                        text = generate_remask(
                            model, tokenizer, inp, device,
                            nfe=args.nfe, gen_length=args.gen_length,
                            block_length=args.block_length,
                            suppress_eos=args.suppress_eos,
                            logits_eos_inf=args.eos_inf,
                            temperature=args.temperature,
                            remasking=remasking,
                        )
            except Exception as e:
                print(f"  Error on {idx}: {e}")
                text = ""

            elapsed = time.time() - t0
            texts.append(text)
            trace.append({
                "idx": idx,
                "local_idx": local_idx,
                "time_s": round(elapsed, 3),
                "n_words": len(text.split()),
                "input_preview": inp[:80],
                "output_preview": text[:120],
            })

        total_time = time.time() - t_total

        # Save result
        result = {
            "model_key": args.model_key,
            "method": args.method,
            "nfe": args.nfe,
            "gen_length": args.gen_length,
            "seed": seed,
            "tag": args.tag,
            "mode": mode,
            "n_samples": len(texts),
            "n_samples_total": len(all_input_texts),
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "selected_indices": selected,
            "temperature": args.temperature,
            "remasking": remasking if args.method in ("remask", "rmdm") else None,
            "total_time_s": round(total_time, 1),
            "avg_time_per_sample": round(total_time / max(len(texts), 1), 2),
            "timestamp": datetime.datetime.now().isoformat(),
            "sde_params": {
                "beta_infer": (
                    args.sde_beta_infer
                    if args.sde_beta_infer is not None
                    else (sde_comp["beta_train"] * 2.0 if sde_comp else None)
                ),
                "noise_scale": (
                    args.sde_noise_scale
                    if args.sde_noise_scale is not None
                    else (_adaptive_ns(args.nfe) if args.method == "sde" else None)
                ),
                "schedule": sde_sch if args.method == "sde" else None,
                "top_k": TOP_K if args.method == "sde" else None,
            },
            "texts": texts,
            "trace": trace,
        }
        if mode == "prefix":
            result["ref_texts"] = ref_texts

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved {out_name} ({len(texts)} samples, {total_time:.0f}s)")

    # Clean up model
    del model
    if sde_comp:
        del sde_comp
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
