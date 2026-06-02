"""Summarization benchmark evaluation across multiple datasets.

Evaluates SDE and Orig+eos+b32 on summarization tasks with ROUGE metrics.

Usage:
    python dsl_llada/eval_summarization.py \
        --dataset xsum --method sde --model_key b1 --gpu 0
    python dsl_llada/eval_summarization.py \
        --dataset cnn_dailymail --method remask --model_key original --gpu 1 \
        --eos_inf --block_length 32
"""
import argparse
import importlib.util
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

_xdm_spec = importlib.util.spec_from_file_location(
    "llada_xdlm_generate",
    os.path.join(_root, "LLaDA-XDLM", "generate.py"),
)
_xdm_mod = importlib.util.module_from_spec(_xdm_spec)
_xdm_spec.loader.exec_module(_xdm_mod)
generate_xdm = _xdm_mod.generate_xdm

MASK_ID = 126336
TOP_K = 512
OUTPUT_DIR = os.path.join(_root, "eval_results", "summarization")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_FILES = {
    "xsum": "eval_data/summarization_100.json",
    "cnn_dailymail": "eval_data/cnn_dailymail_100.json",
    "billsum": "eval_data/billsum_100.json",
    "aeslc": "eval_data/aeslc_100.json",
    "pubmed": "eval_data/pubmed_100.json",
    "arxiv": "eval_data/arxiv_100.json",
}

# Gen length per dataset (shorter for AESLC which is subject lines)
GEN_LENGTHS = {
    "xsum": 128,
    "cnn_dailymail": 256,
    "billsum": 256,
    "aeslc": 64,
    "pubmed": 256,
    "arxiv": 256,
}


def format_prompt(tokenizer, text, prompt_format):
    if prompt_format == "plain":
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


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


def load_model(model_key, device):
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


def adaptive_ns(nfe):
    if nfe <= 8: return 0.1
    elif nfe <= 16: return 0.05
    elif nfe <= 32: return 0.01
    else: return 0.005


def generate_sde_text(model, tokenizer, text, device, sde_comp, nfe=64,
                      gen_length=128, seed=42, beta_infer=None,
                      noise_scale=None, schedule="sensitive",
                      snr_min=0.01, snr_max=100.0,
                      sensitive_low=7.0, sensitive_high=74.0,
                      prompt_format="chat"):
    ne = sde_comp["ne"]
    lb = sde_comp["lb"]
    bw = sde_comp["bw"]
    bb = sde_comp["bb"]
    K = sde_comp["K"]
    nd = ne.shape[1]
    bv = sde_comp["beta_train"] * 2.0 if beta_infer is None else beta_infer
    ns = adaptive_ns(nfe) if noise_scale is None else noise_scale
    steps = nfe // 2

    formatted = format_prompt(tokenizer, text, prompt_format)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    P = prompt_ids.shape[1]

    pe = model.model.transformer.wte(prompt_ids).float()
    dummy = torch.zeros(1, P + gen_length, dtype=torch.long, device=device)

    snr_min = max(float(snr_min), 1e-6)
    snr_max = max(float(snr_max), snr_min * 1.0001)
    sensitive_low = min(max(float(sensitive_low), snr_min * 1.0001), snr_max)
    sensitive_high = min(max(float(sensitive_high), sensitive_low * 1.0001), snr_max)
    if schedule == "uniform":
        snrs = torch.exp(
            torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1)
        ).to(device)
    else:
        n1 = max(1, int(steps * 0.05))
        n2 = max(1, int(steps * 0.90))
        n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(sensitive_low), n1 + 1)),
            torch.exp(torch.linspace(math.log(sensitive_low), math.log(sensitive_high), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(sensitive_high), math.log(snr_max), n3 + 1))[1:],
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


def generate_remask_text(model, tokenizer, text, device, nfe=64,
                         gen_length=128, block_length=256,
                         eos_inf=False, suppress_eos=False,
                         temperature=0.0, remasking="low_confidence",
                         prompt_format="chat"):
    formatted = format_prompt(tokenizer, text, prompt_format)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # LLaDA generate() asserts gen_length % block_length == 0. A caller that passes a
    # block_length > gen_length (e.g. default 256 on XSum's 128-token budget) really
    # means "single block = no block structure"; honor that intent.
    bl = min(block_length, gen_length)
    # Further snap to a divisor of gen_length if needed (e.g. block=32 on gen=64 already OK;
    # block=32 on gen=96 would need 48).
    if gen_length % bl != 0:
        # pick the largest divisor of gen_length that is <= the requested bl
        for cand in range(bl, 0, -1):
            if gen_length % cand == 0:
                bl = cand; break

    with torch.no_grad():
        out = generate(
            model, input_ids,
            steps=nfe, gen_length=gen_length, block_length=bl,
            temperature=temperature, cfg_scale=0.0, remasking=remasking,
            eos_suppress_ratio=1.0 if suppress_eos else 0.0,
            logits_eos_inf=eos_inf,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def generate_xdm_text(model, tokenizer, text, device, nfe=64,
                      gen_length=128, block_length=256,
                      eos_inf=False, temperature=0.0, k1=0.1,
                      prompt_format="plain"):
    formatted = format_prompt(tokenizer, text, prompt_format)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    bl = min(block_length, gen_length)
    if gen_length % bl != 0:
        for cand in range(bl, 0, -1):
            if gen_length % cand == 0:
                bl = cand
                break

    with torch.no_grad():
        out = generate_xdm(
            model, input_ids, attention_mask,
            steps=nfe, gen_length=gen_length, block_length=bl,
            temperature=temperature, cfg_scale=0.0,
            remasking="low_confidence",
            logits_eos_inf=eos_inf,
            confidence_eos_eot_inf=False,
            k1=k1,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def compute_rouge(generated, references):
    """Returns (aggregate_dict, per_sample_list) so callers can do case analysis."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    except ModuleNotFoundError:
        scorer = None

    def toks(text):
        return text.lower().split()

    def rouge_n_f(ref, gen, n):
        rt, gt = toks(ref), toks(gen)
        if len(rt) < n or len(gt) < n:
            return 0.0
        def grams(xs):
            out = {}
            for i in range(len(xs) - n + 1):
                g = tuple(xs[i:i+n])
                out[g] = out.get(g, 0) + 1
            return out
        rg, gg = grams(rt), grams(gt)
        overlap = sum(min(gg.get(g, 0), c) for g, c in rg.items())
        if overlap == 0:
            return 0.0
        prec = overlap / sum(gg.values())
        rec = overlap / sum(rg.values())
        return 2 * prec * rec / (prec + rec)

    def rouge_l_f(ref, gen):
        rt, gt = toks(ref), toks(gen)
        if not rt or not gt:
            return 0.0
        prev = [0] * (len(gt) + 1)
        for r in rt:
            cur = [0]
            for j, g in enumerate(gt, start=1):
                cur.append(prev[j - 1] + 1 if r == g else max(prev[j], cur[-1]))
            prev = cur
        lcs = prev[-1]
        if lcs == 0:
            return 0.0
        prec = lcs / len(gt)
        rec = lcs / len(rt)
        return 2 * prec * rec / (prec + rec)

    per = []
    r1s, r2s, rls = [], [], []
    for gen, ref in zip(generated, references):
        if not gen.strip():
            per.append({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
            r1s.append(0.0); r2s.append(0.0); rls.append(0.0)
            continue
        if scorer is not None:
            s = scorer.score(ref, gen)
            r1 = s['rouge1'].fmeasure
            r2 = s['rouge2'].fmeasure
            rl = s['rougeL'].fmeasure
        else:
            r1 = rouge_n_f(ref, gen, 1)
            r2 = rouge_n_f(ref, gen, 2)
            rl = rouge_l_f(ref, gen)
        per.append({"rouge1": round(r1 * 100, 2),
                    "rouge2": round(r2 * 100, 2),
                    "rougeL": round(rl * 100, 2)})
        r1s.append(r1); r2s.append(r2); rls.append(rl)
    agg = {
        "rouge1": round(np.mean(r1s) * 100, 2),
        "rouge2": round(np.mean(r2s) * 100, 2),
        "rougeL": round(np.mean(rls) * 100, 2),
    }
    return agg, per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_FILES.keys()))
    parser.add_argument(
        "--method",
        required=True,
        choices=["sde", "remask", "rmdm", "rmdm_conf", "rmdm_cap", "xdm"],
        help="Generation method. rmdm is an alias for confidence-based ReMDM; xdm uses XDLM clean-token replacement.",
    )
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos_inf", action="store_true")
    parser.add_argument("--block_length", type=int, default=256)
    parser.add_argument("--suppress_eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature for discrete remasking")
    parser.add_argument("--xdm_k1", type=float, default=0.1,
                        help="XDLM clean-token replacement ratio used by generate_xdm")
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--data_file", type=str, default=None,
                        help="override path to dataset JSON (relative to repo root)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="0-indexed shard id (for data-parallel eval)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="total number of shards; data is split by i % num_shards == shard_id")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="optional extra tag appended to output filename")
    parser.add_argument("--prompt_format", choices=["auto", "chat", "plain"], default="auto",
                        help="chat for instruct checkpoints, plain for base checkpoints; auto uses plain for xdlm")
    parser.add_argument("--limit", type=int, default=None,
                        help="limit total samples (applied before sharding) for debugging")
    parser.add_argument("--sde_beta_infer", type=float, default=None,
                        help="override SDE converter sharpness for inference")
    parser.add_argument("--sde_noise_scale", type=float, default=None,
                        help="override SDE stochastic noise scale")
    parser.add_argument("--sde_top_k", type=int, default=None,
                        help="override top-k tokens used in the SDE denoiser expectation")
    parser.add_argument("--sde_schedule", choices=["sensitive", "uniform"], default="sensitive",
                        help="override SDE SNR schedule")
    parser.add_argument("--sde_snr_min", type=float, default=0.01,
                        help="starting SNR for SDE inference")
    parser.add_argument("--sde_snr_max", type=float, default=100.0,
                        help="ending SNR for SDE inference")
    parser.add_argument("--sde_sensitive_low", type=float, default=7.0,
                        help="lower transition SNR for sensitive SDE schedule")
    parser.add_argument("--sde_sensitive_high", type=float, default=74.0,
                        help="upper transition SNR for sensitive SDE schedule")
    args = parser.parse_args()
    global TOP_K
    if args.sde_top_k is not None:
        TOP_K = args.sde_top_k

    device = torch.device(f"cuda:{args.gpu}")
    gen_length = args.gen_length or GEN_LENGTHS[args.dataset]
    prompt_format = "plain" if args.prompt_format == "auto" and args.model_key == "xdlm" else args.prompt_format
    if prompt_format == "auto":
        prompt_format = "chat"

    # Load dataset
    data_path_rel = args.data_file or DATASET_FILES[args.dataset]
    data_path = os.path.join(_root, data_path_rel)
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {data_path_rel}")
    if args.limit is not None:
        data = data[: args.limit]
        print(f"  limited to first {len(data)}")
    # Keep absolute index so shards can be merged back
    indexed = [(i, item) for i, item in enumerate(data)]
    if args.num_shards > 1:
        indexed = [(i, it) for (i, it) in indexed if i % args.num_shards == args.shard_id]
        print(f"  shard {args.shard_id}/{args.num_shards}: {len(indexed)} samples "
              f"(global idx {indexed[0][0]}..{indexed[-1][0]})")

    # Load model
    model, tokenizer, entry, sde_comp = load_model(args.model_key, device)

    # Generate
    texts = []
    global_idxs = []
    t0 = time.time()
    for local_i, (g_idx, item) in enumerate(tqdm(indexed, desc=f"{args.dataset}/{args.method}")):
        prompt = item["prompt"]
        try:
            if args.method == "sde":
                text = generate_sde_text(
                    model, tokenizer, prompt, device, sde_comp,
                    nfe=args.nfe, gen_length=gen_length, seed=args.seed,
                    beta_infer=args.sde_beta_infer,
                    noise_scale=args.sde_noise_scale,
                    schedule=args.sde_schedule,
                    snr_min=args.sde_snr_min,
                    snr_max=args.sde_snr_max,
                    sensitive_low=args.sde_sensitive_low,
                    sensitive_high=args.sde_sensitive_high,
                    prompt_format=prompt_format,
                )
            elif args.method == "xdm":
                text = generate_xdm_text(
                    model, tokenizer, prompt, device,
                    nfe=args.nfe, gen_length=gen_length,
                    block_length=args.block_length,
                    eos_inf=args.eos_inf,
                    temperature=args.temperature,
                    k1=args.xdm_k1,
                    prompt_format=prompt_format,
                )
            else:
                remasking = {
                    "remask": "low_confidence",
                    "rmdm": "remdm_conf",
                    "rmdm_conf": "remdm_conf",
                    "rmdm_cap": "remdm_cap",
                }[args.method]
                text = generate_remask_text(
                    model, tokenizer, prompt, device,
                    nfe=args.nfe, gen_length=gen_length,
                    block_length=args.block_length,
                    eos_inf=args.eos_inf, suppress_eos=args.suppress_eos,
                    temperature=args.temperature, remasking=remasking,
                    prompt_format=prompt_format,
                )
        except Exception as e:
            print(f"  Sample local={local_i} global={g_idx} failed: {e}")
            text = ""
        texts.append(text)
        global_idxs.append(g_idx)

    elapsed = time.time() - t0
    n_here = len(texts)
    print(f"Generation done: {elapsed:.1f}s ({elapsed / max(n_here,1):.1f}s/sample)")

    # Compute ROUGE (per-sample + aggregate on this shard)
    references = [item["reference"] for (_, item) in indexed]
    prompts = [item["prompt"] for (_, item) in indexed]
    rouge, per_sample_rouge = compute_rouge(texts, references)
    print(f"ROUGE-1: {rouge['rouge1']}, ROUGE-2: {rouge['rouge2']}, ROUGE-L: {rouge['rougeL']}")

    # Stats
    valid = sum(1 for t in texts if len(t.strip()) > 0)
    word_counts = [len(t.split()) for t in texts]
    avg_words = float(np.mean(word_counts)) if n_here > 0 else 0.0
    degen = sum(1 for t in texts if any(t.count(p) > 3 for p in [':::','...','{}','""','quest','\u200b']))

    # Per-sample records: indexed by global id so shards merge cleanly
    samples = [
        {
            "id": g_idx,
            "reference": ref,
            "generated": gen,
            "gen_words": wc,
            "rouge1": pr["rouge1"],
            "rouge2": pr["rouge2"],
            "rougeL": pr["rougeL"],
        }
        for g_idx, ref, gen, wc, pr in zip(global_idxs, references, texts, word_counts, per_sample_rouge)
    ]

    # Build method tag
    method_tag = f"{args.model_key}_{args.method}"
    if args.eos_inf:
        method_tag += "_eosInf"
    if args.block_length != 256:
        method_tag += f"_b{args.block_length}"
    if args.suppress_eos:
        method_tag += "_noEOS"
    if args.temperature != 0.0:
        method_tag += f"_t{str(args.temperature).replace('.', 'p')}"
    if args.out_tag:
        method_tag += f"_{args.out_tag}"

    result = {
        "dataset": args.dataset,
        "data_file": data_path_rel,
        "method": method_tag,
        "model_key": args.model_key,
        "gen_method": args.method,
        "nfe": args.nfe,
        "gen_length": gen_length,
        "temperature": args.temperature,
        "prompt_format": prompt_format,
        "xdm_k1": args.xdm_k1 if args.method == "xdm" else None,
        "sde_params": {
            "beta_infer": (
                args.sde_beta_infer
                if args.sde_beta_infer is not None
                else (sde_comp["beta_train"] * 2.0 if sde_comp else None)
            ),
            "noise_scale": (
                args.sde_noise_scale
                if args.sde_noise_scale is not None
                else (adaptive_ns(args.nfe) if args.method == "sde" else None)
            ),
            "top_k": TOP_K if args.method == "sde" else None,
            "schedule": args.sde_schedule if args.method == "sde" else None,
            "snr_min": args.sde_snr_min if args.method == "sde" else None,
            "snr_max": args.sde_snr_max if args.method == "sde" else None,
            "sensitive_low": args.sde_sensitive_low if args.method == "sde" else None,
            "sensitive_high": args.sde_sensitive_high if args.method == "sde" else None,
            "remasking": {
                "remask": "low_confidence",
                "rmdm": "remdm_conf",
                "rmdm_conf": "remdm_conf",
                "rmdm_cap": "remdm_cap",
                "xdm": f"generate_xdm(k1={args.xdm_k1})",
            }.get(args.method),
        },
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_samples_total": len(data),
        "n_samples_here": n_here,
        "valid": valid,
        "avg_words": round(avg_words, 2),
        "degenerate_pct": round(degen / max(n_here,1) * 100, 2),
        **rouge,
        "time_s": round(elapsed, 1),
        "samples": samples,
    }

    shard_suffix = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    out_file = os.path.join(
        OUTPUT_DIR, f"{args.dataset}_{method_tag}_nfe{args.nfe}{shard_suffix}.json"
    )
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
