"""Block SDE generation for DSL-LLaDA.

Combines block diffusion (sequential blocks) with SDE (continuous denoising within each block).
Each block runs SDE from noise→convergence, then commits discrete tokens.
Previous blocks provide context as committed wte embeddings.

Usage:
    CUDA_VISIBLE_DEVICES=4 python dsl_llada/eval/eval_block_sde.py --gpu 0

    # Specific config:
    CUDA_VISIBLE_DEVICES=4 python dsl_llada/eval/eval_block_sde.py --gpu 0 \
        --block_size 32 --steps_per_block 8 --n_problems 50
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _root)
_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "LLaDA"))
from generate import generate as llada_generate

MASK_ID = 126336
TOP_K = 512


# ===================================================================
# Answer extraction (from eval_unified.py)
# ===================================================================

def extract_number(text):
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"(?:answer|result)\s*(?:is|=)\s*\$?([+-]?[\d,]+\.?\d*)", text, re.I)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def normalize_answer(s):
    s = s.strip().lower()
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "")
    s = s.rstrip(".")
    return s


# ===================================================================
# SNR schedule helpers
# ===================================================================

def make_sensitive_schedule(steps, snr_start=0.01, snr_end=100.0, device="cpu"):
    """Sensitive schedule: 5% steps for low SNR, 90% for mid, 5% for high."""
    n1 = max(1, int(steps * 0.05))
    n2 = max(1, int(steps * 0.90))
    n3 = steps - n1 - n2
    snrs = torch.cat([
        torch.exp(torch.linspace(math.log(max(snr_start, 0.01)), math.log(7), n1 + 1)),
        torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
        torch.exp(torch.linspace(math.log(74), math.log(snr_end), n3 + 1))[1:],
    ]).to(device)
    return snrs


# ===================================================================
# Block SDE generation
# ===================================================================

@torch.no_grad()
def block_sde_generate(
    model, ne, lb, bw, bb, K, bi, tokenizer, prompt_text,
    gen_length=256, block_size=32, steps_per_block=8,
    ns=0.01, snr_start=0.01, snr_end=100.0, device="cuda",
):
    """
    Block SDE generation.

    Args:
        model: LLaDA model (bf16, eval mode)
        ne: noise_embed weight (V, d_noise), float32
        lb: converter logit_bias (1, 1, V+1), float32
        bw: backbone_embedding weight (d_backbone, V+1), float32
        bb: backbone_embedding bias (d_backbone,), float32
        K: noise_embed with mask row appended (V+1, d_noise), float32
        bi: beta_infer scalar
        tokenizer: LLaDA tokenizer
        prompt_text: raw prompt string
        gen_length: total tokens to generate
        block_size: tokens per block
        steps_per_block: SDE steps within each block
        ns: noise scale for SDE
        snr_start: starting SNR for each block's SDE
        snr_end: ending SNR for each block's SDE
        device: cuda device

    Returns:
        generated_text: decoded string (gen positions only)
    """
    nd = ne.shape[1]  # noise dim
    n_blocks = gen_length // block_size

    # Encode prompt
    msgs = [{"role": "user", "content": prompt_text}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    pl = iids.shape[1]

    # Prompt embedding (wte, for backbone input)
    prompt_wte = model.model.transformer.wte(iids).float()  # (1, pl, d_backbone)

    # Mask embedding (wte of MASK_ID)
    mask_wte = model.model.transformer.wte(
        torch.tensor([[MASK_ID]], device=device)
    ).float()  # (1, 1, d_backbone)

    # Track committed tokens for each block
    committed_tokens = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)

    # SNR schedule for each block (same schedule reused)
    snrs = make_sensitive_schedule(steps_per_block, snr_start, snr_end, device)

    for block_idx in range(n_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size

        # Initialize SDE y-values for current block (random unit-norm)
        y = F.normalize(torch.randn(1, block_size, nd, device=device), dim=-1)

        def xhat(yy, ss):
            """Compute denoised estimate for current block positions.

            Build full sequence embedding:
            [prompt_wte | committed_blocks_wte | SDE_converter_output | mask_wte_for_future]
            """
            # Current block: SDE → converter → backbone embedding
            z = ss * yy  # (1, block_size, nd)
            cl = bi * (z.float() @ K.T) + lb  # (1, block_size, V+1)
            p_ = F.softmax(cl.float(), dim=-1)  # fp32 softmax critical
            h_block = F.linear(p_, bw, bb).to(torch.bfloat16)  # (1, block_size, d_backbone)

            # Build full embedding sequence
            parts = [prompt_wte.to(torch.bfloat16)]  # prompt

            # Committed previous blocks (use wte)
            if block_start > 0:
                committed_wte = model.model.transformer.wte(
                    committed_tokens[:, :block_start]
                )  # (1, block_start, d_backbone)
                parts.append(committed_wte.to(torch.bfloat16))

            # Current block (SDE converter output)
            parts.append(h_block)

            # Future blocks (MASK embedding)
            remaining = gen_length - block_end
            if remaining > 0:
                future_mask = mask_wte.to(torch.bfloat16).expand(1, remaining, -1)
                parts.append(future_mask)

            full_embed = torch.cat(parts, dim=1)  # (1, pl + gen_length, d_backbone)

            # Dummy input_ids for model forward
            dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)

            # Model forward
            lo = model(input_ids=dummy, inputs_embeds=full_embed).logits
            # Extract logits for current block positions only
            block_logits = lo[:, pl + block_start: pl + block_end, :].float()

            # Top-K weighted noise embedding (denoised estimate)
            bp = F.softmax(block_logits, dim=-1)
            tv, ti = bp.topk(min(TOP_K, bp.shape[-1]), dim=-1)
            tv = tv / tv.sum(dim=-1, keepdim=True)
            return (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)

        # Heun SDE integration
        for i in range(len(snrs) - 1):
            s, sn = snrs[i], snrs[i + 1]
            ds = sn - s
            dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

            xh = xhat(y, s)
            f_ = (xh - y) / s
            g = ns / s
            ye = y + f_ * ds + g * dW

            xhe = xhat(ye, sn)
            fe = (xhe - ye) / sn
            ge = ns / sn
            y = y + 0.5 * (f_ + fe) * ds + 0.5 * (g + ge) * dW

        # Final prediction for this block
        zf = snrs[-1] * y
        cf = bi * (zf.float() @ K.T) + lb
        pf = F.softmax(cf.float(), dim=-1)
        hf = F.linear(pf, bw, bb).to(torch.bfloat16)

        # Build full embedding for final prediction
        parts = [prompt_wte.to(torch.bfloat16)]
        if block_start > 0:
            committed_wte = model.model.transformer.wte(committed_tokens[:, :block_start])
            parts.append(committed_wte.to(torch.bfloat16))
        parts.append(hf)
        remaining = gen_length - block_end
        if remaining > 0:
            future_mask = mask_wte.to(torch.bfloat16).expand(1, remaining, -1)
            parts.append(future_mask)
        full_embed = torch.cat(parts, dim=1)
        dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)
        lo = model(input_ids=dummy, inputs_embeds=full_embed).logits
        block_logits = lo[:, pl + block_start: pl + block_end, :].float()
        block_tokens = block_logits.argmax(dim=-1)  # (1, block_size)

        # Commit this block
        committed_tokens[:, block_start:block_end] = block_tokens

    return tokenizer.decode(committed_tokens[0], skip_special_tokens=True)


# ===================================================================
# Pure SDE generation (baseline for comparison)
# ===================================================================

@torch.no_grad()
def pure_sde_generate(
    model, ne, lb, bw, bb, K, bi, tokenizer, prompt_text,
    gen_length=256, steps=32, ns=0.01,
    snr_start=0.01, snr_end=100.0, device="cuda",
):
    """Pure SDE (all positions simultaneously), from sde_eval_single_ckpt.py."""
    nd = ne.shape[1]
    msgs = [{"role": "user", "content": prompt_text}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)

    snrs = make_sensitive_schedule(steps, snr_start, snr_end, device)

    torch.manual_seed(42)
    y = F.normalize(torch.randn(1, gen_length, nd, device=device), dim=-1)

    def xhat(yy, ss):
        z = ss * yy
        cl = bi * (z.float() @ K.T) + lb
        p_ = F.softmax(cl.float(), dim=-1)
        h = F.linear(p_, bw, bb).to(torch.bfloat16)
        e = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        lo = model(input_ids=dummy, inputs_embeds=e).logits[:, pl:, :].float()
        bp = F.softmax(lo, dim=-1)
        tv, ti = bp.topk(min(TOP_K, bp.shape[-1]), dim=-1)
        tv = tv / tv.sum(dim=-1, keepdim=True)
        return (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)

    for i in range(len(snrs) - 1):
        s, sn = snrs[i], snrs[i + 1]
        ds = sn - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = xhat(y, s)
        f_ = (xh - y) / s
        g = ns / s
        ye = y + f_ * ds + g * dW
        xhe = xhat(ye, sn)
        fe = (xhe - ye) / sn
        ge = ns / sn
        y = y + 0.5 * (f_ + fe) * ds + 0.5 * (g + ge) * dW

    zf = snrs[-1] * y
    cf = bi * (zf.float() @ K.T) + lb
    pf = F.softmax(cf.float(), dim=-1)
    hf = F.linear(pf, bw, bb).to(torch.bfloat16)
    ef = torch.cat([pe.to(torch.bfloat16), hf], dim=1)
    lo = model(input_ids=dummy, inputs_embeds=ef).logits[:, pl:, :].float()
    return tokenizer.decode(lo.argmax(dim=-1)[0], skip_special_tokens=True)


# ===================================================================
# Standard remask generation (baseline)
# ===================================================================

@torch.no_grad()
def remask_generate(
    model, tokenizer, prompt_text, gen_length=256, steps=64,
    block_length=256, device="cuda",
):
    """Standard LLaDA remasking generation."""
    msgs = [{"role": "user", "content": prompt_text}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    attn = torch.ones_like(iids)
    out = llada_generate(
        model, iids, attn,
        steps=steps, gen_length=gen_length,
        block_length=block_length, temperature=0.0,
        cfg_scale=0.0, remasking="low_confidence",
    )
    return tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)


# ===================================================================
# Main evaluation
# ===================================================================

def load_dsl_weights(ckpt_dir, device):
    """Load DSL weights (noise_embed, converter) from safetensors."""
    from safetensors.torch import load_file
    ne, lb, bw, bb = None, None, None, None
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith(".safetensors"):
            st = load_file(os.path.join(ckpt_dir, f), device=device)
            if "noise_embed.weight" in st:
                ne = st["noise_embed.weight"].float()
                lb = st["converter.logit_bias"].float()
                bw = st["converter.backbone_embedding.weight"].float()
                bb = st["converter.backbone_embedding.bias"].float()
    if ne is None:
        raise RuntimeError(f"No DSL weights found in {ckpt_dir}")
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
    return ne, lb, bw, bb, K


def run_gsm8k_eval(generate_fn, data, desc="eval"):
    """Run GSM8K eval with a generate function that takes prompt_text → generated_text."""
    correct = 0
    cases = []
    for item in tqdm(data, desc=desc):
        prompt_text = (
            f"Solve the following math problem step by step. "
            f"Show your work and put your final answer after ####.\n\n"
            f"Question: {item['question']}\n\nAnswer:"
        )
        try:
            gen_text = generate_fn(prompt_text)
        except Exception as e:
            print(f"  Error on item {item['id']}: {e}")
            cases.append({"id": item["id"], "gold": item["gold_answer"],
                          "pred": "", "correct": False, "text": f"ERROR: {e}"})
            continue

        pred = extract_number(gen_text)
        gold = str(item["gold_answer"])
        is_correct = normalize_answer(pred) == normalize_answer(gold)
        correct += int(is_correct)
        cases.append({
            "id": item["id"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "text": gen_text[:500],
        })

    acc = correct / len(data) if data else 0.0
    return acc, correct, cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/pertoken_b1_d100_1k/checkpoint-1000")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bi", type=float, default=2.0)
    p.add_argument("--ns", type=float, default=0.01)
    p.add_argument("--snr_start", type=float, default=0.01)
    p.add_argument("--snr_end", type=float, default=100.0)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--n_problems", type=int, default=50)
    # Block SDE configs to test
    p.add_argument("--block_sizes", type=str, default="16,32,64,128",
                   help="Comma-separated block sizes to test")
    p.add_argument("--steps_per_block", type=str, default="4,8,16",
                   help="Comma-separated steps per block to test")
    # Which baselines to run
    p.add_argument("--skip_baselines", action="store_true",
                   help="Skip baseline evaluations (pure SDE, remask)")
    p.add_argument("--only", type=str, default=None,
                   help="Only run specific config, e.g. 'block_32_8' or 'pure_sde_32' or 'remask_64'")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"

    # Load model
    print("Loading model...")
    from transformers import AutoModel, AutoTokenizer
    ckpt_path = os.path.join(_root, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

    # Load DSL weights
    print("Loading DSL weights...")
    ne, lb, bw, bb, K = load_dsl_weights(ckpt_path, device)
    print(f"  noise_embed: {ne.shape}, logit_bias: {lb.shape}")

    # Load GSM8K data
    data_path = os.path.join(_root, "eval_data", "gsm8k_100.json")
    with open(data_path) as f:
        all_data = json.load(f)
    data = all_data[:args.n_problems]
    print(f"Evaluating on {len(data)} GSM8K problems")

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    steps_list = [int(x) for x in args.steps_per_block.split(",")]

    results = {}

    # ---------------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------------
    if not args.skip_baselines and args.only is None:
        # Pure SDE 32 steps (NFE=64 with Heun)
        print("\n" + "=" * 60)
        print("Baseline: Pure SDE 32 steps (NFE=64)")
        print("=" * 60)
        t0 = time.time()
        acc, correct, cases = run_gsm8k_eval(
            lambda pt: pure_sde_generate(
                model, ne, lb, bw, bb, K, args.bi, tokenizer, pt,
                gen_length=args.gen_length, steps=32, ns=args.ns,
                snr_start=args.snr_start, snr_end=args.snr_end, device=device,
            ),
            data, desc="Pure SDE 32"
        )
        elapsed = time.time() - t0
        results["pure_sde_32"] = {
            "accuracy": round(acc, 4), "correct": correct, "n": len(data),
            "nfe": 64, "time_s": round(elapsed, 1), "cases": cases,
        }
        print(f"  Pure SDE 32: {correct}/{len(data)} = {acc:.3f} ({elapsed:.0f}s)")

        # Standard remask 64 steps, block=256 (full)
        print("\n" + "=" * 60)
        print("Baseline: Remask 64 steps, block=256")
        print("=" * 60)
        t0 = time.time()
        acc, correct, cases = run_gsm8k_eval(
            lambda pt: remask_generate(
                model, tokenizer, pt,
                gen_length=args.gen_length, steps=64,
                block_length=256, device=device,
            ),
            data, desc="Remask 64 b256"
        )
        elapsed = time.time() - t0
        results["remask_64_b256"] = {
            "accuracy": round(acc, 4), "correct": correct, "n": len(data),
            "nfe": 64, "time_s": round(elapsed, 1), "cases": cases,
        }
        print(f"  Remask 64 b256: {correct}/{len(data)} = {acc:.3f} ({elapsed:.0f}s)")

        # Standard remask 64 steps, block=32
        print("\n" + "=" * 60)
        print("Baseline: Remask 64 steps, block=32")
        print("=" * 60)
        t0 = time.time()
        acc, correct, cases = run_gsm8k_eval(
            lambda pt: remask_generate(
                model, tokenizer, pt,
                gen_length=args.gen_length, steps=64,
                block_length=32, device=device,
            ),
            data, desc="Remask 64 b32"
        )
        elapsed = time.time() - t0
        results["remask_64_b32"] = {
            "accuracy": round(acc, 4), "correct": correct, "n": len(data),
            "nfe": 64 * (256 // 32),  # actual NFE for block remask (steps_per_block * n_blocks but steps is total)
            "time_s": round(elapsed, 1), "cases": cases,
        }
        print(f"  Remask 64 b32: {correct}/{len(data)} = {acc:.3f} ({elapsed:.0f}s)")

    elif args.only and args.only.startswith("remask"):
        # Run specific remask baseline
        parts = args.only.split("_")
        steps = int(parts[1])
        bl = int(parts[2].replace("b", "")) if len(parts) > 2 else 256
        print(f"\nRunning: Remask {steps} steps, block={bl}")
        t0 = time.time()
        acc, correct, cases = run_gsm8k_eval(
            lambda pt: remask_generate(
                model, tokenizer, pt,
                gen_length=args.gen_length, steps=steps,
                block_length=bl, device=device,
            ),
            data, desc=f"Remask {steps} b{bl}"
        )
        elapsed = time.time() - t0
        results[args.only] = {
            "accuracy": round(acc, 4), "correct": correct, "n": len(data),
            "nfe": steps, "time_s": round(elapsed, 1), "cases": cases,
        }
        print(f"  {args.only}: {correct}/{len(data)} = {acc:.3f} ({elapsed:.0f}s)")

    elif args.only and args.only.startswith("pure_sde"):
        parts = args.only.split("_")
        steps = int(parts[2])
        print(f"\nRunning: Pure SDE {steps} steps")
        t0 = time.time()
        acc, correct, cases = run_gsm8k_eval(
            lambda pt: pure_sde_generate(
                model, ne, lb, bw, bb, K, args.bi, tokenizer, pt,
                gen_length=args.gen_length, steps=steps, ns=args.ns,
                snr_start=args.snr_start, snr_end=args.snr_end, device=device,
            ),
            data, desc=f"Pure SDE {steps}"
        )
        elapsed = time.time() - t0
        results[args.only] = {
            "accuracy": round(acc, 4), "correct": correct, "n": len(data),
            "nfe": steps * 2, "time_s": round(elapsed, 1), "cases": cases,
        }
        print(f"  {args.only}: {correct}/{len(data)} = {acc:.3f} ({elapsed:.0f}s)")

    # ---------------------------------------------------------------
    # Block SDE configs
    # ---------------------------------------------------------------
    for bs in block_sizes:
        if args.gen_length % bs != 0:
            print(f"  Skipping block_size={bs} (gen_length={args.gen_length} not divisible)")
            continue
        n_blocks = args.gen_length // bs
        for spb in steps_list:
            config_name = f"block_{bs}_{spb}"

            if args.only and args.only != config_name:
                continue

            nfe = n_blocks * spb * 2  # Heun = 2 NFE per step
            # +1 final prediction per block
            nfe_total = n_blocks * (spb * 2 + 1)

            print(f"\n{'=' * 60}")
            print(f"Block SDE: B={bs}, steps/block={spb}, n_blocks={n_blocks}, NFE~{nfe_total}")
            print(f"{'=' * 60}")

            t0 = time.time()
            acc, correct, cases = run_gsm8k_eval(
                lambda pt, _bs=bs, _spb=spb: block_sde_generate(
                    model, ne, lb, bw, bb, K, args.bi, tokenizer, pt,
                    gen_length=args.gen_length, block_size=_bs,
                    steps_per_block=_spb, ns=args.ns,
                    snr_start=args.snr_start, snr_end=args.snr_end,
                    device=device,
                ),
                data, desc=f"Block B={bs} S={spb}"
            )
            elapsed = time.time() - t0
            results[config_name] = {
                "accuracy": round(acc, 4), "correct": correct, "n": len(data),
                "block_size": bs, "steps_per_block": spb,
                "n_blocks": n_blocks, "nfe": nfe_total,
                "time_s": round(elapsed, 1), "cases": cases,
            }
            print(f"  Block B={bs} S={spb}: {correct}/{len(data)} = {acc:.3f} "
                  f"(NFE={nfe_total}, {elapsed:.0f}s)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'Acc':>8} {'NFE':>6} {'Time':>8}")
    print("-" * 50)
    for name, r in sorted(results.items(), key=lambda x: x[1].get("nfe", 0)):
        acc_str = f"{r['correct']}/{r['n']} ({r['accuracy']:.1%})"
        print(f"{name:<25} {acc_str:>8} {r.get('nfe', '?'):>6} {r['time_s']:>7.0f}s")

    # Save results (strip cases for summary, keep in full)
    out_path = os.path.join(_root, "eval_results", "block_sde.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.ckpt,
        "params": {"bi": args.bi, "ns": args.ns, "snr_start": args.snr_start,
                    "snr_end": args.snr_end, "gen_length": args.gen_length,
                    "n_problems": args.n_problems},
        "results": {k: {kk: vv for kk, vv in v.items()}
                    for k, v in results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
