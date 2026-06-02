"""Absorbing SDE for GSM8K reasoning evaluation.

At each SDE step, positions whose backbone confidence exceeds a threshold
are "absorbed" (locked) and stop being updated. This simulates remasking's
"commit" behavior within the continuous SDE framework.

Usage:
    CUDA_VISIBLE_DEVICES=2 python dsl_llada/absorbing_sde.py --gpu 0
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
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

TOP_K = 512
CKPT = "checkpoints/pertoken_b1_d100_1k/checkpoint-1000"
GSM8K_PATH = "eval_data/gsm8k_100.json"
MASK_ID = 126336


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
    s = s.replace(" ", "").rstrip(".")
    return s


def load_model_and_dsl(ckpt_path, device):
    """Load LLaDA model and DSL weights from safetensors."""
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    ne, lb, bw, bb, conv_embed, beta_val = None, None, None, None, None, None
    for fname in sorted(os.listdir(ckpt_path)):
        if fname.endswith(".safetensors"):
            st = load_file(os.path.join(ckpt_path, fname), device=device)
            if "noise_embed.weight" in st:
                ne = st["noise_embed.weight"].float()
                lb = st["converter.logit_bias"].float()
                bw = st["converter.backbone_embedding.weight"].float()
                bb = st["converter.backbone_embedding.bias"].float()
                conv_embed = st["converter.embed.weight"].float()
                beta_val = st["converter.beta"].float()

    assert ne is not None, "DSL weights not found in checkpoint"
    # K for converter: use converter.embed (what the converter was trained with)
    K = torch.cat([conv_embed, torch.zeros(1, conv_embed.shape[1], device=device)], dim=0)
    print(f"  noise_dim={ne.shape[1]}, vocab={ne.shape[0]}, beta={beta_val.item():.3f}")
    return model, tokenizer, ne, lb, bw, bb, K, beta_val


def sde_generate_absorbing(
    model, ne, lb, bw, bb, K, beta_infer, tokenizer,
    prompt, steps, schedule, ns, device,
    gen_length=256, threshold=None, seed=42,
):
    """SDE generation with optional absorbing (locking high-confidence positions).

    Returns: (text, lock_stats)
        lock_stats: list of (step, num_locked, num_total) per step, or None if no absorbing
    """
    msgs = [{"role": "user", "content": prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = K.shape[1]  # noise dim
    dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)

    # SNR schedule
    snr_min = 0.01
    if schedule == "sensitive":
        n1, n2 = max(1, int(steps * 0.05)), max(1, int(steps * 0.90))
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

    bv = beta_infer

    # Absorbing state
    use_absorbing = threshold is not None
    locked = torch.zeros(1, gen_length, dtype=torch.bool, device=device) if use_absorbing else None
    lock_stats = [] if use_absorbing else None

    def get_xhat_with_probs(yy, ss):
        """Returns (x_hat embedding, backbone_probs for confidence)."""
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        bp = F.softmax(logits, dim=-1)

        # x_hat via top-K weighted sum of noise embeddings
        tv, ti = bp.topk(min(TOP_K, bp.shape[-1]), dim=-1)
        tv = tv / tv.sum(dim=-1, keepdim=True)
        xhat = (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)

        return xhat, bp

    for i in range(len(snrs) - 1):
        s, s_next = snrs[i], snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        xh, backbone_probs = get_xhat_with_probs(y, s)

        # Check confidence and lock positions
        if use_absorbing:
            conf = backbone_probs.max(dim=-1).values[0]  # (gen_length,)
            newly_locked = (conf > threshold) & (~locked[0])
            locked[0] |= newly_locked
            n_locked = locked[0].sum().item()
            lock_stats.append((i, n_locked, gen_length))

        # SDE Heun update
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW

        xh_e, _ = get_xhat_with_probs(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y_new = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

        # Apply absorbing: only update unlocked positions
        if use_absorbing and locked[0].any():
            y = torch.where(locked.unsqueeze(-1), y, y_new)
        else:
            y = y_new

    # Final decode
    z_f = snrs[-1] * y
    cl_f = bv * (z_f.float() @ K.T) + lb
    pf = F.softmax(cl_f.float(), dim=-1)
    hf = F.linear(pf, bw, bb).to(torch.bfloat16)
    ef = torch.cat([pe.to(torch.bfloat16), hf], dim=1)
    with torch.no_grad():
        lo = model(input_ids=dummy, inputs_embeds=ef).logits[:, pl:, :].float()
    text = tokenizer.decode(lo.argmax(dim=-1)[0], skip_special_tokens=True)
    return text, lock_stats


def run_gsm8k_eval(model, ne, lb, bw, bb, K, beta_infer, tokenizer, device,
                   data, steps, schedule, ns, threshold, n_problems=50, seed=42):
    """Run GSM8K evaluation with given config. Returns results dict."""
    correct = 0
    total = 0
    all_lock_stats = []
    cases = []

    label = f"steps={steps} thr={threshold}" if threshold else f"steps={steps} baseline"
    print(f"\n  Config: {label}")

    for idx, item in enumerate(data[:n_problems]):
        prompt_text = (
            f"Solve the following math problem step by step. "
            f"Show your work and put your final answer after ####.\n\n"
            f"Question: {item['question']}\n\nAnswer:"
        )

        try:
            text, lock_stats = sde_generate_absorbing(
                model, ne, lb, bw, bb, K, beta_infer, tokenizer,
                prompt_text, steps=steps, schedule=schedule, ns=ns,
                device=device, gen_length=256, threshold=threshold, seed=seed,
            )

            pred = extract_number(text)
            gold = str(item["gold_answer"])
            is_correct = normalize_answer(pred) == normalize_answer(gold)
            correct += int(is_correct)
            total += 1

            case = {
                "id": item["id"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "text": text[:500],
            }

            if lock_stats:
                # Summarize lock stats
                final_locked = lock_stats[-1][1] if lock_stats else 0
                first_lock_step = None
                for s_idx, n_l, n_t in lock_stats:
                    if n_l > 0 and first_lock_step is None:
                        first_lock_step = s_idx
                case["final_locked"] = final_locked
                case["first_lock_step"] = first_lock_step
                all_lock_stats.append(lock_stats)

            cases.append(case)

            if (idx + 1) % 5 == 0:
                print(f"    [{idx+1}/{n_problems}] acc={correct}/{total} ({100*correct/total:.1f}%)")

        except Exception as e:
            print(f"    Error on item {item.get('id', '?')}: {e}")
            total += 1
            cases.append({"id": item.get("id", -1), "gold": str(item.get("gold_answer", "")),
                          "pred": "", "correct": False, "text": f"ERROR: {e}"})

    acc = correct / total if total > 0 else 0.0

    # Aggregate lock statistics
    lock_summary = None
    if all_lock_stats:
        # Average locked positions per step across all problems
        max_steps = max(len(ls) for ls in all_lock_stats)
        avg_locked_per_step = []
        for step_i in range(max_steps):
            vals = [ls[step_i][1] for ls in all_lock_stats if step_i < len(ls)]
            avg_locked_per_step.append(float(np.mean(vals)))

        first_locks = [c.get("first_lock_step") for c in cases if c.get("first_lock_step") is not None]
        final_locks = [c.get("final_locked", 0) for c in cases if "final_locked" in c]

        lock_summary = {
            "avg_locked_per_step": [round(v, 1) for v in avg_locked_per_step],
            "avg_first_lock_step": round(float(np.mean(first_locks)), 1) if first_locks else None,
            "avg_final_locked": round(float(np.mean(final_locks)), 1) if final_locks else None,
            "pct_final_locked": round(100 * float(np.mean(final_locks)) / 256, 1) if final_locks else None,
        }

    result = {
        "steps": steps,
        "nfe": steps * 2,  # Heun
        "threshold": threshold,
        "schedule": schedule,
        "noise_scale": ns,
        "beta_infer": float(beta_infer),
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "lock_summary": lock_summary,
        "cases": cases,
    }

    print(f"  => Accuracy: {correct}/{total} = {100*acc:.1f}%")
    if lock_summary:
        print(f"     Avg first lock step: {lock_summary['avg_first_lock_step']}")
        print(f"     Avg final locked: {lock_summary['avg_final_locked']}/256 ({lock_summary['pct_final_locked']}%)")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--beta_infer", type=float, default=2.0)
    parser.add_argument("--noise_scale", type=float, default=0.01)
    parser.add_argument("--schedule", type=str, default="sensitive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_64", action="store_true", help="Also test 64-step configs")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CKPT)
    gsm8k_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), GSM8K_PATH)

    print(f"Loading model from {ckpt_path}")
    model, tokenizer, ne, lb, bw, bb, K, beta_val = load_model_and_dsl(ckpt_path, device)
    print(f"  Trained beta={beta_val.item():.3f}, using beta_infer={args.beta_infer}")

    data = json.load(open(gsm8k_path))
    print(f"Loaded {len(data)} GSM8K problems, evaluating first {args.n_problems}")

    bi = args.beta_infer
    ns = args.noise_scale
    sch = args.schedule

    # Test configurations
    configs = [
        # (steps, threshold)
        (32, None),        # baseline SDE 32 steps
        (32, 0.5),         # absorbing threshold=0.5
        (32, 0.7),         # absorbing threshold=0.7
        (32, 0.9),         # absorbing threshold=0.9
        (32, 0.95),        # absorbing threshold=0.95
    ]
    if args.include_64:
        configs += [
            (64, None),        # baseline SDE 64 steps
            (64, 0.5),         # absorbing threshold=0.5, 64 steps
            (64, 0.7),         # absorbing threshold=0.7, 64 steps
            (64, 0.9),         # absorbing threshold=0.9, 64 steps
            (64, 0.95),        # absorbing threshold=0.95, 64 steps
        ]

    all_results = []
    for steps, thr in configs:
        t0 = time.time()
        result = run_gsm8k_eval(
            model, ne, lb, bw, bb, K, bi, tokenizer, device,
            data, steps=steps, schedule=sch, ns=ns, threshold=thr,
            n_problems=args.n_problems, seed=args.seed,
        )
        result["time_s"] = round(time.time() - t0, 1)
        all_results.append(result)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Absorbing SDE on GSM8K")
    print("=" * 70)
    print(f"{'Steps':>5} {'NFE':>5} {'Threshold':>10} {'Acc':>8} {'Final%Locked':>14} {'FirstLock':>10} {'Time':>8}")
    print("-" * 70)
    for r in all_results:
        thr_str = f"{r['threshold']}" if r['threshold'] else "none"
        ls = r.get("lock_summary") or {}
        fl_str = f"{ls.get('pct_final_locked', '-')}%" if ls.get('pct_final_locked') is not None else "-"
        fls_str = f"{ls.get('avg_first_lock_step', '-')}" if ls.get('avg_first_lock_step') is not None else "-"
        print(f"{r['steps']:>5} {r['nfe']:>5} {thr_str:>10} {100*r['accuracy']:>7.1f}% {fl_str:>14} {fls_str:>10} {r['time_s']:>7.1f}s")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eval_results", "absorbing_sde.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        "config": {
            "model": CKPT,
            "beta_infer": args.beta_infer,
            "noise_scale": args.noise_scale,
            "schedule": args.schedule,
            "n_problems": args.n_problems,
            "seed": args.seed,
        },
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
