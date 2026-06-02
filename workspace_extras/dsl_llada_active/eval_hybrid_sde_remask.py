"""Hybrid SDE → Remask generation for GSM8K evaluation.

Phase 1 (SDE): Run Heun SDE for K steps to establish global structure
Phase 2 (Remask): Convert SDE output to discrete tokens, mask low-confidence
    positions, then run standard remasking for M steps to refine reasoning.

Usage:
    python dsl_llada/eval_hybrid_sde_remask.py --gpu 0
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_root, "LLaDA"))
from generate import generate

MASK_ID = 126336
GEN_LENGTH = 512
SDE_TOP_K = 512
SEED = 42


def load_model_and_dsl(checkpoint_dir, device):
    """Load LLaDA model and extract DSL weights from safetensors."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Extract DSL weights
    ne, lb, bw, bb, bv = None, None, None, None, None
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(checkpoint_dir, fname), device=str(device))
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].float().item()
                break

    assert ne is not None, f"No DSL weights found in {checkpoint_dir}"
    # K = [ne; zeros(1, d)] for mask slot
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)

    print(f"  DSL weights: ne={ne.shape}, beta={bv:.3f}, noise_dim={ne.shape[1]}")
    return model, tokenizer, ne, lb, bw, bb, bv, K


def build_snr_schedule(steps, snr_max, device, sensitive=True):
    """Build SNR schedule (sensitive = 90% in SNR 7-74)."""
    snr_min = 0.01
    if sensitive:
        n1 = max(1, int(steps * 0.05))
        n2 = max(1, int(steps * 0.90))
        n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(torch.linspace(
            math.log(snr_min), math.log(snr_max), steps + 1
        )).to(device)
    return snrs


def sde_phase(model, tokenizer, prompt_text, ne, lb, bw, bb, bv, K,
              sde_steps, noise_scale, snr_max, device, seed=42):
    """Run SDE Heun integration for sde_steps steps.

    Returns:
        backbone_probs: (1, GEN, V) probability distribution from backbone
        top1_tokens: (GEN,) argmax token ids
        confidence: (GEN,) per-position confidence (max prob)
    """
    msgs = [{'role': 'user', 'content': prompt_text}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    GEN = GEN_LENGTH
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)

    snrs = build_snr_schedule(sde_steps, snr_max, device, sensitive=True)

    torch.manual_seed(seed)
    # Norm init
    y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)

    ns = noise_scale

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        backbone_probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = backbone_probs.topk(min(SDE_TOP_K, backbone_probs.shape[-1]), dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        xh = (top_vals.unsqueeze(-1) * ne[top_idx.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)
        return xh, backbone_probs

    actual_steps = len(snrs) - 1
    for i in range(actual_steps):
        s = snrs[i]
        s_next = snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        xh, _, = get_xhat(y, s)
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e, _ = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    # Final decode
    z_final = snrs[-1] * y
    cl_final = bv * (z_final.float() @ K.T) + lb
    final_probs = F.softmax(cl_final.float(), dim=-1)
    h_final = F.linear(final_probs, bw, bb).to(torch.bfloat16)
    embeds_final = torch.cat([pe.to(torch.bfloat16), h_final], dim=1)
    with torch.no_grad():
        logits_final = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
    backbone_probs = F.softmax(logits_final, dim=-1)
    top1 = logits_final.argmax(dim=-1)[0]  # (GEN,)
    conf = backbone_probs[0].max(dim=-1).values  # (GEN,)

    return top1, conf, iids


def remask_phase(model, prompt_ids, sde_tokens, sde_confidence, threshold,
                 remask_steps, device, temperature=0.0):
    """Run standard remasking on partially-masked sequence.

    Positions with SDE confidence > threshold: keep as committed tokens.
    Positions with SDE confidence <= threshold: replace with MASK.
    """
    pl = prompt_ids.shape[1]
    GEN = GEN_LENGTH

    # Build initial sequence: prompt + (committed or MASK)
    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=device)
    x[0, :pl] = prompt_ids[0]

    # Commit high-confidence positions from SDE
    commit_mask = sde_confidence > threshold
    n_committed = commit_mask.sum().item()
    x[0, pl:pl + GEN][commit_mask] = sde_tokens[commit_mask]

    n_masked = GEN - n_committed
    if n_masked == 0:
        # All committed, just return
        return x

    # Run standard remasking with d3im (confidence-based)
    out = generate(
        model, x[:, :pl],  # just the prompt
        steps=remask_steps,
        gen_length=GEN,
        block_length=GEN,
        temperature=temperature,
        remasking='low_confidence',
        mask_id=MASK_ID,
    )

    # Actually we need to use the partially-masked sequence, not start from scratch.
    # The generate() function starts from all-MASK. We need a custom loop.
    # Let's do it manually.
    return _remask_from_partial(model, x, pl, GEN, remask_steps, temperature, device)


def _remask_from_partial(model, x, prompt_len, gen_len, steps, temperature, device):
    """Run remasking starting from partially-masked sequence x."""
    mask_index = (x == MASK_ID)
    # Only gen positions can be masked
    initial_mask_count = mask_index[:, prompt_len:].sum().item()

    if initial_mask_count == 0:
        return x

    # Compute tokens to unmask per step
    base = initial_mask_count // steps
    remainder = initial_mask_count % steps
    num_transfer = [base] * steps
    for i in range(remainder):
        num_transfer[i] += 1

    for s in range(steps):
        mask_index = (x == MASK_ID)
        n_remaining = mask_index[:, prompt_len:].sum().item()
        if n_remaining == 0:
            break

        with torch.no_grad():
            logits = model(x).logits

        if temperature > 0:
            logits_noised = logits / temperature
            x0 = torch.argmax(logits_noised, dim=-1)
        else:
            x0 = torch.argmax(logits, dim=-1)

        # Confidence = softmax prob of top-1
        p = F.softmax(logits.float(), dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

        # Only consider masked positions
        x0_p[~mask_index] = -float('inf')

        # Apply predictions only to masked positions
        x0 = torch.where(mask_index, x0, x)

        # Select top-confidence positions to unmask
        n_unmask = min(num_transfer[s], n_remaining)
        if n_unmask > 0:
            confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
            _, idx = confidence.view(-1).topk(n_unmask)
            transfer_mask = torch.zeros_like(x.view(-1), dtype=torch.bool)
            transfer_mask[idx] = True
            transfer_mask = transfer_mask.view_as(x)
            x[transfer_mask] = x0[transfer_mask]

    return x


def pure_sde_generate(model, tokenizer, prompt_text, ne, lb, bw, bb, bv, K,
                      sde_steps, noise_scale, snr_max, device, seed=42):
    """Pure SDE generation (no remask phase)."""
    top1, conf, iids = sde_phase(
        model, tokenizer, prompt_text, ne, lb, bw, bb, bv, K,
        sde_steps, noise_scale, snr_max, device, seed
    )
    return top1


def pure_remask_generate(model, tokenizer, prompt_text, remask_steps, device, seed=42):
    """Pure remasking generation."""
    msgs = [{'role': 'user', 'content': prompt_text}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)

    torch.manual_seed(seed)
    out = generate(
        model, iids,
        steps=remask_steps,
        gen_length=GEN_LENGTH,
        block_length=GEN_LENGTH,
        temperature=0.,
        remasking='low_confidence',
        mask_id=MASK_ID,
    )
    return out[0, iids.shape[1]:]


def hybrid_generate(model, tokenizer, prompt_text, ne, lb, bw, bb, bv, K,
                    sde_steps, remask_steps, threshold, noise_scale, snr_max,
                    device, seed=42):
    """Hybrid SDE→Remask generation."""
    # Phase 1: SDE
    top1, conf, iids = sde_phase(
        model, tokenizer, prompt_text, ne, lb, bw, bb, bv, K,
        sde_steps, noise_scale, snr_max, device, seed
    )

    n_committed = (conf > threshold).sum().item()

    # Phase 2: Remask from partial
    pl = iids.shape[1]
    GEN = GEN_LENGTH

    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=device)
    x[0, :pl] = iids[0]

    commit_mask = conf > threshold
    x[0, pl:pl + GEN][commit_mask] = top1[commit_mask]

    torch.manual_seed(seed + 1)  # different seed for remask phase
    x = _remask_from_partial(model, x, pl, GEN, remask_steps, 0.0, device)

    return x[0, pl:pl + GEN], n_committed


def extract_gsm8k_answer(text):
    """Extract numeric answer after ####."""
    # Look for #### followed by a number
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        ans = match.group(1).replace(',', '').strip()
        try:
            return str(int(float(ans)))
        except ValueError:
            return ans

    # Fallback: look for last number after "answer is" or similar
    match = re.search(r'(?:answer is|= |equals)\s*\$?(-?[\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        ans = match.group(1).replace(',', '').strip()
        try:
            return str(int(float(ans)))
        except ValueError:
            return ans

    return None


def format_prompt(question):
    """Format GSM8K question for LLaDA."""
    return (
        "Solve the following math problem step by step. "
        "Show your work and put your final answer after ####.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def run_evaluation(args):
    device = f"cuda:{args.gpu}"
    print(f"Device: {device}")

    # Load data
    data_path = os.path.join(_root, "eval_data", "gsm8k_100.json")
    with open(data_path) as f:
        problems = json.load(f)[:args.n_problems]
    print(f"Loaded {len(problems)} GSM8K problems")

    # Load model
    ckpt = os.path.join(_root, args.checkpoint)
    print(f"Loading model from {ckpt}...")
    model, tokenizer, ne, lb, bw, bb, bv, K = load_model_and_dsl(ckpt, device)
    print("Model loaded.")

    # Define configs
    configs = []

    if args.nfe == 64:
        # Pure SDE: 32 Heun steps = NFE 64
        configs.append({
            'name': 'Pure_SDE_32step',
            'type': 'sde',
            'sde_steps': 32,
            'nfe': 64,
        })
        # Hybrid configs
        for sde_s, remask_s in [(4, 56), (8, 48), (16, 32)]:
            for thr in args.thresholds:
                configs.append({
                    'name': f'Hybrid_{sde_s}+{remask_s}_thr{thr}',
                    'type': 'hybrid',
                    'sde_steps': sde_s,
                    'remask_steps': remask_s,
                    'threshold': thr,
                    'nfe': sde_s * 2 + remask_s,
                })
        # Pure Remask: 64 steps
        configs.append({
            'name': 'Pure_Remask_64step',
            'type': 'remask',
            'remask_steps': 64,
            'nfe': 64,
        })

    if args.nfe == 32:
        # Pure SDE: 16 Heun steps = NFE 32
        configs.append({
            'name': 'Pure_SDE_16step',
            'type': 'sde',
            'sde_steps': 16,
            'nfe': 32,
        })
        for sde_s, remask_s in [(4, 24), (8, 16)]:
            for thr in args.thresholds:
                configs.append({
                    'name': f'Hybrid_{sde_s}+{remask_s}_thr{thr}',
                    'type': 'hybrid',
                    'sde_steps': sde_s,
                    'remask_steps': remask_s,
                    'threshold': thr,
                    'nfe': sde_s * 2 + remask_s,
                })
        configs.append({
            'name': 'Pure_Remask_32step',
            'type': 'remask',
            'remask_steps': 32,
            'nfe': 32,
        })

    if args.nfe == 0:
        # Run both NFE=64 and NFE=32
        # NFE=64
        configs.append({'name': 'Pure_SDE_32step', 'type': 'sde', 'sde_steps': 32, 'nfe': 64})
        for sde_s, remask_s in [(4, 56), (8, 48), (16, 32)]:
            for thr in args.thresholds:
                configs.append({
                    'name': f'Hybrid_{sde_s}+{remask_s}_thr{thr}',
                    'type': 'hybrid', 'sde_steps': sde_s, 'remask_steps': remask_s,
                    'threshold': thr, 'nfe': sde_s * 2 + remask_s,
                })
        configs.append({'name': 'Pure_Remask_64step', 'type': 'remask', 'remask_steps': 64, 'nfe': 64})
        # NFE=32
        configs.append({'name': 'Pure_SDE_16step', 'type': 'sde', 'sde_steps': 16, 'nfe': 32})
        for sde_s, remask_s in [(4, 24), (8, 16)]:
            for thr in args.thresholds:
                configs.append({
                    'name': f'Hybrid_{sde_s}+{remask_s}_thr{thr}',
                    'type': 'hybrid', 'sde_steps': sde_s, 'remask_steps': remask_s,
                    'threshold': thr, 'nfe': sde_s * 2 + remask_s,
                })
        configs.append({'name': 'Pure_Remask_32step', 'type': 'remask', 'remask_steps': 32, 'nfe': 32})

    all_results = {}

    for cfg in configs:
        name = cfg['name']
        print(f"\n{'='*60}")
        print(f"Config: {name} (NFE={cfg['nfe']})")
        print(f"{'='*60}")

        correct = 0
        total = 0
        details = []
        t0 = time.time()

        for prob in tqdm(problems, desc=name):
            prompt_text = format_prompt(prob['question'])
            gold = str(prob['gold_answer'])

            try:
                if cfg['type'] == 'sde':
                    tokens = pure_sde_generate(
                        model, tokenizer, prompt_text, ne, lb, bw, bb, bv, K,
                        cfg['sde_steps'], args.noise_scale, args.snr_max,
                        device, SEED
                    )
                    n_committed = -1
                elif cfg['type'] == 'remask':
                    tokens = pure_remask_generate(
                        model, tokenizer, prompt_text, cfg['remask_steps'],
                        device, SEED
                    )
                    n_committed = -1
                elif cfg['type'] == 'hybrid':
                    tokens, n_committed = hybrid_generate(
                        model, tokenizer, prompt_text, ne, lb, bw, bb, bv, K,
                        cfg['sde_steps'], cfg['remask_steps'], cfg['threshold'],
                        args.noise_scale, args.snr_max, device, SEED
                    )
                else:
                    raise ValueError(f"Unknown type: {cfg['type']}")

                text = tokenizer.decode(tokens, skip_special_tokens=True)
                pred_answer = extract_gsm8k_answer(text)
                is_correct = pred_answer is not None and pred_answer.strip() == gold.strip()
                if is_correct:
                    correct += 1
                total += 1

                details.append({
                    'id': prob['id'],
                    'gold': gold,
                    'pred': pred_answer,
                    'correct': is_correct,
                    'committed': n_committed,
                    'text_preview': text[:200],
                })

            except Exception as e:
                print(f"  Error on problem {prob['id']}: {e}")
                total += 1
                details.append({
                    'id': prob['id'],
                    'gold': gold,
                    'pred': None,
                    'correct': False,
                    'error': str(e),
                })

        elapsed = time.time() - t0
        acc = correct / total * 100 if total > 0 else 0
        print(f"  Result: {correct}/{total} = {acc:.1f}% ({elapsed:.1f}s)")

        # For hybrid, report committed stats
        if cfg['type'] == 'hybrid':
            committed_vals = [d['committed'] for d in details if d['committed'] >= 0]
            if committed_vals:
                avg_committed = sum(committed_vals) / len(committed_vals)
                print(f"  Avg committed from SDE: {avg_committed:.1f}/{GEN_LENGTH} "
                      f"({avg_committed/GEN_LENGTH*100:.1f}%)")

        all_results[name] = {
            'config': cfg,
            'accuracy': acc,
            'correct': correct,
            'total': total,
            'elapsed_s': elapsed,
            'details': details,
        }

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Config':<40} {'NFE':>5} {'Accuracy':>10} {'Time':>8}")
    print(f"{'='*80}")

    # Group by NFE
    for nfe_group in sorted(set(c['nfe'] for c in configs)):
        for name, res in all_results.items():
            if res['config']['nfe'] == nfe_group:
                acc_str = f"{res['correct']}/{res['total']} ({res['accuracy']:.1f}%)"
                time_str = f"{res['elapsed_s']:.0f}s"
                print(f"{name:<40} {nfe_group:>5} {acc_str:>10} {time_str:>8}")
        print(f"{'-'*80}")

    # Save results
    out_path = os.path.join(_root, "eval_results", "hybrid_sde_remask.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/pertoken_b1_d100_1k/checkpoint-1000')
    parser.add_argument('--n_problems', type=int, default=50)
    parser.add_argument('--nfe', type=int, default=0,
                        help='NFE budget (32 or 64). 0 = run both.')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.5, 0.7, 0.9])
    parser.add_argument('--noise_scale', type=float, default=0.05)
    parser.add_argument('--snr_max', type=float, default=80.0)
    args = parser.parse_args()

    run_evaluation(args)
