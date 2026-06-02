"""Token crystallization analysis for SDE generation.

Records per-step per-position data:
1. Lock step: when does each position's argmax token stop changing?
2. Lock order: function words vs content words vs digits vs punctuation
3. Confidence trajectory: backbone confidence per position over time
4. Degeneracy detection: repetitive patterns

Usage:
    CUDA_VISIBLE_DEVICES=4 python dsl_llada/analyze_crystallization.py --gpu 0
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict

# Monkey-patch json encoder to handle torch.dtype (LLaDA config has dtype fields)
import json
_orig_default = json.JSONEncoder().default
class _DtypeEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, '__module__') and 'torch' in str(type(o)):
            return str(o)
        return _orig_default(o)
json.JSONEncoder.default = _DtypeEncoder.default

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

MASK_ID = 126336
SDE_TOP_K = 512
GEN_LENGTH = 128

PROMPTS = [
    "Question: What is 27 + 58? Show the reasoning briefly, then give the final answer.",
    "A store had 45 apples. It sold 18, then got 27 more. How many apples does it have now?",
    "Plan a 3-day Tokyo trip for a first-time visitor. Day 1, Day 2, Day 3. Keep each day to 3 items.",
    "Write a short story about a robot hearing rain for the first time.",
    "There are three boxes: red, blue, green. Exactly one contains a coin. Red is empty. If blue has the coin, green is empty. Where can the coin be?",
]

# Function words (determiners, prepositions, conjunctions, pronouns, auxiliaries)
FUNCTION_WORDS = set([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'must', 'need',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up',
    'about', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'over',
    'and', 'or', 'but', 'nor', 'so', 'yet', 'if', 'then', 'that', 'which',
    'who', 'whom', 'whose', 'where', 'when', 'how', 'what', 'why',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'this', 'these', 'those', 'not', 'no', 'as',
])


def classify_token(text):
    """Classify a token into: function, content, digit, punctuation, special."""
    stripped = text.strip().lower()
    if not stripped:
        return 'special'
    if stripped in FUNCTION_WORDS:
        return 'function'
    if re.match(r'^[\d]+$', stripped):
        return 'digit'
    if re.match(r'^[^\w\s]+$', stripped):
        return 'punctuation'
    if re.match(r'^[a-zA-Z]', stripped):
        return 'content'
    return 'special'


MODEL_CONFIGS = {
    'sm_b2': {
        'ckpt': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
        'beta_infer': 4.0,
        'noise_scale': 0.01,
        'schedule': 'uniform',
        'steps': 64,
    },
    'sem_b05_3k': {
        'ckpt': 'checkpoints/semantic_b05_d100_10k/checkpoint-3000',
        'beta_infer': 1.0,
        'noise_scale': 0.05,
        'schedule': 'sensitive',
        'steps': 64,
    },
}


def load_model_and_dsl(ckpt_path, device):
    """Load LLaDA model and DSL weights."""
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    ne, lb, bw, bb, bv = None, None, None, None, 2.5
    for fname in sorted(os.listdir(ckpt_path)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt_path, fname), device=str(device))
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].item()
                break

    # K = [ne; zeros] for V+1 slots
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
    return model, ne, lb, bw, bb, bv, K


def build_snr_schedule(steps, snr_max, schedule, device):
    """Build SNR schedule."""
    snr_min = 0.01
    if schedule == 'sensitive':
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


def sde_crystallization_analysis(prompt, model, ne, lb, bw, bb, bv_orig, K,
                                  tokenizer, config, device):
    """Run SDE generation and record crystallization data."""
    steps = config['steps']
    ns = config['noise_scale']
    bv = config['beta_infer']
    schedule = config['schedule']
    snr_max = 100.0  # default

    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    dummy = torch.zeros(1, pl + GEN_LENGTH, dtype=torch.long, device=device)

    snrs = build_snr_schedule(steps, snr_max, schedule, device)

    torch.manual_seed(42)
    y = torch.randn(1, GEN_LENGTH, nd, device=device)

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h_low = F.linear(probs, bw, bb)
        h = h_low.to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        backbone_probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = backbone_probs.topk(min(SDE_TOP_K, backbone_probs.shape[-1]), dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        xh = (top_vals.unsqueeze(-1) * ne[top_idx.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)
        return xh, backbone_probs

    actual_steps = len(snrs) - 1

    # Storage: [steps, GEN_LENGTH]
    token_history = []       # list of [GEN_LENGTH] tensors
    confidence_history = []  # list of [GEN_LENGTH] tensors
    top5_entropy_history = []

    for i in range(actual_steps):
        s = snrs[i]
        s_next = snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        xh, backbone_probs = get_xhat(y, s)
        top1 = backbone_probs[0].argmax(dim=-1)  # [GEN_LENGTH]
        conf = backbone_probs[0].max(dim=-1).values  # [GEN_LENGTH]

        # Top-5 entropy for diversity measure
        top5_vals = backbone_probs[0].topk(5, dim=-1).values
        top5_vals = top5_vals / top5_vals.sum(dim=-1, keepdim=True)
        top5_ent = -(top5_vals * (top5_vals + 1e-10).log()).sum(dim=-1)

        token_history.append(top1.cpu())
        confidence_history.append(conf.cpu())
        top5_entropy_history.append(top5_ent.cpu())

        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e, _ = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

        if (i + 1) % 16 == 0:
            text = tokenizer.decode(top1, skip_special_tokens=True)
            print(f"  Step {i+1}/{actual_steps} | conf={conf.mean().item():.3f} | text[:60]={text[:60]}")

    # Final decode
    z_final = snrs[-1] * y
    cl_final = bv * (z_final.float() @ K.T) + lb
    final_probs = F.softmax(cl_final.float(), dim=-1)
    h_final = F.linear(final_probs, bw, bb).to(torch.bfloat16)
    embeds_final = torch.cat([pe.to(torch.bfloat16), h_final], dim=1)
    with torch.no_grad():
        lo_final = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
    final_toks = lo_final.argmax(dim=-1)[0]
    final_conf = F.softmax(lo_final, dim=-1)[0].max(dim=-1).values

    token_history.append(final_toks.cpu())
    confidence_history.append(final_conf.cpu())

    # Compute lock step per position
    # Lock step = last step where token changed (then it stayed fixed until end)
    n_steps_recorded = len(token_history)
    lock_steps = []
    for pos in range(GEN_LENGTH):
        lock = 0
        for t in range(n_steps_recorded - 1, 0, -1):
            if token_history[t][pos].item() != token_history[t - 1][pos].item():
                lock = t
                break
        lock_steps.append(lock)

    # Decode final tokens and classify
    final_tokens_decoded = []
    final_token_ids = final_toks.cpu().tolist()
    for tid in final_token_ids:
        text = tokenizer.decode([tid], skip_special_tokens=False)
        final_tokens_decoded.append(text)

    token_types = [classify_token(t) for t in final_tokens_decoded]

    # Confidence at lock step
    conf_at_lock = []
    for pos in range(GEN_LENGTH):
        ls = lock_steps[pos]
        conf_at_lock.append(confidence_history[ls][pos].item())

    # Degeneracy: count positions where token == previous token (repetition)
    degen_per_step = []
    for t in range(n_steps_recorded):
        toks = token_history[t]
        n_repeat = 0
        for pos in range(1, GEN_LENGTH):
            if toks[pos].item() == toks[pos - 1].item():
                n_repeat += 1
        degen_per_step.append(n_repeat)

    # Consecutive repetition in final output
    final_repeats = 0
    for pos in range(1, GEN_LENGTH):
        if final_token_ids[pos] == final_token_ids[pos - 1]:
            final_repeats += 1

    # Build per-position data
    per_position = []
    for pos in range(GEN_LENGTH):
        per_position.append({
            'pos': pos,
            'token_id': final_token_ids[pos],
            'token_text': final_tokens_decoded[pos],
            'token_type': token_types[pos],
            'lock_step': lock_steps[pos],
            'conf_at_lock': round(conf_at_lock[pos], 4),
            'final_conf': round(confidence_history[-1][pos].item(), 4),
            'conf_trajectory': [round(confidence_history[t][pos].item(), 4)
                                for t in range(0, n_steps_recorded, max(1, n_steps_recorded // 16))],
        })

    # Summary stats by type
    type_stats = {}
    for ttype in ['function', 'content', 'digit', 'punctuation', 'special']:
        positions = [p for p in per_position if p['token_type'] == ttype]
        if positions:
            locks = [p['lock_step'] for p in positions]
            confs = [p['conf_at_lock'] for p in positions]
            type_stats[ttype] = {
                'count': len(positions),
                'avg_lock_step': round(sum(locks) / len(locks), 1),
                'median_lock_step': sorted(locks)[len(locks) // 2],
                'avg_conf_at_lock': round(sum(confs) / len(confs), 4),
                'min_lock_step': min(locks),
                'max_lock_step': max(locks),
            }

    final_text = tokenizer.decode(final_toks, skip_special_tokens=True)

    return {
        'prompt': prompt,
        'final_text': final_text,
        'per_position': per_position,
        'type_stats': type_stats,
        'degen_per_step': degen_per_step,
        'final_repeats': final_repeats,
        'n_steps': n_steps_recorded,
    }


def print_summary(results, model_name):
    """Print summary for one model."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    # Aggregate across all prompts
    all_type_data = defaultdict(lambda: {'locks': [], 'confs': []})
    total_final_repeats = 0
    total_positions = 0

    for r in results:
        for pp in r['per_position']:
            ttype = pp['token_type']
            all_type_data[ttype]['locks'].append(pp['lock_step'])
            all_type_data[ttype]['confs'].append(pp['conf_at_lock'])
        total_final_repeats += r['final_repeats']
        total_positions += GEN_LENGTH

    print(f"\n--- Lock Step by Token Type (aggregated over {len(results)} prompts) ---")
    print(f"{'Type':<12} {'Count':>6} {'Avg Lock':>10} {'Med Lock':>10} {'Avg Conf':>10}")
    for ttype in ['function', 'content', 'digit', 'punctuation', 'special']:
        data = all_type_data[ttype]
        if data['locks']:
            locks = data['locks']
            confs = data['confs']
            avg_l = sum(locks) / len(locks)
            med_l = sorted(locks)[len(locks) // 2]
            avg_c = sum(confs) / len(confs)
            print(f"{ttype:<12} {len(locks):>6} {avg_l:>10.1f} {med_l:>10} {avg_c:>10.4f}")

    print(f"\n--- Degeneracy ---")
    print(f"Total final-token repeats: {total_final_repeats}/{total_positions} "
          f"({100*total_final_repeats/total_positions:.1f}%)")

    # Per-prompt degeneracy evolution
    for i, r in enumerate(results):
        degen = r['degen_per_step']
        n = len(degen)
        # Show at steps 1, n//4, n//2, 3n//4, n
        checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        vals = [degen[c] for c in checkpoints]
        print(f"  Prompt {i}: degen @ steps {checkpoints} = {vals}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--models', nargs='+', default=['sm_b2', 'sem_b05_3k'])
    parser.add_argument('--output', default='eval_results/sde_analysis_crystallization.json')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    all_results = {}

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        print(f"\n{'#'*70}")
        print(f"Loading model: {model_name} from {cfg['ckpt']}")
        print(f"  beta_infer={cfg['beta_infer']}, noise_scale={cfg['noise_scale']}, "
              f"schedule={cfg['schedule']}, steps={cfg['steps']}")

        model, ne, lb, bw, bb, bv, K = load_model_and_dsl(cfg['ckpt'], device)
        print(f"  Loaded. ne={ne.shape}, beta_trained={bv:.4f}")

        model_results = []
        for pi, prompt in enumerate(PROMPTS):
            print(f"\n--- Prompt {pi}: {prompt[:60]}... ---")
            t0 = time.time()
            result = sde_crystallization_analysis(
                prompt, model, ne, lb, bw, bb, bv, K, tokenizer, cfg, device
            )
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s. Final text: {result['final_text'][:80]}...")
            model_results.append(result)

        print_summary(model_results, model_name)
        all_results[model_name] = model_results

        # Free GPU memory before loading next model
        del model, ne, lb, bw, bb, K
        torch.cuda.empty_cache()

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")

    # Final comparison
    if len(args.models) == 2:
        m1, m2 = args.models
        print(f"\n{'='*70}")
        print(f"COMPARISON: {m1} vs {m2}")
        print(f"{'='*70}")

        for ttype in ['function', 'content', 'digit', 'punctuation', 'special']:
            locks1 = [p['lock_step'] for r in all_results[m1] for p in r['per_position'] if p['token_type'] == ttype]
            locks2 = [p['lock_step'] for r in all_results[m2] for p in r['per_position'] if p['token_type'] == ttype]
            if locks1 and locks2:
                avg1 = sum(locks1) / len(locks1)
                avg2 = sum(locks2) / len(locks2)
                print(f"  {ttype:<12}: {m1} avg_lock={avg1:.1f} ({len(locks1)} tokens) | "
                      f"{m2} avg_lock={avg2:.1f} ({len(locks2)} tokens)")

        rep1 = sum(r['final_repeats'] for r in all_results[m1])
        rep2 = sum(r['final_repeats'] for r in all_results[m2])
        total = GEN_LENGTH * len(PROMPTS)
        print(f"\n  Degeneracy: {m1}={rep1}/{total} ({100*rep1/total:.1f}%) | "
              f"{m2}={rep2}/{total} ({100*rep2/total:.1f}%)")


if __name__ == '__main__':
    main()
