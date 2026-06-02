"""SDE drift dynamics analysis for DSL-LLaDA models.

Analyzes per-step: drift norm, y norm, x_hat stability, noise-to-signal ratio,
and divergence detection across multiple prompts and models.

Usage: CUDA_VISIBLE_DEVICES=6 python dsl_llada/analyze_sde_drift.py
"""
import torch
import torch.nn.functional as F
import sys, os, math, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
SDE_TOP_K = 512

PROMPTS = [
    "Question: What is 27 + 58? Show the reasoning briefly, then give the final answer.",
    "Plan a 3-day Tokyo trip for a first-time visitor. Day 1, Day 2, Day 3. Keep each day to 3 items.",
    "Write a short story about a robot hearing rain for the first time.",
    "There are three boxes: red, blue, green. Exactly one contains a coin. Red is empty. If blue has the coin, green is empty. Where can the coin be?",
    "Lena reads 12 pages on Monday, 15 on Tuesday, and twice as many on Wednesday as Monday. How many pages did she read in total?",
]

MODEL_CONFIGS = {
    'sm_b2': {
        'ckpt': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
        'beta_infer': 4.0,
        'noise_scale': 0.01,
        'use_sensitive': False,  # uniform schedule
    },
    'sem_b05_3k': {
        'ckpt': 'checkpoints/semantic_b05_d100_10k/checkpoint-3000',
        'beta_infer': 1.0,
        'noise_scale': 0.05,
        'use_sensitive': True,  # sensitive schedule
    },
}

STEPS = 64
GEN = 256
SNR_MAX = 80.0
SEED = 42


def load_dsl_weights(ckpt_path, device):
    """Load DSL weights from checkpoint safetensors."""
    ne, lb, bw, bb, bv = None, None, None, None, None
    for fn in sorted(os.listdir(ckpt_path)):
        if fn.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt_path, fn), device=device)
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].item()
                break
    assert ne is not None, f"No DSL weights found in {ckpt_path}"
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
    return ne, lb, bw, bb, bv, K


def make_snr_schedule(steps, snr_max, use_sensitive, device):
    """Create SNR schedule."""
    snr_min = 0.01
    if use_sensitive:
        n1 = max(1, int(steps * 0.05))
        n2 = max(1, int(steps * 0.90))
        n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1)).to(device)
    return snrs


def run_sde_analysis(model_name, cfg, device):
    """Run SDE on all prompts and collect drift dynamics."""
    ckpt = cfg['ckpt']
    beta_infer = cfg['beta_infer']
    ns = cfg['noise_scale']
    use_sensitive = cfg['use_sensitive']

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name} (ckpt={ckpt})")
    print(f"  beta_infer={beta_infer}, ns={ns}, sensitive={use_sensitive}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    ne, lb, bw, bb, bv, K = load_dsl_weights(ckpt, device)
    nd = ne.shape[1]

    print(f"  Loaded: ne={ne.shape}, beta_trained={bv:.4f}, using beta_infer={beta_infer}")

    snrs = make_snr_schedule(STEPS, SNR_MAX, use_sensitive, device)
    print(f"  SNR schedule: {len(snrs)-1} steps, [{snrs[0].item():.4f} -> {snrs[-1].item():.2f}]")

    def get_xhat(yy, ss, pe, pl, dummy):
        """Get x_hat and backbone predictions."""
        z = ss * yy
        cl = beta_infer * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h_low = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h_low], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        backbone_probs = F.softmax(logits, dim=-1)
        # x_hat via top-k weighted sum of noise_embed
        top_vals, top_idx = backbone_probs.topk(min(SDE_TOP_K, backbone_probs.shape[-1]), dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        xh = (top_vals.unsqueeze(-1) * ne[top_idx.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)
        return xh, backbone_probs

    model_results = []

    for pi, prompt in enumerate(PROMPTS):
        t0 = time.time()
        msgs = [{'role': 'user', 'content': prompt}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        pl = iids.shape[1]
        pe = model.model.transformer.wte(iids).float()
        dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)

        torch.manual_seed(SEED)
        y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)

        step_data = []
        prev_xhat_argmax = None

        actual_steps = len(snrs) - 1
        for i in range(actual_steps):
            s = snrs[i]
            s_next = snrs[i + 1]
            ds = s_next - s
            dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

            # Predictor
            xh, backbone_probs = get_xhat(y, s, pe, pl, dummy)
            top1 = backbone_probs[0].argmax(dim=-1)
            f = (xh - y) / s  # drift
            g = ns / s        # noise coefficient

            # Compute metrics BEFORE update
            drift_vec = f * ds  # actual displacement from drift
            noise_vec = g * dW  # actual displacement from noise

            drift_norm = drift_vec.norm(dim=-1).mean().item()  # avg across positions
            noise_norm = noise_vec.norm(dim=-1).mean().item()
            y_norm = y.norm(dim=-1).mean().item()
            f_norm = f.norm(dim=-1).mean().item()  # raw drift magnitude

            # x_hat stability: fraction of positions where argmax matches previous step
            if prev_xhat_argmax is not None:
                xhat_agreement = (top1 == prev_xhat_argmax).float().mean().item()
            else:
                xhat_agreement = 0.0

            # Noise-to-signal ratio
            nsr = noise_norm / max(drift_norm, 1e-10)

            # Per-position drift norm stats
            per_pos_drift = drift_vec[0].norm(dim=-1)  # (GEN,)
            drift_max = per_pos_drift.max().item()
            drift_min = per_pos_drift.min().item()
            drift_std = per_pos_drift.std().item()

            # Confidence
            conf = backbone_probs[0].max(dim=-1).values.mean().item()

            step_data.append({
                'step': i,
                'snr': float(s.item()),
                'snr_next': float(s_next.item()),
                'ds': float(ds.item()),
                'drift_norm': drift_norm,
                'drift_max': drift_max,
                'drift_min': drift_min,
                'drift_std': drift_std,
                'noise_norm': noise_norm,
                'noise_to_signal': nsr,
                'y_norm': y_norm,
                'f_norm_raw': f_norm,
                'xhat_agreement': xhat_agreement,
                'confidence': conf,
            })

            prev_xhat_argmax = top1.clone()

            # Heun corrector
            y_euler = y + f * ds + g * dW
            xh_e, _ = get_xhat(y_euler, s_next, pe, pl, dummy)
            f_e = (xh_e - y_euler) / s_next
            g_e = ns / s_next
            y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

        # Final decode
        z_final = snrs[-1] * y
        cl_final = beta_infer * (z_final.float() @ K.T) + lb
        final_probs = F.softmax(cl_final.float(), dim=-1)
        h_final = F.linear(final_probs, bw, bb).to(torch.bfloat16)
        embeds_final = torch.cat([pe.to(torch.bfloat16), h_final], dim=1)
        with torch.no_grad():
            lo_final = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
        final_toks = lo_final.argmax(dim=-1)[0]
        final_text = tokenizer.decode(final_toks, skip_special_tokens=True)

        elapsed = time.time() - t0
        print(f"  Prompt {pi+1}/{len(PROMPTS)} done ({elapsed:.1f}s): {prompt[:50]}...")

        model_results.append({
            'prompt_idx': pi,
            'prompt': prompt[:80],
            'steps': step_data,
            'final_text_preview': final_text[:200],
        })

    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()

    return model_results


def compute_summary(model_name, results):
    """Compute aggregate summary statistics."""
    all_steps = [s for r in results for s in r['steps']]
    n_steps = len(results[0]['steps'])

    # Per-step averages across prompts
    per_step_avg = {}
    for si in range(n_steps):
        step_vals = [r['steps'][si] for r in results]
        per_step_avg[si] = {
            'snr': step_vals[0]['snr'],
            'drift_norm': sum(s['drift_norm'] for s in step_vals) / len(step_vals),
            'y_norm': sum(s['y_norm'] for s in step_vals) / len(step_vals),
            'noise_to_signal': sum(s['noise_to_signal'] for s in step_vals) / len(step_vals),
            'xhat_agreement': sum(s['xhat_agreement'] for s in step_vals) / len(step_vals),
            'confidence': sum(s['confidence'] for s in step_vals) / len(step_vals),
            'f_norm_raw': sum(s['f_norm_raw'] for s in step_vals) / len(step_vals),
        }

    # Find phases
    # Phase 1: noise dominates (nsr > 1)
    noise_dominant_end = 0
    for si in range(n_steps):
        if per_step_avg[si]['noise_to_signal'] < 1.0:
            noise_dominant_end = si
            break

    # Phase 2: drift spike detection (drift_norm > 2x moving average)
    drift_norms = [per_step_avg[si]['drift_norm'] for si in range(n_steps)]
    spikes = []
    window = 5
    for si in range(window, n_steps):
        avg_prev = sum(drift_norms[si-window:si]) / window
        if avg_prev > 0 and drift_norms[si] > 2 * avg_prev:
            spikes.append(si)

    # Phase 3: stability convergence (xhat_agreement > 0.95)
    stability_step = n_steps
    for si in range(1, n_steps):
        if per_step_avg[si]['xhat_agreement'] > 0.95:
            stability_step = si
            break

    # Drift collapse detection (drift < 0.001 while y_norm still changing)
    collapse_step = None
    for si in range(n_steps - 1):
        if drift_norms[si] < 0.001 and per_step_avg[si]['y_norm'] < 0.5 * per_step_avg[n_steps-1]['y_norm']:
            collapse_step = si
            break

    return {
        'per_step_avg': {str(k): v for k, v in per_step_avg.items()},
        'noise_dominant_until_step': noise_dominant_end,
        'noise_dominant_until_snr': per_step_avg[noise_dominant_end]['snr'] if noise_dominant_end < n_steps else None,
        'drift_spikes_at_steps': spikes,
        'stability_converged_step': stability_step,
        'stability_converged_snr': per_step_avg[stability_step]['snr'] if stability_step < n_steps else None,
        'drift_collapse_step': collapse_step,
        'final_y_norm': per_step_avg[n_steps-1]['y_norm'],
        'final_confidence': per_step_avg[n_steps-1]['confidence'],
        'final_xhat_agreement': per_step_avg[n_steps-1]['xhat_agreement'],
    }


def print_summary(model_name, summary):
    """Print human-readable summary."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  Noise dominates until step {summary['noise_dominant_until_step']} (SNR={summary['noise_dominant_until_snr']:.3f})" if summary['noise_dominant_until_snr'] else "  Noise never dominates")
    print(f"  x_hat stabilizes at step {summary['stability_converged_step']} (SNR={summary['stability_converged_snr']:.2f})" if summary['stability_converged_snr'] else f"  x_hat never fully stabilizes (best agreement at final step)")
    print(f"  Drift spikes at steps: {summary['drift_spikes_at_steps']}" if summary['drift_spikes_at_steps'] else "  No drift spikes detected")
    print(f"  Drift collapse at step: {summary['drift_collapse_step']}" if summary['drift_collapse_step'] else "  No drift collapse")
    print(f"  Final: y_norm={summary['final_y_norm']:.3f}, conf={summary['final_confidence']:.4f}, agreement={summary['final_xhat_agreement']:.4f}")

    # Print trajectory at key points
    avg = summary['per_step_avg']
    n = len(avg)
    key_steps = [0, n//8, n//4, n//2, 3*n//4, n-1]
    print(f"\n  {'Step':>5} {'SNR':>8} {'Drift':>10} {'||y||':>8} {'N/S ratio':>10} {'Agree':>8} {'Conf':>8}")
    print(f"  {'-'*63}")
    for si in key_steps:
        s = avg[str(si)]
        print(f"  {si:5d} {s['snr']:8.3f} {s['drift_norm']:10.5f} {s['y_norm']:8.3f} {s['noise_to_signal']:10.4f} {s['xhat_agreement']:8.4f} {s['confidence']:8.4f}")


def main():
    device = 'cuda:0'
    os.makedirs('eval_results', exist_ok=True)

    all_output = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        results = run_sde_analysis(model_name, cfg, device)
        summary = compute_summary(model_name, results)
        print_summary(model_name, summary)

        all_output[model_name] = {
            'config': {
                'ckpt': cfg['ckpt'],
                'beta_infer': cfg['beta_infer'],
                'noise_scale': cfg['noise_scale'],
                'use_sensitive': cfg['use_sensitive'],
                'steps': STEPS,
                'gen_length': GEN,
                'snr_max': SNR_MAX,
                'seed': SEED,
            },
            'prompts': [r['prompt'] for r in results],
            'per_prompt_steps': [r['steps'] for r in results],
            'final_texts': [r['final_text_preview'] for r in results],
            'summary': summary,
        }

    out_path = 'eval_results/sde_analysis_drift.json'
    with open(out_path, 'w') as f:
        json.dump(all_output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
