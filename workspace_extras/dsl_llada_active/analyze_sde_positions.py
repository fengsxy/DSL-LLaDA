"""SDE per-position trajectory analysis for DSL-LLaDA models.

Records full per-position token trajectory across SDE steps, classifies
failure modes, and compares converter vs backbone confidence.

Usage:
    CUDA_VISIBLE_DEVICES=7 python dsl_llada/analyze_sde_positions.py --gpu 0
"""
import os, sys, math, json, argparse
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['DSL_RESIDUAL'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
SDE_TOP_K = 512

MODELS_CONFIG = {
    'sm_b2': {
        'ckpt': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
        'beta_infer': 4.0,
        'noise_scale': 0.01,
        'use_sensitive': False,  # uniform schedule
        'steps': 64,
        'snr_max': 80.0,
    },
    'sem_b05_3k': {
        'ckpt': 'checkpoints/semantic_b05_d100_10k/checkpoint-3000',
        'beta_infer': 1.0,
        'noise_scale': 0.05,
        'use_sensitive': True,  # sensitive schedule
        'steps': 64,
        'snr_max': 80.0,
    },
}

PROMPTS = [
    "Write a short paragraph about photosynthesis.",
    "Question: What is 27 + 58? Show the reasoning briefly.",
    "Plan a 3-day Tokyo trip. Keep each day to 3 items.",
]

GEN_LENGTH = 256
SEED = 42


def load_dsl_model(ckpt_path, device):
    """Load LLaDA model + DSL weights from checkpoint."""
    # Fix dtype serialization issue in some checkpoints
    import json as _json
    cfg_path = os.path.join(ckpt_path, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = _json.load(f)
        if 'dtype' in cfg:
            del cfg['dtype']
            with open(cfg_path, 'w') as f:
                _json.dump(cfg, f, indent=2)

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    ne, lb, bw, bb, bv, rw, rb = None, None, None, None, 2.5, None, None
    for fname in sorted(os.listdir(ckpt_path)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt_path, fname), device=str(device))
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].float().item()
                rw = st.get('converter.residual_proj.weight')
                rb = st.get('converter.residual_proj.bias')
                if rw is not None: rw = rw.float()
                if rb is not None: rb = rb.float()
                break

    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
    return model, ne, lb, bw, bb, bv, rw, rb, K


def run_sde_with_trajectory(model, ne, lb, bw, bb, bv, rw, rb, K, tokenizer,
                            prompt, steps, gen_length, noise_scale, snr_max,
                            use_sensitive, device, seed=42):
    """Run SDE and record full per-position trajectory."""
    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    GEN = gen_length
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)

    # SNR schedule
    snr_min = 0.01
    if use_sensitive:
        n1 = max(1, int(steps * 0.05)); n2 = max(1, int(steps * 0.90)); n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1)).to(device)

    # Init
    torch.manual_seed(seed)
    y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)

    ns = noise_scale
    use_res = (rw is not None)

    # Trajectory storage
    # token_trajectory[step][pos] = token_id
    token_trajectory = []
    converter_conf_trajectory = []  # [step][pos] = max converter prob
    backbone_conf_trajectory = []   # [step][pos] = max backbone prob
    converter_token_trajectory = []  # [step][pos] = converter argmax token
    snr_values = []

    def get_xhat_detailed(yy, ss):
        """Returns xhat, backbone_probs, per-position converter info."""
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)

        # Converter per-position info
        conv_max_prob, conv_argmax = probs.max(dim=-1)  # [1, GEN]
        conv_max_prob = conv_max_prob[0]  # [GEN]
        conv_argmax = conv_argmax[0]  # [GEN]

        h_low = F.linear(probs, bw, bb)
        if use_res and rw is not None:
            embed_w = ne.to(z.dtype)
            z_low = probs[:, :, :embed_w.shape[0]].to(z.dtype) @ embed_w
            z_ln = z_low / (z_low.norm(dim=-1, keepdim=True) + 1e-8)
            ps = (z * z_ln).sum(dim=-1, keepdim=True)
            z_high = z - ps * z_ln
            h_high = F.linear(z_high.to(rw.dtype), rw, rb)
            h = (h_low + h_high).to(torch.bfloat16)
        else:
            h = h_low.to(torch.bfloat16)

        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        backbone_probs = F.softmax(logits, dim=-1)

        # Backbone per-position info
        bb_max_prob, bb_argmax = backbone_probs.max(dim=-1)  # [1, GEN]
        bb_max_prob = bb_max_prob[0]  # [GEN]
        bb_argmax = bb_argmax[0]  # [GEN]

        # xhat for SDE drift
        top_vals, top_idx = backbone_probs.topk(min(SDE_TOP_K, backbone_probs.shape[-1]), dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        xh = (top_vals.unsqueeze(-1) * ne[top_idx.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)

        return xh, bb_argmax, bb_max_prob, conv_argmax, conv_max_prob

    actual_steps = len(snrs) - 1
    print(f"  Running {actual_steps} SDE steps...", flush=True)

    for i in range(actual_steps):
        s = snrs[i]
        s_next = snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        xh, bb_tok, bb_conf, cv_tok, cv_conf = get_xhat_detailed(y, s)

        # Record trajectory
        token_trajectory.append(bb_tok.cpu().tolist())
        backbone_conf_trajectory.append(bb_conf.cpu().tolist())
        converter_conf_trajectory.append(cv_conf.cpu().tolist())
        converter_token_trajectory.append(cv_tok.cpu().tolist())
        snr_values.append(float(s))

        # Heun step
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW

        xh_e, _, _, _, _ = get_xhat_detailed(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

        if (i + 1) % 16 == 0:
            print(f"    Step {i+1}/{actual_steps}, SNR={float(s):.2f}, "
                  f"bb_conf={bb_conf.mean():.3f}, cv_conf={cv_conf.mean():.3f}", flush=True)

    # Final step
    xh_final, bb_tok_final, bb_conf_final, cv_tok_final, cv_conf_final = get_xhat_detailed(y, snrs[-1])
    token_trajectory.append(bb_tok_final.cpu().tolist())
    backbone_conf_trajectory.append(bb_conf_final.cpu().tolist())
    converter_conf_trajectory.append(cv_conf_final.cpu().tolist())
    converter_token_trajectory.append(cv_tok_final.cpu().tolist())
    snr_values.append(float(snrs[-1]))

    final_tokens = bb_tok_final.cpu().tolist()
    final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)

    return {
        'prompt': prompt,
        'prompt_length': pl,
        'gen_length': GEN,
        'final_text': final_text,
        'final_tokens': final_tokens,
        'token_trajectory': token_trajectory,       # [step+1][pos]
        'backbone_conf': backbone_conf_trajectory,   # [step+1][pos]
        'converter_conf': converter_conf_trajectory, # [step+1][pos]
        'converter_tokens': converter_token_trajectory, # [step+1][pos]
        'snr_values': snr_values,
    }


def classify_positions(result, tokenizer):
    """Classify each position's failure mode."""
    n_steps = len(result['token_trajectory'])
    n_pos = len(result['token_trajectory'][0])
    final_tokens = result['final_tokens']

    classifications = []
    convergence_steps = []

    for pos in range(n_pos):
        tok = final_tokens[pos]
        tok_text = tokenizer.decode([tok], skip_special_tokens=False)

        # Check convergence step: when did this position lock in?
        lock_step = n_steps - 1
        for s in range(n_steps - 2, -1, -1):
            if result['token_trajectory'][s][pos] != tok:
                lock_step = s + 1
                break
        else:
            lock_step = 0  # always same token
        convergence_steps.append(lock_step)

        # Check repetition with neighbors
        is_repetition = False
        if pos > 0 and final_tokens[pos] == final_tokens[pos - 1]:
            is_repetition = True
        if pos < n_pos - 1 and final_tokens[pos] == final_tokens[pos + 1]:
            is_repetition = True

        # Check for longer repetition patterns (3+ consecutive same tokens)
        rep_count = 1
        for p2 in range(pos + 1, min(pos + 10, n_pos)):
            if final_tokens[p2] == tok:
                rep_count += 1
            else:
                break
        for p2 in range(pos - 1, max(pos - 10, -1), -1):
            if final_tokens[p2] == tok:
                rep_count += 1
            else:
                break

        # Degenerate: stuck on punctuation/special/very common tokens
        is_degenerate = tok_text.strip() in ['', '.', ',', '!', '?', '\n', '<|endoftext|>',
                                              '<s>', '</s>', '<pad>', ' ', '  '] and rep_count >= 3

        # Early lock: locked before step 30 (of ~65 total steps) to final token
        is_early_lock = lock_step < 30 and lock_step > 0

        # Classify
        if is_degenerate and rep_count >= 3:
            cls = 'DEGENERATE'
        elif is_repetition and rep_count >= 3:
            cls = 'REPETITION'
        elif is_early_lock and (is_repetition or is_degenerate):
            cls = 'EARLY_LOCK'
        else:
            # Check if it's a reasonable token (not just checking correctness
            # since we don't have ground truth, but looking for structural issues)
            cls = 'NORMAL'

        classifications.append({
            'position': pos,
            'token_id': tok,
            'token_text': tok_text,
            'classification': cls,
            'convergence_step': lock_step,
            'repetition_count': rep_count,
        })

    return classifications, convergence_steps


def analyze_converter_backbone_disagreement(result):
    """Find cases where converter and backbone disagree."""
    n_steps = len(result['token_trajectory'])
    n_pos = len(result['token_trajectory'][0])

    disagreements = []
    for s in range(n_steps):
        n_disagree = 0
        conv_right_bb_wrong = 0
        bb_right_conv_wrong = 0
        both_wrong = 0
        conv_confident_wrong = 0

        for pos in range(n_pos):
            bb_tok = result['token_trajectory'][s][pos]
            cv_tok = result['converter_tokens'][s][pos]
            final_tok = result['final_tokens'][pos]

            if bb_tok != cv_tok:
                n_disagree += 1
                cv_conf = result['converter_conf'][s][pos]

                if cv_tok == final_tok and bb_tok != final_tok:
                    conv_right_bb_wrong += 1
                elif bb_tok == final_tok and cv_tok != final_tok:
                    bb_right_conv_wrong += 1
                elif bb_tok != final_tok and cv_tok != final_tok:
                    both_wrong += 1

                # Converter confident but wrong (backbone correct)
                if cv_conf > 0.5 and cv_tok != final_tok and bb_tok == final_tok:
                    conv_confident_wrong += 1

        disagreements.append({
            'step': s,
            'snr': result['snr_values'][s],
            'n_disagree': n_disagree,
            'n_total': n_pos,
            'disagree_pct': n_disagree / n_pos * 100,
            'conv_right_bb_wrong': conv_right_bb_wrong,
            'bb_right_conv_wrong': bb_right_conv_wrong,
            'both_wrong': both_wrong,
            'conv_confident_wrong': conv_confident_wrong,
        })

    return disagreements


def print_summary(all_results):
    """Print comparison summary across models."""
    print("\n" + "=" * 80)
    print("SDE PER-POSITION ANALYSIS SUMMARY")
    print("=" * 80)

    for model_name, model_results in all_results.items():
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        total_normal = 0
        total_rep = 0
        total_degen = 0
        total_early = 0
        total_pos = 0
        all_conv_steps = []
        all_disagree = []

        for prompt_result in model_results:
            result = prompt_result['trajectory']
            classifications = prompt_result['classifications']
            disagreements = prompt_result['disagreements']
            convergence_steps = prompt_result['convergence_steps']

            prompt_short = result['prompt'][:50]
            print(f"\n  Prompt: \"{prompt_short}...\"")
            print(f"  Final text (first 200 chars): {result['final_text'][:200]}")

            # Count classifications
            counts = defaultdict(int)
            for c in classifications:
                counts[c['classification']] += 1
            n = len(classifications)
            total_pos += n

            for cls_name in ['NORMAL', 'REPETITION', 'DEGENERATE', 'EARLY_LOCK']:
                ct = counts.get(cls_name, 0)
                pct = ct / n * 100 if n > 0 else 0
                print(f"    {cls_name}: {ct}/{n} ({pct:.1f}%)")
                if cls_name == 'NORMAL': total_normal += ct
                elif cls_name == 'REPETITION': total_rep += ct
                elif cls_name == 'DEGENERATE': total_degen += ct
                elif cls_name == 'EARLY_LOCK': total_early += ct

            # Position vs convergence
            # Bin by distance from start (proxy for prompt distance)
            bins = [(0, 32), (32, 64), (64, 128), (128, 192), (192, 256)]
            print(f"\n    Convergence step by position range:")
            for lo, hi in bins:
                steps_in_bin = [convergence_steps[p] for p in range(lo, min(hi, len(convergence_steps)))]
                if steps_in_bin:
                    avg = sum(steps_in_bin) / len(steps_in_bin)
                    print(f"      pos [{lo:3d}, {hi:3d}): avg convergence step = {avg:.1f}")
            all_conv_steps.extend(convergence_steps)

            # Converter-backbone disagreement summary
            print(f"\n    Converter-Backbone disagreement by SNR range:")
            snr_bins = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 100)]
            for slo, shi in snr_bins:
                matching = [d for d in disagreements if slo <= d['snr'] < shi]
                if matching:
                    avg_disagree = sum(d['disagree_pct'] for d in matching) / len(matching)
                    avg_conv_conf_wrong = sum(d['conv_confident_wrong'] for d in matching) / len(matching)
                    avg_bb_right = sum(d['bb_right_conv_wrong'] for d in matching) / len(matching)
                    print(f"      SNR [{slo:3d}, {shi:3d}): disagree={avg_disagree:.1f}%, "
                          f"conv_confident_wrong={avg_conv_conf_wrong:.1f}, "
                          f"bb_overrides_conv={avg_bb_right:.1f}")

            all_disagree.extend(disagreements)

        # Model totals
        print(f"\n  === MODEL TOTALS ===")
        print(f"  NORMAL:     {total_normal}/{total_pos} ({total_normal/total_pos*100:.1f}%)")
        print(f"  REPETITION: {total_rep}/{total_pos} ({total_rep/total_pos*100:.1f}%)")
        print(f"  DEGENERATE: {total_degen}/{total_pos} ({total_degen/total_pos*100:.1f}%)")
        print(f"  EARLY_LOCK: {total_early}/{total_pos} ({total_early/total_pos*100:.1f}%)")

        if all_conv_steps:
            avg_conv = sum(all_conv_steps) / len(all_conv_steps)
            print(f"  Average convergence step: {avg_conv:.1f} / {len(all_results[model_name][0]['trajectory']['token_trajectory'])}")

    # Cross-model comparison
    print(f"\n{'=' * 80}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 80}")

    model_names = list(all_results.keys())
    if len(model_names) >= 2:
        for pi, prompt in enumerate(PROMPTS):
            print(f"\n  Prompt {pi+1}: \"{prompt[:50]}...\"")
            for mn in model_names:
                result = all_results[mn][pi]
                cls = result['classifications']
                counts = defaultdict(int)
                for c in cls: counts[c['classification']] += 1
                n = len(cls)
                normal_pct = counts['NORMAL'] / n * 100
                rep_pct = counts['REPETITION'] / n * 100
                degen_pct = counts['DEGENERATE'] / n * 100
                avg_conv = sum(result['convergence_steps']) / len(result['convergence_steps'])
                # Average final confidence
                final_bb_conf = sum(result['trajectory']['backbone_conf'][-1]) / n
                final_cv_conf = sum(result['trajectory']['converter_conf'][-1]) / n
                print(f"    {mn:15s}: Normal={normal_pct:5.1f}% Rep={rep_pct:5.1f}% "
                      f"Degen={degen_pct:5.1f}% AvgConv={avg_conv:5.1f} "
                      f"FinalBBconf={final_bb_conf:.3f} FinalCVconf={final_cv_conf:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, default='eval_results/sde_analysis_positions.json')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    print(f"Using device: {device}")

    # Load tokenizer
    tok_path = 'GSAI-ML/LLaDA-8B-Instruct'
    # Try local first
    local_tok = 'checkpoints/semantic_b05_d100_10k/checkpoint-3000'
    if os.path.exists(os.path.join(local_tok, 'tokenizer.json')):
        tok_path = local_tok
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    print(f"Tokenizer loaded from {tok_path}")

    all_results = {}

    for model_name, cfg in MODELS_CONFIG.items():
        print(f"\n{'=' * 60}")
        print(f"Loading model: {model_name}")
        print(f"{'=' * 60}")

        ckpt = cfg['ckpt']
        model, ne, lb, bw, bb, bv, rw, rb, K = load_dsl_model(ckpt, device)
        print(f"  Loaded. beta_train={bv:.3f}, beta_infer={cfg['beta_infer']}, "
              f"noise_dim={ne.shape[1]}, noise_scale={cfg['noise_scale']}")

        # Override beta for inference
        bv_infer = cfg['beta_infer']

        model_results = []
        for pi, prompt in enumerate(PROMPTS):
            print(f"\n  Prompt {pi+1}: \"{prompt[:50]}...\"")
            result = run_sde_with_trajectory(
                model, ne, lb, bw, bb, bv_infer, rw, rb, K, tokenizer,
                prompt, cfg['steps'], GEN_LENGTH, cfg['noise_scale'],
                cfg['snr_max'], cfg['use_sensitive'], device, SEED
            )
            print(f"  Final text: {result['final_text'][:150]}...")

            classifications, convergence_steps = classify_positions(result, tokenizer)
            disagreements = analyze_converter_backbone_disagreement(result)

            model_results.append({
                'trajectory': result,
                'classifications': classifications,
                'convergence_steps': convergence_steps,
                'disagreements': disagreements,
            })

        all_results[model_name] = model_results

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # Print summary
    print_summary(all_results)

    # Save to JSON (make trajectory data lighter for JSON)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_data = {}
    for model_name, model_results in all_results.items():
        save_data[model_name] = []
        for pr in model_results:
            # Keep everything but subsample trajectory for file size
            traj = pr['trajectory']
            # Keep every 4th step + first + last for trajectory
            n_steps = len(traj['token_trajectory'])
            keep_steps = sorted(set([0, n_steps - 1] + list(range(0, n_steps, 4))))

            save_data[model_name].append({
                'prompt': traj['prompt'],
                'final_text': traj['final_text'],
                'final_tokens': traj['final_tokens'],
                'snr_values': [traj['snr_values'][s] for s in keep_steps],
                'token_trajectory_sampled': {
                    'steps': keep_steps,
                    'tokens': [traj['token_trajectory'][s] for s in keep_steps],
                },
                'backbone_conf_sampled': {
                    'steps': keep_steps,
                    'values': [traj['backbone_conf'][s] for s in keep_steps],
                },
                'converter_conf_sampled': {
                    'steps': keep_steps,
                    'values': [traj['converter_conf'][s] for s in keep_steps],
                },
                'classifications': pr['classifications'],
                'convergence_steps': pr['convergence_steps'],
                'disagreements': pr['disagreements'],
            })

    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
