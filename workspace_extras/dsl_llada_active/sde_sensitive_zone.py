"""SDE Sensitive Zone Analysis: Converter response curve + custom schedule optimization.

Usage: CUDA_VISIBLE_DEVICES=5 python dsl_llada/sde_sensitive_zone.py --gpu 0
"""
import argparse, json, os, sys, math, time, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import load_file

TOP_K = 512

MODELS = {
    'sm_b2': {'path': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000'},
    'b1': {'path': 'checkpoints/pertoken_b1_d100_1k/checkpoint-1000'},
    'sem_b05_3k': {'path': 'checkpoints/semantic_b05_d100_10k/checkpoint-3000'},
    'sem_b2': {'path': 'checkpoints/semantic_b2_d100_1k/checkpoint-1000'},
}

SNR_PROBE_POINTS = [0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]

PROMPTS = [
    "Write a short paragraph about photosynthesis.",
    "Explain machine learning in simple terms.",
    "What is the water cycle?",
    "Describe the solar system.",
    "Summarize the key aspects of democracy.",
    "Question: What is 27 + 58? Show the reasoning briefly.",
    "A store had 45 apples. It sold 18, then got 27 more. How many?",
    "Write a short story about a robot discovering music.",
    "Plan a 3-day Tokyo trip. Keep each day to 3 items.",
    "Explain quantum mechanics in simple terms.",
]


def load_dsl_weights(ckpt_path, device):
    """Load DSL weights from checkpoint safetensors."""
    ne, lb, bw, bb, bv = None, None, None, None, None
    for fname in sorted(os.listdir(ckpt_path)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt_path, fname), device=device)
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].item()
    # K needs V+1 rows (last row = mask slot embedding, zero vector)
    # ne is (V, d), lb is (V+1,), so pad ne with a zero row
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
    return ne, lb, bw, bb, K, bv


def converter_accuracy_at_snr(ne, lb, bw, bb, K, bv, tokens, snr_val, device, n_trials=5):
    """Measure converter accuracy at a fixed SNR value, averaged over n_trials noise samples."""
    accs = []
    for trial in range(n_trials):
        z = snr_val * ne[tokens[0]] + math.sqrt(snr_val) * torch.randn(1, tokens.shape[1], ne.shape[1], device=device)
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        pred = probs[:, :, :-1].argmax(dim=-1)  # exclude mask slot
        accs.append((pred == tokens).float().mean().item())
    return float(np.mean(accs))


def converter_mask_prob_at_snr(ne, lb, bw, bb, K, bv, tokens, snr_val, device):
    """Measure avg probability assigned to mask slot at given SNR."""
    z = snr_val * ne[tokens[0]] + math.sqrt(snr_val) * torch.randn(1, tokens.shape[1], ne.shape[1], device=device)
    cl = bv * (z.float() @ K.T) + lb
    probs = F.softmax(cl.float(), dim=-1)
    return probs[:, :, -1].mean().item()  # mask slot probability


def find_transition_zone(snr_points, accuracies, low_thresh=0.10, high_thresh=0.90):
    """Find SNR range where accuracy transitions from low_thresh to high_thresh."""
    snr_low, snr_high = snr_points[0], snr_points[-1]

    # Find lowest SNR where accuracy >= low_thresh
    for s, a in zip(snr_points, accuracies):
        if a >= low_thresh:
            snr_low = s
            break

    # Find lowest SNR where accuracy >= high_thresh
    for s, a in zip(snr_points, accuracies):
        if a >= high_thresh:
            snr_high = s
            break

    return snr_low, snr_high


def make_custom_schedule(snr_low, snr_high, steps, snr_min=0.01, snr_max=100):
    """Create schedule with 80% of steps in [snr_low, snr_high] transition zone."""
    n_pre = max(1, int(steps * 0.10))   # 10% below transition
    n_trans = max(1, int(steps * 0.80)) # 80% in transition
    n_post = steps - n_pre - n_trans    # 10% above transition
    if n_post < 1:
        n_post = 1
        n_trans = steps - n_pre - n_post

    snrs = torch.cat([
        torch.exp(torch.linspace(math.log(snr_min), math.log(max(snr_low, snr_min+0.01)), n_pre + 1)),
        torch.exp(torch.linspace(math.log(max(snr_low, snr_min+0.01)), math.log(snr_high), n_trans + 1))[1:],
        torch.exp(torch.linspace(math.log(snr_high), math.log(snr_max), n_post + 1))[1:],
    ])
    return snrs


def make_sensitive_schedule(steps, snr_min=0.01, snr_max=100):
    """Default sensitive schedule [7, 74]."""
    n1, n2 = max(1, int(steps * 0.05)), max(1, int(steps * 0.90))
    n3 = steps - n1 - n2
    if n3 < 1: n3 = 1; n2 = steps - n1 - n3
    snrs = torch.cat([
        torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
        torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
        torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
    ])
    return snrs


def make_uniform_schedule(steps, snr_min=0.01, snr_max=100):
    """Uniform log-spaced schedule."""
    return torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1))


def compute_ppl(texts, gpt2, gpt2_tok, device):
    nlls = []
    for t in texts:
        w = ' '.join(t.split()[:128])
        if not w.strip(): continue
        ids = gpt2_tok(w, return_tensors='pt', truncation=True, max_length=512)['input_ids'].to(device)
        if ids.shape[1] < 2: continue
        with torch.no_grad(): nlls.append(gpt2(ids, labels=ids).loss.item())
    return float(np.exp(np.mean(nlls))) if nlls else 999.


def rep_rate(texts):
    total, rep = 0, 0
    for t in texts:
        ws = t.split()
        for i in range(1, len(ws)):
            total += 1
            if ws[i] == ws[i-1]: rep += 1
    return rep / max(total, 1)


def sde_generate(model, ne, lb, bw, bb, K, bv, tokenizer, prompt, steps, snrs, ns, device, gen_length=256):
    """SDE generation with custom SNR schedule (snrs tensor)."""
    msgs = [{'role': 'user', 'content': prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)
    snrs = snrs.to(device)

    torch.manual_seed(42)
    y = F.normalize(torch.randn(1, gen_length, nd, device=device), dim=-1)

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        bp = F.softmax(logits, dim=-1)
        tv, ti = bp.topk(min(TOP_K, bp.shape[-1]), dim=-1)
        tv = tv / tv.sum(dim=-1, keepdim=True)
        return (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)

    for i in range(len(snrs) - 1):
        s, s_next = snrs[i], snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s; g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next; g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    z_f = snrs[-1] * y
    cl_f = bv * (z_f.float() @ K.T) + lb
    pf = F.softmax(cl_f.float(), dim=-1)
    hf = F.linear(pf, bw, bb).to(torch.bfloat16)
    ef = torch.cat([pe.to(torch.bfloat16), hf], dim=1)
    with torch.no_grad():
        lo = model(input_ids=dummy, inputs_embeds=ef).logits[:, pl:, :].float()
    return tokenizer.decode(lo.argmax(dim=-1)[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load best params
    search = json.load(open(os.path.join(base_dir, 'eval_results/sde_param_search.json')))

    # Load texts for converter accuracy measurement
    texts_data = json.load(open(os.path.join(base_dir, 'eval_data/texts_100.json')))

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    all_results = {}

    # ===== PHASE 1: Converter Response Curve =====
    print("=" * 70)
    print("PHASE 1: Converter Response Curve")
    print("=" * 70)

    for mk, minfo in MODELS.items():
        ckpt = os.path.join(base_dir, minfo['path'])
        print(f"\n--- {mk} ---")

        ne, lb, bw, bb, K, bv = load_dsl_weights(ckpt, device)
        print(f"  beta_trained = {bv:.4f}, noise_dim = {ne.shape[1]}")

        # Use 10 texts
        snr_accs = {snr: [] for snr in SNR_PROBE_POINTS}
        snr_mask_probs = {snr: [] for snr in SNR_PROBE_POINTS}

        for i, td in enumerate(texts_data[:10]):
            text = td['text']
            tids = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, add_special_tokens=False)['input_ids'].to(device)

            for snr_val in SNR_PROBE_POINTS:
                acc = converter_accuracy_at_snr(ne, lb, bw, bb, K, bv, tids, snr_val, device, n_trials=3)
                mp = converter_mask_prob_at_snr(ne, lb, bw, bb, K, bv, tids, snr_val, device)
                snr_accs[snr_val].append(acc)
                snr_mask_probs[snr_val].append(mp)

        # Average across texts
        curve = {}
        print(f"  {'SNR':>6s}  {'Acc':>6s}  {'MaskP':>6s}")
        for snr_val in SNR_PROBE_POINTS:
            avg_acc = float(np.mean(snr_accs[snr_val]))
            avg_mp = float(np.mean(snr_mask_probs[snr_val]))
            curve[str(snr_val)] = {'accuracy': round(avg_acc, 4), 'mask_prob': round(avg_mp, 4)}
            print(f"  {snr_val:6.1f}  {avg_acc:6.3f}  {avg_mp:6.3f}")

        # Find transition zone
        accs_list = [curve[str(s)]['accuracy'] for s in SNR_PROBE_POINTS]
        snr_low, snr_high = find_transition_zone(SNR_PROBE_POINTS, accs_list)
        print(f"  Transition zone (10%-90%): [{snr_low}, {snr_high}]")

        all_results[mk] = {
            'beta_trained': round(bv, 4),
            'noise_dim': ne.shape[1],
            'converter_curve': curve,
            'transition_zone': {'snr_low': snr_low, 'snr_high': snr_high},
        }

        # Free memory
        del ne, lb, bw, bb, K
        torch.cuda.empty_cache()

    # ===== PHASE 2: Design Custom Schedules =====
    print("\n" + "=" * 70)
    print("PHASE 2: Custom Schedule Design")
    print("=" * 70)

    for mk in MODELS:
        tz = all_results[mk]['transition_zone']
        print(f"\n{mk}: transition [{tz['snr_low']}, {tz['snr_high']}]")

        # Design schedule: 80% of steps in transition zone
        steps = 64
        custom_snrs = make_custom_schedule(tz['snr_low'], tz['snr_high'], steps)

        all_results[mk]['custom_schedule'] = {
            'snr_low': tz['snr_low'],
            'snr_high': tz['snr_high'],
            'description': f"80% of steps in [{tz['snr_low']}, {tz['snr_high']}]",
            'n_steps': steps,
        }
        print(f"  Custom: 80% of {steps} steps in [{tz['snr_low']}, {tz['snr_high']}]")
        print(f"  SNR range: {custom_snrs[0]:.3f} -> {custom_snrs[-1]:.3f}")

    # ===== PHASE 3: GenPPL Comparison =====
    print("\n" + "=" * 70)
    print("PHASE 3: GenPPL Comparison (custom vs uniform vs sensitive)")
    print("=" * 70)

    print("Loading GPT-2 Large for PPL evaluation...")
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
    gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2-large')

    steps = 64

    for mk, minfo in MODELS.items():
        ckpt = os.path.join(base_dir, minfo['path'])
        best = search[mk]['best']
        bi = float(best['beta_infer'])
        ns = float(best['noise_scale'])

        print(f"\n--- {mk} (bi={bi}, ns={ns}) ---")

        ne, lb, bw, bb, K, bv = load_dsl_weights(ckpt, device)

        model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        tz = all_results[mk]['transition_zone']

        schedules = {
            'uniform': make_uniform_schedule(steps),
            'sensitive_7_74': make_sensitive_schedule(steps),
            'custom': make_custom_schedule(tz['snr_low'], tz['snr_high'], steps),
        }

        gen_results = {}
        for sname, snrs in schedules.items():
            print(f"  {sname:20s}", end='', flush=True)
            t0 = time.time()

            # Override beta for generation
            bv_gen = bi

            texts = []
            for p in PROMPTS:
                txt = sde_generate(model, ne, lb, bw, bb, K, bv_gen, tokenizer, p, steps, snrs, ns, device)
                texts.append(txt)

            valid = [t for t in texts if t.strip()]
            ppl = compute_ppl(valid, gpt2, gpt2_tok, device)
            al = float(np.mean([len(t.split()) for t in valid])) if valid else 0
            rr = rep_rate(valid)
            elapsed = time.time() - t0

            print(f" | PPL={ppl:6.2f} len={al:5.0f} rep={rr:.3f} | {elapsed:.0f}s")

            gen_results[sname] = {
                'gen_ppl': round(ppl, 2),
                'avg_len': round(al, 1),
                'rep_rate': round(rr, 3),
                'time_s': round(elapsed, 1),
            }

        all_results[mk]['gen_ppl_comparison'] = gen_results
        all_results[mk]['best_params'] = {'beta_infer': bi, 'noise_scale': ns}

        del model, ne, lb, bw, bb, K
        torch.cuda.empty_cache()

    # ===== Save Results =====
    outpath = os.path.join(base_dir, 'eval_results/sde_analysis_sensitive_zone.json')
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {outpath}")

    # ===== Summary =====
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for mk in MODELS:
        r = all_results[mk]
        tz = r['transition_zone']
        print(f"\n{mk} (β={r['beta_trained']:.2f}):")
        print(f"  Transition zone: [{tz['snr_low']}, {tz['snr_high']}]")
        if 'gen_ppl_comparison' in r:
            gc = r['gen_ppl_comparison']
            for sname, sr in gc.items():
                print(f"  {sname:20s}: PPL={sr['gen_ppl']:6.2f}  rep={sr['rep_rate']:.3f}")


if __name__ == '__main__':
    main()
