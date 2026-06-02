"""SDE parameter search for semantic_b05_d100_1k model.
Efficient 2-phase: coarse (1 prompt, gen=32, steps=16) -> fine (3 prompts, gen=128, steps varied).
"""
import os, sys, math, json, time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

MASK_ID = 126336
device = 'cuda:0'
CKPT = 'checkpoints/semantic_b05_d100_1k/checkpoint-1000'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

# Load DSL weights
ne, lb, bw, bb, bv = None, None, None, None, None
for fname in sorted(os.listdir(CKPT)):
    if fname.endswith('.safetensors'):
        st = load_file(os.path.join(CKPT, fname), device=device)
        if 'noise_embed.weight' in st:
            ne = st['noise_embed.weight'].float()
            lb = st['converter.logit_bias'].float()
            bw = st['converter.backbone_embedding.weight'].float()
            bb = st['converter.backbone_embedding.bias'].float()
            bv = st['converter.beta'].float().item()
            break

assert ne is not None, "DSL weights not found"
nd = ne.shape[1]
K = torch.cat([ne, torch.zeros(1, nd, device=device)], dim=0)
print(f"DSL: noise_dim={nd}, beta={bv:.4f}")

# ---- SNR schedules ----
def build_snr_loglinear(snr_min, snr_max, steps):
    return torch.exp(torch.linspace(math.log(max(snr_min, 1e-6)), math.log(snr_max), steps + 1))

def build_snr_linear(snr_min, snr_max, steps):
    return torch.linspace(snr_min, snr_max, steps + 1)

def build_snr_karras(snr_min, snr_max, steps, rho=7.0):
    sigma_min = math.sqrt(1.0 / max(snr_max, 1e-12))
    sigma_max = math.sqrt(1.0 / max(snr_min, 1e-6))
    ramp = torch.linspace(0.0, 1.0, steps + 1)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return 1.0 / (sigmas ** 2)

def build_snr_sensitive(snr_min, snr_max, steps):
    n1 = max(1, int(steps * 0.05)); n2 = max(1, int(steps * 0.90)); n3 = steps - n1 - n2
    mid_lo = max(snr_min * 10, 3.0); mid_hi = snr_max * 0.7
    return torch.cat([
        torch.exp(torch.linspace(math.log(max(snr_min, 1e-6)), math.log(mid_lo), n1 + 1)),
        torch.exp(torch.linspace(math.log(mid_lo), math.log(mid_hi), n2 + 1))[1:],
        torch.exp(torch.linspace(math.log(mid_hi), math.log(snr_max), n3 + 1))[1:],
    ])

schedules = {
    'loglinear': build_snr_loglinear,
    'linear': build_snr_linear,
    'karras': build_snr_karras,
    'sensitive': build_snr_sensitive,
}

# ---- SDE Heun ----
@torch.no_grad()
def sde_heun_generate(prompt_text, snrs, noise_scale, gen_length, seed=42):
    msgs = [{'role': 'user', 'content': prompt_text}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    GEN = gen_length
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)
    snrs_dev = snrs.to(device).float()

    torch.manual_seed(seed)
    y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)
    ns = noise_scale
    actual_steps = len(snrs_dev) - 1

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        bp = F.softmax(logits, dim=-1)
        sp, si = bp.sort(dim=-1, descending=True)
        cm = sp.cumsum(dim=-1); mask = cm - sp > 0.9
        sp[mask] = 0; sp = sp / sp.sum(dim=-1, keepdim=True)
        filt = torch.zeros_like(bp); filt.scatter_(-1, si, sp)
        xh = filt[:, :, :ne.shape[0]] @ ne
        return xh, bp, probs.max(dim=-1).values.mean().item()

    for i in range(actual_steps):
        s, s_next = snrs_dev[i], snrs_dev[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh, bp, cc = get_xhat(y, s)
        f = (xh - y) / s; g = ns / s if s > 0 else 0
        y_e = y + f * ds + g * dW
        xh_e, _, _ = get_xhat(y_e, s_next)
        f_e = (xh_e - y_e) / s_next; g_e = ns / s_next if s_next > 0 else 0
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    # Final decode
    z_final = snrs_dev[-1] * y
    cl_final = bv * (z_final.float() @ K.T) + lb
    fp = F.softmax(cl_final.float(), dim=-1)
    h_final = F.linear(fp, bw, bb).to(torch.bfloat16)
    embeds_final = torch.cat([pe.to(torch.bfloat16), h_final], dim=1)
    lo = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
    toks = lo.argmax(dim=-1)[0]
    text = tokenizer.decode(toks, skip_special_tokens=True)
    final_bp = F.softmax(lo, dim=-1)
    return text, {
        'final_conf': final_bp[0].max(dim=-1).values.mean().item(),
        'conv_conf': fp.max(dim=-1).values.mean().item(),
        'y_norm': y.norm(dim=-1).mean().item(),
    }


# ---- Scoring ----
def score_text(text, prompt):
    if text.startswith('ERROR'): return -100
    score = 0
    words = text.split()
    score += min(len(words), 15)
    if len(words) > 2:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        score += (len(set(bigrams)) / len(bigrams)) * 10 if bigrams else 10
    if '27 + 58' in prompt and '85' in text: score += 20
    if 'capital of France' in prompt and ('Paris' in text or 'paris' in text): score += 20
    # Penalize pure garbage (all same char etc)
    if len(set(text[:50])) < 5: score -= 20
    return score


PROMPTS = [
    "What is 27 + 58?",
    "The capital of France is",
    "Write a haiku about the ocean.",
]

# ========== Phase 1: Coarse (1 prompt, gen=32, steps=16) ==========
print("\n" + "="*80)
print("Phase 1: Coarse sweep (1 prompt, gen=32, steps=16)")
print("="*80)

snr_mins = [0.01, 1.0, 3.0, 5.0, 7.0, 10.0]
snr_maxs = [40, 50, 80, 100]
noise_scales = [0.0, 0.05, 0.1, 0.3]

coarse = []
total = len(schedules) * len(snr_mins) * len(snr_maxs) * len(noise_scales)
t0 = time.time()

for idx, sched_name in enumerate(schedules):
    for snr_min in snr_mins:
        for snr_max in snr_maxs:
            for ns in noise_scales:
                try:
                    snrs = schedules[sched_name](snr_min, snr_max, 16)
                except:
                    continue
                text, meta = sde_heun_generate(PROMPTS[0], snrs, ns, 32, seed=42)
                s = score_text(text, PROMPTS[0])
                entry = {
                    'schedule': sched_name, 'snr_min': snr_min, 'snr_max': snr_max,
                    'noise_scale': ns, 'text': text[:150], 'score': s, **meta,
                }
                coarse.append(entry)
                n = len(coarse)
                if n % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / n * (total - n)
                    print(f"  [{n}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining | last: {sched_name} min={snr_min} max={snr_max} ns={ns} score={s:.0f}")

coarse.sort(key=lambda x: x['score'], reverse=True)
elapsed = time.time() - t0
print(f"\nPhase 1 done: {len(coarse)} configs in {elapsed:.0f}s")

print("\nTop 15 coarse:")
for i, c in enumerate(coarse[:15]):
    print(f"  #{i+1} score={c['score']:.0f} | {c['schedule']} min={c['snr_min']} max={c['snr_max']} ns={c['noise_scale']} conf={c['final_conf']:.3f} | {c['text'][:100]}")

# ========== Phase 2: Fine (3 prompts, gen=32+128, steps=16/32/64) on top 10 ==========
print("\n" + "="*80)
print("Phase 2: Fine sweep on top configs")
print("="*80)

seen = set()
top_combos = []
for c in coarse[:30]:
    key = (c['schedule'], c['snr_min'], c['snr_max'], c['noise_scale'])
    if key not in seen:
        seen.add(key)
        top_combos.append(key)
    if len(top_combos) >= 10:
        break

fine = []
for combo_idx, (sched, smin, smax, ns) in enumerate(top_combos):
    print(f"\n  Combo {combo_idx+1}/{len(top_combos)}: {sched} min={smin} max={smax} ns={ns}")
    for steps in [16, 32, 64]:
        for gen_len in [32, 128]:
            try:
                snrs = schedules[sched](smin, smax, steps)
            except:
                continue
            total_score = 0
            outputs = []
            for prompt in PROMPTS:
                text, meta = sde_heun_generate(prompt, snrs, ns, gen_len, seed=42)
                s = score_text(text, prompt)
                total_score += s
                outputs.append({'prompt': prompt, 'text': text[:300], 'score': s, **meta})

            entry = {
                'schedule': sched, 'snr_min': smin, 'snr_max': smax,
                'noise_scale': ns, 'steps': steps, 'gen_length': gen_len,
                'total_score': total_score, 'outputs': outputs,
            }
            fine.append(entry)
            if gen_len == 128:
                print(f"    steps={steps} gen={gen_len} score={total_score:.0f}")
                for o in outputs:
                    print(f"      [{o['prompt'][:25]}] conf={o['final_conf']:.3f} | {o['text'][:120]}")

# ========== Final ranking ==========
print("\n" + "="*80)
print("FINAL RANKING (gen=128)")
print("="*80)

gen128 = [r for r in fine if r['gen_length'] == 128]
gen128.sort(key=lambda x: x['total_score'], reverse=True)

for i, cfg in enumerate(gen128[:15]):
    print(f"\n{'='*60}")
    print(f"#{i+1} | score={cfg['total_score']:.1f} | {cfg['schedule']} min={cfg['snr_min']} max={cfg['snr_max']} ns={cfg['noise_scale']} steps={cfg['steps']}")
    for o in cfg['outputs']:
        print(f"  [{o['prompt'][:30]}] conf={o['final_conf']:.3f} conv={o['conv_conf']:.3f} |y|={o['y_norm']:.2f}")
        print(f"    {o['text'][:200]}")

# Save
os.makedirs('results/eval_semantic_b05', exist_ok=True)
all_results = {
    'model': CKPT, 'beta': bv, 'noise_dim': nd,
    'coarse_top50': coarse[:50],
    'fine_results': fine,
    'ranking_gen128': [{**c, 'rank': i+1} for i, c in enumerate(gen128[:15])],
}
with open('results/eval_semantic_b05/sde_param_search.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nSaved to results/eval_semantic_b05/sde_param_search.json")
print(f"Total: coarse={len(coarse)}, fine={len(fine)}")
