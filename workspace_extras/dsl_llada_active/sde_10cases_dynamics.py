"""SDE dynamics for all 10 app.py preset cases.
Records per-step token changes, crystallization, and final text.
Uses app.py default params: Beta1, steps=32, gen=256, ns=0.05, snr_max=80, bi=3.0.

Usage: CUDA_VISIBLE_DEVICES=0 python sde_10cases_dynamics.py
"""
import torch, torch.nn.functional as F
import sys, os, math, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

ckpt = 'checkpoints/beta1_d100_1k/checkpoint-1000'
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

for fn in sorted(os.listdir(ckpt)):
    if fn.endswith('.safetensors'):
        st = load_file(os.path.join(ckpt, fn), device=device)
        if 'noise_embed.weight' in st:
            ne = st['noise_embed.weight'].float()
            lb = st['converter.logit_bias'].float()
            bw = st['converter.backbone_embedding.weight'].float()
            bb = st['converter.backbone_embedding.bias'].float()
            bv = st['converter.beta'].item()
            break
K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
nd = ne.shape[1]

CASES = {
    "1. Arithmetic": "Question: What is 27 + 58? Show the reasoning briefly, then give the final answer.",
    "2. Word Math": "A store had 45 apples. It sold 18, then got 27 more. How many apples does it have now?",
    "3. GSM8K Style": "Lena reads 12 pages on Monday, 15 on Tuesday, and twice as many on Wednesday as Monday. How many pages did she read in total?",
    "4. Exact Digits": "Repeat this exactly at the end of your answer: 2027-04-19. First explain why exact copying can be hard.",
    "5. Logic": "There are three boxes: red, blue, green. Exactly one contains a coin. Red is empty. If blue has the coin, green is empty. Where can the coin be?",
    "6. Trip Plan": "Plan a 3-day Tokyo trip for a first-time visitor. Day 1, Day 2, Day 3. Keep each day to 3 items.",
    "7. Creative Story": "Write a short story about a robot hearing rain for the first time. Make it vivid but not cheesy.",
    "8. Style Control": "Write a paragraph in the style of a calm scientific field note describing a lighthouse at dusk.",
    "9. Correction": "The boye walkd to the markit and bought 3 appls.",
    "10. Format Constraint": "Output exactly 5 bullet points. Each bullet must contain exactly 6 words.",
}

STEPS = 32
GEN = 256
BI = 3.0  # app.py default beta_infer
NS = 0.05
SNR_MAX = 80.0
SEED = 42

# Sensitive schedule
snr_min = 3.0
n1 = max(1, int(STEPS * 0.05)); n2 = max(1, int(STEPS * 0.90)); n3 = STEPS - n1 - n2
snrs = torch.cat([
    torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
    torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
    torch.exp(torch.linspace(math.log(74), math.log(SNR_MAX), n3 + 1))[1:],
]).to(device)

all_results = {}

for case_name, prompt in CASES.items():
    t0 = time.time()
    msgs = [{'role': 'user', 'content': prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)

    torch.manual_seed(SEED)
    y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)

    def get_xhat_tokens(yy, ss):
        z = ss * yy
        cl = BI * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        bp = F.softmax(logits, dim=-1)
        tokens = bp[0].argmax(dim=-1)
        conf = bp[0].max(dim=-1).values.mean().item()
        conv_conf = probs[0, :, :-1].max(dim=-1).values.mean().item()
        xh = bp[:, :, :ne.shape[0]] @ ne
        return xh, tokens, conf, conv_conf

    step_data = []
    for i in range(len(snrs) - 1):
        sc, sn = snrs[i], snrs[i + 1]
        ds = sn - sc
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh, tokens, conf, cc = get_xhat_tokens(y, sc)
        step_data.append({'snr': sc.item(), 'tokens': tokens.cpu(), 'conf': conf, 'conv_conf': cc})
        f = (xh - y) / sc; g = NS / sc
        ye = y + f * ds + g * dW
        xhe, _, _, _ = get_xhat_tokens(ye, sn)
        fe = (xhe - ye) / sn; ge = NS / sn
        y = y + 0.5 * (f + fe) * ds + 0.5 * (g + ge) * dW

    # Final
    xh, tokens, conf, cc = get_xhat_tokens(y, snrs[-1])
    step_data.append({'snr': snrs[-1].item(), 'tokens': tokens.cpu(), 'conf': conf, 'conv_conf': cc})

    # Analyze
    n_steps = len(step_data)
    changes = []
    for s in range(1, n_steps):
        c = (step_data[s]['tokens'] != step_data[s-1]['tokens']).sum().item()
        changes.append(c)

    # Crystallization: step after which no more changes
    crystal_step = n_steps - 1
    for s in range(n_steps - 1, 0, -1):
        if changes[s-1] > 0:
            crystal_step = s
            break

    final_text = tokenizer.decode(step_data[-1]['tokens'], skip_special_tokens=True)

    # Peak volatility
    peak_step = max(range(len(changes)), key=lambda i: changes[i])
    peak_changes = changes[peak_step]

    # Task type classification
    total_changes_last5 = sum(changes[-5:])
    crystallized = total_changes_last5 == 0

    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{case_name} ({dt:.1f}s)")
    print(f"{'='*60}")
    print(f"Crystallized: {'YES' if crystallized else 'NO (still changing)'}")
    print(f"Crystal step: {crystal_step}/{n_steps} (SNR={step_data[crystal_step]['snr']:.1f})")
    print(f"Peak volatility: step {peak_step+1}, {peak_changes}/{GEN} tokens changed (SNR={step_data[peak_step+1]['snr']:.1f})")
    print(f"Final conv_conf: {step_data[-1]['conv_conf']:.3f}")
    print(f"Final backbone_conf: {step_data[-1]['conf']:.3f}")
    print(f"Last 5 steps changes: {changes[-5:]}")
    print(f"Text: {final_text[:200]}")

    # Show key snapshots
    for s_idx in [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]:
        text = tokenizer.decode(step_data[s_idx]['tokens'][:40], skip_special_tokens=True)
        print(f"  step {s_idx:2d} SNR={step_data[s_idx]['snr']:6.1f} cc={step_data[s_idx]['conv_conf']:.3f}: {text[:80]}")

    all_results[case_name] = {
        'crystallized': crystallized,
        'crystal_step': crystal_step,
        'crystal_snr': step_data[crystal_step]['snr'],
        'peak_step': peak_step + 1,
        'peak_changes': peak_changes,
        'final_conf': step_data[-1]['conf'],
        'final_conv_conf': step_data[-1]['conv_conf'],
        'last5_changes': changes[-5:],
        'text': final_text[:300],
    }

# Summary table
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"{'Case':<20s} {'Crystal?':>8s} {'CrystalSNR':>10s} {'PeakΔ':>6s} {'Last5Δ':>8s} {'ConvConf':>8s} {'Quality':>8s}")
print("-" * 80)
for name, r in all_results.items():
    quality = "✓" if r['crystallized'] and r['final_conv_conf'] > 0.3 else "△" if r['crystallized'] else "✗"
    last5 = sum(r['last5_changes'])
    print(f"{name:<20s} {'YES' if r['crystallized'] else 'NO':>8s} {r['crystal_snr']:>10.1f} {r['peak_changes']:>6d} {last5:>8d} {r['final_conv_conf']:>8.3f} {quality:>8s}")

os.makedirs('results', exist_ok=True)
with open('results/sde_10cases_dynamics.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to results/sde_10cases_dynamics.json")
