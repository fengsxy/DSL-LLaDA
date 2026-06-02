"""Quick follow-up: test steps=128 for the best configs from SDE param search."""
import json, os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))

from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import load_file

MASK_ID = 126336
TOP_K = 512
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

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

# Load GPT-2 for scoring
print("Loading GPT-2 Large...")
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2-large')

def compute_gen_ppl(texts):
    nlls = []
    for text in texts:
        words = text.split()[:128]
        t = ' '.join(words)
        if not t.strip(): continue
        enc = gpt2_tok(t, return_tensors='pt', truncation=True, max_length=512)
        ids = enc['input_ids'].to(device)
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            nlls.append(gpt2(ids, labels=ids).loss.item())
    return float(np.exp(np.mean(nlls))) if nlls else float('inf')

def sde_generate(model, ne, lb, bw, bb, bv, prompt, steps, schedule, ns, gen_length=256):
    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    K = ne
    dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)

    snr_min = 0.01
    if schedule == 'sensitive':
        n1 = max(1, int(steps * 0.05)); n2 = max(1, int(steps * 0.90)); n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1+1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2+1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(100), n3+1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(torch.linspace(math.log(snr_min), math.log(100), steps+1)).to(device)

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
        xh = (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0]-1)]).sum(dim=-2)
        return xh, bp

    for i in range(len(snrs)-1):
        s, s_next = snrs[i], snrs[i+1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh, _ = get_xhat(y, s)
        f = (xh - y) / s; g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e, _ = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next; g_e = ns / s_next
        y = y + 0.5*(f+f_e)*ds + 0.5*(g+g_e)*dW

    # Final decode
    z_final = snrs[-1] * y
    cl_final = bv * (z_final.float() @ K.T) + lb
    probs_final = F.softmax(cl_final.float(), dim=-1)
    h_final = F.linear(probs_final, bw, bb).to(torch.bfloat16)
    embeds_final = torch.cat([pe.to(torch.bfloat16), h_final], dim=1)
    with torch.no_grad():
        lo = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
    toks = lo.argmax(dim=-1)[0]
    return tokenizer.decode(toks, skip_special_tokens=True)

# Wait for param search to finish, then read best configs
search_path = 'eval_results/sde_param_search.json'
if not os.path.exists(search_path):
    print("Waiting for sde_param_search.json...")
    import time
    while not os.path.exists(search_path):
        time.sleep(10)

search = json.load(open(search_path))

# For each model, take best config and test with steps=128
models_to_test = {
    'sm_b2': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
    'b1': 'checkpoints/pertoken_b1_d100_1k/checkpoint-1000',
    'sem_b05_3k': 'checkpoints/semantic_b05_d100_10k/checkpoint-3000',
    'sem_b2': 'checkpoints/semantic_b2_d100_1k/checkpoint-1000',
}

results_128 = {}

for model_key, ckpt in models_to_test.items():
    best = search[model_key]['best']
    bi = best['beta_infer']
    ns = best['noise_scale']

    print(f"\n{'='*60}")
    print(f"Model: {model_key} | best bi={bi}, ns={ns}")
    print(f"{'='*60}")

    # Load model
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    ne, lb, bw, bb, bv = None, None, None, None, 2.0
    for fname in sorted(os.listdir(ckpt)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt, fname), device=device)
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].item()

    if ne is None:
        print(f"  No DSL weights, skipping")
        del model; torch.cuda.empty_cache()
        continue

    model_results = {}
    # Test top-3 beta_infer values from search + the best ns
    top3 = search[model_key].get('top3_beta_infer', [bi])
    if not isinstance(top3, list) or len(top3) == 0:
        top3 = [bi]
    # Also always include 2*beta_train as candidate
    beta_train = best.get('beta_train', bv)
    candidates_bi = sorted(set(top3 + [bi]))
    # For large steps (>=256), only test best bi to save time

    for steps in [32, 64, 128, 256, 512, 1024]:
        bi_list = candidates_bi if steps <= 128 else [bi]
        for test_bi in bi_list:
            for sch in ['uniform', 'sensitive']:
                key = f"s={steps}_bi={test_bi}_sch={sch}"
                print(f"  {key} ns={ns}...", end='', flush=True)
                t0 = time.time()

                texts = []
                for p in PROMPTS:
                    text = sde_generate(model, ne, lb, bw, bb, test_bi, p, steps, sch, ns)
                    texts.append(text)

                valid = [t for t in texts if t.strip()]
                ppl = compute_gen_ppl(valid) if valid else float('inf')
                avg_len = np.mean([len(t.split()) for t in valid]) if valid else 0
                rep = sum(1 for t in valid for i in range(1, len(t.split())) if t.split()[i] == t.split()[i-1]) / max(sum(len(t.split())-1 for t in valid), 1)

                elapsed = time.time() - t0
                print(f" PPL={ppl:.1f} len={avg_len:.0f} rep={rep:.3f} | {elapsed:.0f}s")

                model_results[key] = {
                    'steps': steps, 'schedule': sch, 'beta_infer': test_bi, 'noise_scale': ns,
                    'gen_ppl': round(ppl, 2), 'avg_len': round(avg_len, 1), 'rep_rate': round(rep, 3),
                }

    results_128[model_key] = {
        'best_from_search': best,
        'results_128': model_results,
    }

    del model; torch.cuda.empty_cache()

# Save
with open('eval_results/sde_128step_results.json', 'w') as f:
    json.dump(results_128, f, indent=2, default=str)

print("\n\n=== SUMMARY: steps=128 vs best from search ===")
for mk, r in results_128.items():
    best = r['best_from_search']
    print(f"\n{mk}: best@s={best['steps']} PPL={best['gen_ppl']:.1f}")
    for key, v in r['results_128'].items():
        delta = v['gen_ppl'] - best['gen_ppl']
        print(f"  {key}: PPL={v['gen_ppl']:.1f} ({delta:+.1f}) len={v['avg_len']:.0f} rep={v['rep_rate']:.3f}")

print("\nDone!")
