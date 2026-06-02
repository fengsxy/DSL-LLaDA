"""SDE step sweep for a SINGLE model. Run multiple instances in parallel on different GPUs.

Usage: CUDA_VISIBLE_DEVICES=X python sde_sweep_single.py --model_key KEY --gpu 0
"""
import argparse, json, os, sys, math, time, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import load_file

TOP_K = 512

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
            total += 1;
            if ws[i] == ws[i-1]: rep += 1
    return rep / max(total, 1)

def sde_generate(model, ne, lb, bw, bb, K, bv, tokenizer, prompt, steps, schedule, ns, device, gen_length=256):
    msgs = [{'role': 'user', 'content': prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)

    snr_min = 0.01
    if schedule == 'sensitive':
        n1, n2 = max(1, int(steps*0.05)), max(1, int(steps*0.90))
        n3 = steps - n1 - n2
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
        return (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0]-1)]).sum(dim=-2)

    for i in range(len(snrs)-1):
        s, s_next = snrs[i], snrs[i+1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s; g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next; g_e = ns / s_next
        y = y + 0.5*(f+f_e)*ds + 0.5*(g+g_e)*dW

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
    parser.add_argument('--model_key', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    mk = args.model_key

    search = json.load(open('eval_results/sde_param_search.json'))
    reg = json.load(open('eval_results/registry.json'))

    best = search[mk]['best']
    bi = float(best['beta_infer'])
    ns = float(best['noise_scale'])
    ckpt = os.path.join(os.path.dirname(os.path.dirname(__file__)), reg[mk]['path'])

    print(f"Model: {mk} | bi={bi} ns={ns} | ckpt={ckpt} | device={device}")

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    ne, lb, bw, bb, K = None, None, None, None, None
    for fname in sorted(os.listdir(ckpt)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt, fname), device=device)
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)  # +mask slot

    print("Loading GPT-2 Large...")
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
    gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2-large')

    results = []
    for steps in [32, 64, 128, 256, 512, 1024]:
        for sch in ['uniform', 'sensitive']:
            print(f"  s={steps:4d} {sch:10s}", end='', flush=True)
            t0 = time.time()
            texts = [sde_generate(model, ne, lb, bw, bb, K, bi, tokenizer, p, steps, sch, ns, device) for p in PROMPTS]
            valid = [t for t in texts if t.strip()]
            ppl = compute_ppl(valid, gpt2, gpt2_tok, device)
            al = float(np.mean([len(t.split()) for t in valid])) if valid else 0
            rr = rep_rate(valid)
            elapsed = time.time() - t0
            print(f" | PPL={ppl:6.1f} len={al:5.0f} rep={rr:.3f} | {elapsed:.0f}s")
            results.append({'steps': steps, 'schedule': sch, 'beta_infer': bi, 'noise_scale': ns,
                           'gen_ppl': round(ppl, 2), 'avg_len': round(al, 1), 'rep_rate': round(rr, 3),
                           'nfe': steps*2, 'time_s': round(elapsed, 1)})

    out = {'model': mk, 'best_params': best, 'step_sweep': results}
    outpath = f'eval_results/sde_sweep_{mk}.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == '__main__':
    main()
