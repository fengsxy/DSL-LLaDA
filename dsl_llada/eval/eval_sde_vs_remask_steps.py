"""Compare SDE vs Remask at matched NFE across step counts.
Tests: NFE = 8, 16, 32, 64, 128 (SDE uses Heun so steps = NFE/2)

Usage: CUDA_VISIBLE_DEVICES=X python eval_sde_vs_remask_steps.py --model_key KEY --gpu 0
"""
import argparse, json, os, sys, math, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'LLaDA'))
from generate import generate
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
            total += 1
            if ws[i] == ws[i-1]: rep += 1
    return rep / max(total, 1)

def remask_gen(model, tokenizer, prompt, steps, device, gen_length=int(os.environ.get("GEN_LENGTH","256")),
               block_length=None, suppress_eos=False):
    msgs = [{'role': 'user', 'content': prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    attn = torch.ones_like(iids)
    bl = block_length if block_length else gen_length
    with torch.no_grad():
        out = generate(model, iids, attn, steps=steps, gen_length=gen_length,
                       block_length=bl, temperature=0., cfg_scale=0.,
                       remasking='low_confidence',
                       eos_suppress_ratio=1.0 if suppress_eos else 0.0)
    return tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)

def sde_gen(model, ne, lb, bw, bb, K, bv, tokenizer, prompt, steps, snr_s, snr_e, ns, sch, device, gen_length=int(os.environ.get("GEN_LENGTH","256"))):
    msgs = [{'role': 'user', 'content': prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]; pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]; dummy = torch.zeros(1, pl+gen_length, dtype=torch.long, device=device)
    if sch == 'sensitive':
        n1, n2 = max(1, int(steps*0.05)), max(1, int(steps*0.90)); n3 = steps-n1-n2
        snrs = torch.cat([torch.exp(torch.linspace(math.log(max(snr_s,0.01)), math.log(7), n1+1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2+1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_e), n3+1))[1:]]).to(device)
    else:
        snrs = torch.exp(torch.linspace(math.log(max(snr_s,0.01)), math.log(snr_e), steps+1)).to(device)
    torch.manual_seed(42)
    y = F.normalize(torch.randn(1, gen_length, nd, device=device), dim=-1)
    def xhat(yy, ss):
        z=ss*yy; cl=bv*(z.float()@K.T)+lb; p=F.softmax(cl.float(), dim=-1)
        h=F.linear(p, bw, bb).to(torch.bfloat16)
        e=torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad(): lo=model(input_ids=dummy, inputs_embeds=e).logits[:,pl:,:].float()
        bp=F.softmax(lo, dim=-1); tv,ti=bp.topk(min(TOP_K, bp.shape[-1]), dim=-1)
        tv=tv/tv.sum(dim=-1, keepdim=True)
        return (tv.unsqueeze(-1)*ne[ti.clamp_max(ne.shape[0]-1)]).sum(dim=-2)
    for i in range(len(snrs)-1):
        s,sn=snrs[i],snrs[i+1]; ds=sn-s; dW=torch.sqrt(ds.abs())*torch.randn_like(y)
        xh=xhat(y,s); f=(xh-y)/s; g=ns/s; ye=y+f*ds+g*dW
        xhe=xhat(ye,sn); fe=(xhe-ye)/sn; ge=ns/sn
        y=y+0.5*(f+fe)*ds+0.5*(g+ge)*dW
    zf=snrs[-1]*y; cf=bv*(zf.float()@K.T)+lb; pf=F.softmax(cf.float(), dim=-1)
    hf=F.linear(pf, bw, bb).to(torch.bfloat16)
    ef=torch.cat([pe.to(torch.bfloat16), hf], dim=1)
    with torch.no_grad(): lo=model(input_ids=dummy, inputs_embeds=ef).logits[:,pl:,:].float()
    return tokenizer.decode(lo.argmax(dim=-1)[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    mk = args.model_key

    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reg = json.load(open(os.environ.get("DSL_LLADA_REGISTRY", os.path.join(_root, "dsl_llada", "configs", "registry.json"))))
    search = json.load(open(os.path.join(_root, 'eval_results/sde_param_search.json')))

    entry = reg[mk]
    ckpt = os.path.join(_root, entry['path']) if entry.get('type') == 'local' else entry['path']

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    # Load DSL weights if available
    ne, lb, bw, bb, K = None, None, None, None, None
    has_dsl = False
    if entry.get('dsl') and entry.get('type') == 'local':
        for fname in sorted(os.listdir(ckpt)):
            if fname.endswith('.safetensors'):
                st = load_file(os.path.join(ckpt, fname), device=device)
                if 'noise_embed.weight' in st:
                    ne = st['noise_embed.weight'].float()
                    lb = st['converter.logit_bias'].float()
                    bw = st['converter.backbone_embedding.weight'].float()
                    bb = st['converter.backbone_embedding.bias'].float()
                    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
                    has_dsl = True

    # SDE params
    bi, ns, sch, snr_s = 2.0, 0.01, 'sensitive', 0.01
    if mk in search:
        best = search[mk]['best']
        bi = float(best['beta_infer'])
        ns = float(best['noise_scale'])
        sch = best.get('schedule', 'sensitive')
        snr_s = float(best.get('snr_start', 0.01))

    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
    gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2-large')

    print(f"Model: {mk} | DSL={has_dsl} | bi={bi} ns={ns} sch={sch}")
    print(f"{'NFE':>5} {'Method':<12} {'PPL':>6} {'Len':>6} {'Rep':>6} {'OK':>4}")
    print("-" * 45)

    results = []
    # NFE = 8, 16, 32, 64, 128
    for nfe in [8, 16, 32, 64, 128]:
        # 3 remask methods
        remask_configs = [
            ('remask_free', None, False),
            ('remask_noEOS', None, True),
            ('remask_b32_noEOS', 32, True),
        ]
        for method_name, bl, suppress in remask_configs:
            texts = [remask_gen(model, tokenizer, p, steps=nfe, device=device,
                               block_length=bl, suppress_eos=suppress) for p in PROMPTS]
            valid = [t for t in texts if t.strip()]
            ppl = compute_ppl(valid, gpt2, gpt2_tok, device)
            al = float(np.mean([len(t.split()) for t in valid])) if valid else 0
            rr = rep_rate(valid)
            n_ok = sum(1 for t in texts if len(t.split())>=20 and rep_rate([t])<0.2)
            print(f"{nfe:>5} {method_name:<18} {ppl:>5.1f} {al:>5.0f}w {rr:>5.3f} {n_ok:>3}/10")
            results.append({'nfe': nfe, 'method': method_name, 'ppl': round(ppl,2), 'len': round(al,1),
                           'rep': round(rr,3), 'ok': n_ok})

        # SDE: steps = NFE/2 (Heun)
        if has_dsl and nfe >= 8:
            sde_steps = nfe // 2
            texts = [sde_gen(model, ne, lb, bw, bb, K, bi, tokenizer, p, sde_steps, snr_s, 100, ns, sch, device) for p in PROMPTS]
            valid = [t for t in texts if t.strip()]
            ppl = compute_ppl(valid, gpt2, gpt2_tok, device)
            al = float(np.mean([len(t.split()) for t in valid])) if valid else 0
            rr = rep_rate(valid)
            n_ok = sum(1 for t in texts if len(t.split())>=20 and rep_rate([t])<0.2)
            print(f"{nfe:>5} {'SDE':<18} {ppl:>5.1f} {al:>5.0f}w {rr:>5.3f} {n_ok:>3}/10")
            results.append({'nfe': nfe, 'method': 'sde', 'ppl': round(ppl,2), 'len': round(al,1),
                           'rep': round(rr,3), 'ok': n_ok})

    out = {'model': mk, 'has_dsl': has_dsl, 'sde_params': {'bi': bi, 'ns': ns, 'sch': sch}, 'results': results}
    outpath = os.path.join(_root, f'eval_results/sde_vs_remask_{mk}.json')
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == '__main__':
    main()
