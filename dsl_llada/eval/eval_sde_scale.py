"""Scaled SDE vs Remask evaluation: 100 prompts, multiple NFEs.
Each (model, method, NFE) combo runs as a separate invocation for parallelism.

Usage: CUDA_VISIBLE_DEVICES=X python eval_sde_scale.py --model_key KEY --method METHOD --nfe NFE --gpu 0
Methods: remask_free, remask_noEOS, remask_b32_noEOS, sde
"""
import argparse,json,os,sys,math,time,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'LLaDA'))
from generate import generate
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import load_file

TOP_K = 512
# Optimal ns per NFE (from noise_scale sweep)
NS_BY_NFE = {4: 0.1, 8: 0.1, 16: 0.05, 32: 0.01, 64: 0.005, 128: 0.005}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_key', required=True)
    p.add_argument('--method', required=True, choices=['remask_free','remask_noEOS','remask_b32_noEOS','sde'])
    p.add_argument('--nfe', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--n_prompts', type=int, default=100)
    a = p.parse_args()

    device = f'cuda:{a.gpu}'
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    reg = json.load(open(os.environ.get("DSL_LLADA_REGISTRY", os.path.join(_root, "dsl_llada", "configs", "registry.json"))))
    entry = reg[a.model_key]
    ckpt = os.path.join(_root, entry['path']) if entry.get('type') == 'local' else entry['path']

    # Load prompts
    prompts = json.load(open(os.path.join(_root, 'eval_data/sde_prompts_100.json')))[:a.n_prompts]

    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    # Load DSL weights if SDE
    ne, lb, bw, bb, K = None, None, None, None, None
    if a.method == 'sde':
        for fname in sorted(os.listdir(ckpt)):
            if fname.endswith('.safetensors'):
                st = load_file(os.path.join(ckpt, fname), device=device)
                if 'noise_embed.weight' in st:
                    ne=st['noise_embed.weight'].float(); lb=st['converter.logit_bias'].float()
                    bw=st['converter.backbone_embedding.weight'].float(); bb=st['converter.backbone_embedding.bias'].float()
        if ne is None: print("No DSL weights"); return
        K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)

    # SDE params
    search_path = os.path.join(_root, 'eval_results/sde_param_search.json')
    bi = 2.0
    if os.path.exists(search_path) and a.model_key in json.load(open(search_path)):
        bi = float(json.load(open(search_path))[a.model_key]['best']['beta_infer'])
    ns = NS_BY_NFE.get(a.nfe, 0.01)

    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
    gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2-large')

    def compute_ppl(texts):
        nlls = []
        for t in texts:
            w = ' '.join(t.split()[:128])
            if not w.strip(): continue
            ids = gpt2_tok(w, return_tensors='pt', truncation=True, max_length=512)['input_ids'].to(device)
            if ids.shape[1] < 2: continue
            with torch.no_grad(): nlls.append(gpt2(ids, labels=ids).loss.item())
        return float(np.exp(np.mean(nlls))) if nlls else 999.

    def gen_remask(prompt, steps, block=256, suppress_eos=False):
        msgs = [{'role':'user','content':prompt}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        attn = torch.ones_like(iids)
        with torch.no_grad():
            out = generate(model, iids, attn, steps=steps, gen_length=256, block_length=block,
                          temperature=0., cfg_scale=0., remasking='low_confidence',
                          eos_suppress_ratio=1.0 if suppress_eos else 0.0)
        return tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)

    def gen_sde(prompt, steps):
        msgs = [{'role':'user','content':prompt}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        pl=iids.shape[1]; pe=model.model.transformer.wte(iids).float()
        nd=ne.shape[1]; gl=256; dummy=torch.zeros(1,pl+gl,dtype=torch.long,device=device)
        n1,n2=max(1,int(steps*0.05)),max(1,int(steps*0.90)); n3=steps-n1-n2
        snrs=torch.cat([torch.exp(torch.linspace(math.log(0.01),math.log(7),n1+1)),
            torch.exp(torch.linspace(math.log(7),math.log(74),n2+1))[1:],
            torch.exp(torch.linspace(math.log(74),math.log(100),n3+1))[1:]]).to(device)
        torch.manual_seed(42); y=F.normalize(torch.randn(1,gl,nd,device=device),dim=-1)
        def xhat(yy,ss):
            z=ss*yy;cl=bi*(z.float()@K.T)+lb;p_=F.softmax(cl.float(),dim=-1)
            h=F.linear(p_,bw,bb).to(torch.bfloat16);e=torch.cat([pe.to(torch.bfloat16),h],dim=1)
            with torch.no_grad():lo=model(input_ids=dummy,inputs_embeds=e).logits[:,pl:,:].float()
            bp=F.softmax(lo,dim=-1);tv,ti=bp.topk(min(TOP_K,bp.shape[-1]),dim=-1)
            tv=tv/tv.sum(dim=-1,keepdim=True)
            return(tv.unsqueeze(-1)*ne[ti.clamp_max(ne.shape[0]-1)]).sum(dim=-2)
        for i in range(len(snrs)-1):
            s,sn=snrs[i],snrs[i+1];ds=sn-s;dW=torch.sqrt(ds.abs())*torch.randn_like(y)
            xh=xhat(y,s);f_=(xh-y)/s;g=ns/s;ye=y+f_*ds+g*dW
            xhe=xhat(ye,sn);fe=(xhe-ye)/sn;ge=ns/sn;y=y+0.5*(f_+fe)*ds+0.5*(g+ge)*dW
        zf=snrs[-1]*y;cf=bi*(zf.float()@K.T)+lb;pf=F.softmax(cf.float(),dim=-1)
        hf=F.linear(pf,bw,bb).to(torch.bfloat16);ef=torch.cat([pe.to(torch.bfloat16),hf],dim=1)
        with torch.no_grad():lo=model(input_ids=dummy,inputs_embeds=ef).logits[:,pl:,:].float()
        return tokenizer.decode(lo.argmax(dim=-1)[0],skip_special_tokens=True)

    # Generate
    t0 = time.time()
    texts = []
    from tqdm import tqdm
    for item in tqdm(prompts, desc=f"{a.model_key} {a.method} NFE={a.nfe}"):
        try:
            if a.method == 'sde':
                texts.append(gen_sde(item['prompt'], a.nfe // 2))
            elif a.method == 'remask_free':
                texts.append(gen_remask(item['prompt'], a.nfe))
            elif a.method == 'remask_noEOS':
                texts.append(gen_remask(item['prompt'], a.nfe, suppress_eos=True))
            elif a.method == 'remask_b32_noEOS':
                texts.append(gen_remask(item['prompt'], a.nfe, block=32, suppress_eos=True))
        except Exception as e:
            texts.append("")

    valid = [t for t in texts if t.strip()]
    gen_ppl = compute_ppl(valid)
    avg_len = float(np.mean([len(t.split()) for t in valid])) if valid else 0
    rep = sum(1 for t in valid for i in range(1,len(t.split())) if t.split()[i]==t.split()[i-1])/max(sum(len(t.split())-1 for t in valid),1)
    duration = time.time() - t0

    result = {
        'model': a.model_key, 'method': a.method, 'nfe': a.nfe,
        'gen_ppl': round(gen_ppl, 2), 'avg_len': round(avg_len, 1), 'rep_rate': round(rep, 4),
        'n_generated': len(valid), 'duration_s': round(duration, 1),
        'params': {'bi': bi, 'ns': ns} if a.method == 'sde' else {},
        'generated_texts': valid,
    }

    outpath = os.path.join(_root, f'eval_results/scale_{a.model_key}_{a.method}_nfe{a.nfe}.json')
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{a.model_key} {a.method} NFE={a.nfe}: PPL={gen_ppl:.1f} Len={avg_len:.0f}w Rep={rep:.3f} ({duration:.0f}s)")
    print(f"Saved to {outpath}")

if __name__ == '__main__':
    main()
