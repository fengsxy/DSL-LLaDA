"""Evaluate SDE for a single checkpoint. Usage: CUDA_VISIBLE_DEVICES=X python sde_eval_single_ckpt.py --ckpt PATH --bi 2.0 --ns 0.01 --sch sensitive --snr_start 0.01 --label 'b1@1K'"""
import argparse,json,os,math,torch,torch.nn.functional as F,numpy as np
from transformers import AutoModel,AutoTokenizer,GPT2LMHeadModel,GPT2TokenizerFast
from safetensors.torch import load_file

TOP_K=512
PROMPTS=[
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

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True)
    p.add_argument('--bi',type=float,default=2.0)
    p.add_argument('--ns',type=float,default=0.01)
    p.add_argument('--sch',default='sensitive')
    p.add_argument('--snr_start',type=float,default=0.01)
    p.add_argument('--snr_end',type=float,default=100.0)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--label',default='model')
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()

    device=f'cuda:{a.gpu}'
    tokenizer=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct',trust_remote_code=True)
    model=AutoModel.from_pretrained(a.ckpt,trust_remote_code=True,torch_dtype=torch.bfloat16).to(device).eval()
    ne,lb,bw,bb=None,None,None,None
    for f in sorted(os.listdir(a.ckpt)):
        if f.endswith('.safetensors'):
            st=load_file(os.path.join(a.ckpt,f),device=device)
            if 'noise_embed.weight' in st:
                ne=st['noise_embed.weight'].float();lb=st['converter.logit_bias'].float()
                bw=st['converter.backbone_embedding.weight'].float();bb=st['converter.backbone_embedding.bias'].float()
    if ne is None: print(f"No DSL weights in {a.ckpt}");return
    K=torch.cat([ne,torch.zeros(1,ne.shape[1],device=device)],dim=0)

    gpt2=GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
    gpt2_tok=GPT2TokenizerFast.from_pretrained('gpt2-large')

    def ppl(texts):
        nlls=[]
        for t in texts:
            w=' '.join(t.split()[:128])
            if not w.strip():continue
            ids=gpt2_tok(w,return_tensors='pt',truncation=True,max_length=512)['input_ids'].to(device)
            if ids.shape[1]<2:continue
            with torch.no_grad():nlls.append(gpt2(ids,labels=ids).loss.item())
        return float(np.exp(np.mean(nlls))) if nlls else 999.

    def gen(prompt):
        msgs=[{'role':'user','content':prompt}]
        fmt=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        iids=tokenizer(fmt,return_tensors='pt',add_special_tokens=False)['input_ids'].to(device)
        pl=iids.shape[1];pe=model.model.transformer.wte(iids).float()
        nd=ne.shape[1];gl=256;dummy=torch.zeros(1,pl+gl,dtype=torch.long,device=device)
        if a.sch=='sensitive':
            n1,n2=max(1,int(a.steps*0.05)),max(1,int(a.steps*0.90));n3=a.steps-n1-n2
            snrs=torch.cat([torch.exp(torch.linspace(math.log(max(a.snr_start,0.01)),math.log(7),n1+1)),
                torch.exp(torch.linspace(math.log(7),math.log(74),n2+1))[1:],
                torch.exp(torch.linspace(math.log(74),math.log(a.snr_end),n3+1))[1:]]).to(device)
        else:
            snrs=torch.exp(torch.linspace(math.log(max(a.snr_start,0.01)),math.log(a.snr_end),a.steps+1)).to(device)
        torch.manual_seed(42);y=F.normalize(torch.randn(1,gl,nd,device=device),dim=-1)
        def xhat(yy,ss):
            z=ss*yy;cl=a.bi*(z.float()@K.T)+lb;p_=F.softmax(cl.float(),dim=-1)
            h=F.linear(p_,bw,bb).to(torch.bfloat16);e=torch.cat([pe.to(torch.bfloat16),h],dim=1)
            with torch.no_grad():lo=model(input_ids=dummy,inputs_embeds=e).logits[:,pl:,:].float()
            bp=F.softmax(lo,dim=-1);tv,ti=bp.topk(min(TOP_K,bp.shape[-1]),dim=-1)
            tv=tv/tv.sum(dim=-1,keepdim=True)
            return(tv.unsqueeze(-1)*ne[ti.clamp_max(ne.shape[0]-1)]).sum(dim=-2)
        for i in range(len(snrs)-1):
            s,sn=snrs[i],snrs[i+1];ds=sn-s;dW=torch.sqrt(ds.abs())*torch.randn_like(y)
            xh=xhat(y,s);f_=(xh-y)/s;g=a.ns/s;ye=y+f_*ds+g*dW
            xhe=xhat(ye,sn);fe=(xhe-ye)/sn;ge=a.ns/sn;y=y+0.5*(f_+fe)*ds+0.5*(g+ge)*dW
        zf=snrs[-1]*y;cf=a.bi*(zf.float()@K.T)+lb;pf=F.softmax(cf.float(),dim=-1)
        hf=F.linear(pf,bw,bb).to(torch.bfloat16);ef=torch.cat([pe.to(torch.bfloat16),hf],dim=1)
        with torch.no_grad():lo=model(input_ids=dummy,inputs_embeds=ef).logits[:,pl:,:].float()
        return tokenizer.decode(lo.argmax(dim=-1)[0],skip_special_tokens=True)

    texts=[gen(p) for p in PROMPTS]
    valid=[t for t in texts if t.strip()]
    genppl=ppl(valid)
    avg_len=float(np.mean([len(t.split()) for t in valid]))
    rep=sum(1 for t in valid for i in range(1,len(t.split())) if t.split()[i]==t.split()[i-1])/max(sum(len(t.split())-1 for t in valid),1)
    n_ok=sum(1 for t in texts if len(t.split())>=30 and sum(1 for j in range(1,len(t.split())) if t.split()[j]==t.split()[j-1])/max(len(t.split())-1,1)<0.1)

    print(f"{a.label:<20} PPL={genppl:>6.1f} Len={avg_len:>5.0f}w Rep={rep:>5.3f} OK={n_ok}/10")

    out={'label':a.label,'ppl':round(genppl,2),'len':round(avg_len,1),'rep':round(rep,3),'ok':n_ok,
         'params':{'bi':a.bi,'ns':a.ns,'sch':a.sch,'snr_start':a.snr_start,'steps':a.steps},
         'texts':texts}
    safe_label=a.label.replace(' ','_').replace('/','_')
    with open(f'eval_results/sde_single_{safe_label}.json','w') as f:
        json.dump(out,f,indent=2,ensure_ascii=False)

if __name__=='__main__':main()
