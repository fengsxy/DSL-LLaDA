#!/bin/bash
# Automated eval pipeline for semantic_b05_d100_1k
# 1. Wait for training to finish
# 2. Run all evals in parallel
# 3. If results look good, continue training to 5K steps
set -e
source /home/ubuntu/efs/RMDM/.venv/bin/activate

CKPT="./checkpoints/semantic_b05_d100_1k/checkpoint-1000"
ORIGINAL="GSAI-ML/LLaDA-8B-Instruct"
RESULTS_DIR="./results/eval_semantic_b05"
mkdir -p "${RESULTS_DIR}" logs

echo "=== Eval Pipeline for semantic_b05 ==="
echo "Checkpoint: ${CKPT}"

# ---- Wait for training to complete ----
echo "[$(date)] Waiting for training to finish..."
while true; do
    if ! pgrep -f "llada_cpt_dsl.py" > /dev/null 2>&1; then
        echo "[$(date)] Training processes gone"
        if [ -d "${CKPT}" ]; then
            echo "[$(date)] Checkpoint exists, proceeding"
            break
        fi
    fi
    LAST=$(grep "'loss'" logs/train_semantic_b05.log 2>/dev/null | tail -1 | grep -oP '\d+/1000' | head -1 || echo "?")
    echo "[$(date)] Training in progress... step ${LAST}"
    sleep 60
done

sleep 30  # wait for GPU memory to free
echo ""
echo "============================================"
echo "=== Starting evaluations at $(date) ==="
echo "============================================"

# ---- Phase 1: GSM8K (200 questions) + Corruption + SDE (parallel) ----
echo ""
echo "=== Phase 1: GSM8K + Corruption + SDE ==="

# GSM8K 200 questions, semantic model (GPU 0)
echo "[$(date)] GSM8K 200q on semantic_b05..."
CUDA_VISIBLE_DEVICES=0 python -u dsl_llada/eval_reasoning.py \
    --checkpoint "${CKPT}" \
    --method standard --steps 64 --dataset gsm8k \
    --gpu 0 --gen_length 512 --n_samples 200 \
    --output "${RESULTS_DIR}/gsm8k_200q.json" \
    > logs/eval_b05_gsm8k.log 2>&1 &
PID_GSM=$!

# GSM8K 200 questions, original (GPU 1)
echo "[$(date)] GSM8K 200q on original..."
CUDA_VISIBLE_DEVICES=1 python -u dsl_llada/eval_reasoning.py \
    --checkpoint "${ORIGINAL}" \
    --method standard --steps 64 --dataset gsm8k \
    --gpu 0 --gen_length 512 --n_samples 200 \
    --output "${RESULTS_DIR}/gsm8k_200q_original.json" \
    > logs/eval_b05_gsm8k_original.log 2>&1 &
PID_GSM_ORIG=$!

# Corruption probe (GPU 2)
echo "[$(date)] Corruption probe..."
CUDA_VISIBLE_DEVICES=2 python -u dsl_llada/test_corruption_probe.py \
    --checkpoint "${CKPT}" --gpu 0 \
    --output "${RESULTS_DIR}/corruption.json" \
    > logs/eval_b05_corruption.log 2>&1 &
PID_CORR=$!

# SDE generation test (GPU 3) — using best config [10,80] s=64 ns=0.05
echo "[$(date)] SDE generation test..."
CUDA_VISIBLE_DEVICES=3 python -u -c "
import torch, torch.nn.functional as F, os, sys, math, json
sys.path.insert(0, '.')
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336; device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
ckpt = '${CKPT}'
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
ne, lb, bw, bb, bv = None, None, None, None, 0.5
for fn in sorted(os.listdir(ckpt)):
    if fn.endswith('.safetensors'):
        st = load_file(os.path.join(ckpt, fn), device=device)
        if 'noise_embed.weight' in st:
            ne=st['noise_embed.weight'].float(); lb=st['converter.logit_bias'].float()
            bw=st['converter.backbone_embedding.weight'].float(); bb=st['converter.backbone_embedding.bias'].float()
            bv=st['converter.beta'].item(); break
K = torch.cat([ne, torch.zeros(1,ne.shape[1],device=device)],dim=0)
V=ne.shape[0]; nd=ne.shape[1]

def sde_heun(prompt, bi, gen=128, steps=64, snr_min=10.0, snr_max=80.0, ns=0.05, seed=42):
    msgs=[{'role':'user','content':prompt}]
    fmt=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    iids=tokenizer(fmt,return_tensors='pt',add_special_tokens=False)['input_ids'].to(device)
    pl=iids.shape[1]; pe=model.model.transformer.wte(iids).float()
    dummy=torch.zeros(1,pl+gen,dtype=torch.long,device=device)
    snrs=torch.exp(torch.linspace(math.log(snr_min),math.log(snr_max),steps+1)).to(device)
    torch.manual_seed(seed)
    y=F.normalize(torch.randn(1,gen,nd,device=device),dim=-1)
    def xhat(yy,ss):
        z=ss*yy; cl=bi*(z.float()@K.T)+lb; p=F.softmax(cl.float(),dim=-1)
        h=F.linear(p,bw,bb).to(torch.bfloat16)
        e=torch.cat([pe.to(torch.bfloat16),h],dim=1)
        with torch.no_grad(): lo=model(input_ids=dummy,inputs_embeds=e).logits[:,pl:,:].float()
        bp=F.softmax(lo,dim=-1); sp,si=bp.sort(dim=-1,descending=True)
        cm=sp.cumsum(dim=-1); m=cm-sp>0.9; sp[m]=0; sp=sp/sp.sum(dim=-1,keepdim=True)
        f=torch.zeros_like(bp); f.scatter_(-1,si,sp)
        return f[:,:,:V]@ne
    for i in range(steps):
        s=snrs[i]; sn=snrs[i+1]; ds=sn-s
        dW=torch.sqrt(ds.abs())*torch.randn_like(y)
        xh=xhat(y,s); f=(xh-y)/s; g=ns/s; ye=y+f*ds+g*dW
        xe=xhat(ye,sn); fe=(xe-ye)/sn; ge=ns/sn
        y=y+0.5*(f+fe)*ds+0.5*(g+ge)*dW
    zf=snrs[-1]*y; clf=bi*(zf.float()@K.T)+lb
    fp=F.softmax(clf.float(),dim=-1); hf=F.linear(fp,bw,bb).to(torch.bfloat16)
    ef=torch.cat([pe.to(torch.bfloat16),hf],dim=1)
    with torch.no_grad(): lo=model(input_ids=dummy,inputs_embeds=ef).logits[:,pl:,:].float()
    return tokenizer.decode(lo.argmax(-1)[0],skip_special_tokens=True)

prompts = [
    'What is 27 + 58?',
    'The capital of France is',
    'A store had 45 apples. It sold 18, then got 27 more. How many does it have now?',
    'Write a short story about a robot discovering music.',
    'Explain gravity in simple terms.',
]
results = []
for p in prompts:
    for gen in [32, 128]:
        out = sde_heun(p, bi=bv, gen=gen)
        results.append({'prompt': p, 'gen': gen, 'output': out[:300]})
        print(f'gen={gen:3d} | {p[:40]:40s} | {out[:100]}')

with open('${RESULTS_DIR}/sde_generation.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('SDE results saved.')
" > logs/eval_b05_sde.log 2>&1 &
PID_SDE=$!

# Semantic correction test (GPU 4)
echo "[$(date)] Semantic correction test..."
CUDA_VISIBLE_DEVICES=4,5 python -u dsl_llada/test_semantic_correction.py \
    > logs/eval_b05_semantic_correction.log 2>&1 &
PID_SEMCORR=$!

echo "[$(date)] Waiting for Phase 1 (5 parallel jobs)..."
for pid_name in PID_GSM PID_GSM_ORIG PID_CORR PID_SDE PID_SEMCORR; do
    pid=${!pid_name}
    wait $pid 2>/dev/null && echo "  ${pid_name} done (exit 0)" || echo "  ${pid_name} failed (exit $?)"
done

echo ""
echo "============================================"
echo "=== Phase 1 complete at $(date) ==="
echo "============================================"

# ---- Print Results Summary ----
echo ""
echo "=== RESULTS SUMMARY ==="

# GSM8K
echo ""
echo "--- GSM8K 200q ---"
for tag in "200q" "200q_original"; do
    f="${RESULTS_DIR}/gsm8k_${tag}.json"
    if [ -f "$f" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d.get(\"accuracy\",0)*100:.1f}%')" 2>/dev/null || echo "?")
        echo "  ${tag}: ${acc}"
    fi
done

# Corruption
echo ""
echo "--- Corruption ---"
if [ -f "${RESULTS_DIR}/corruption.json" ]; then
    python3 -c "
import json
d = json.load(open('${RESULTS_DIR}/corruption.json'))
for key in ['corrupt', 'mask']:
    if key in d:
        for rate, vals in sorted(d[key].items()):
            if isinstance(vals, dict):
                acc_key = 'corrupted_acc' if key == 'corrupt' else 'mask_acc'
                acc = vals.get(acc_key, '?')
                if isinstance(acc, float): acc = f'{acc:.1%}'
                print(f'  {key} {rate}: {acc}')
" 2>/dev/null || echo "  parse error"
fi

# SDE
echo ""
echo "--- SDE Generation ---"
if [ -f "${RESULTS_DIR}/sde_generation.json" ]; then
    python3 -c "
import json
for r in json.load(open('${RESULTS_DIR}/sde_generation.json')):
    print(f\"  gen={r['gen']:3d} | {r['prompt'][:35]:35s} | {r['output'][:80]}\")
" 2>/dev/null || echo "  parse error"
fi

# Semantic correction
echo ""
echo "--- Semantic Correction ---"
tail -5 logs/eval_b05_semantic_correction.log 2>/dev/null || echo "  not available"

echo ""
echo "=== All results saved to ${RESULTS_DIR}/ ==="
echo "=== Eval complete at $(date) ==="

# ---- Phase 2: Decision on continue training ----
echo ""
echo "=== Check if continue training to 5K is warranted ==="
GSM_ACC=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/gsm8k_200q.json')); print(d.get('accuracy',0))" 2>/dev/null || echo "0")
echo "  GSM8K accuracy: ${GSM_ACC}"
echo "  If > 0.30, recommend continuing to 5K steps"
echo "  Decision left for user review."
