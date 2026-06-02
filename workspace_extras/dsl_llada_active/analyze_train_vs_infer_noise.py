"""分析推理噪声 vs 训练噪声：DSL converter 的训练噪声能否覆盖推理时的错误模式？

Phase 1: 推理错误分析
  - 跑 standard remasking，在每步记录 committed tokens vs gold tokens
  - 分类错误类型：语义相近替换 / 高频词替换 / 完全随机

Phase 2: 训练噪声分析
  - 在不同 SNR 下，converter 输出的 "错误 token" 分类
  - 跟推理错误对比

Usage: CUDA_VISIBLE_DEVICES=0 python analyze_train_vs_infer_noise.py
"""
import torch, torch.nn.functional as F
import sys, os, json, re, math
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
SEED = 42
device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

# Load SM-b2 model
ckpt = 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000'
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
wte = model.model.transformer.wte.weight.float()  # (V, 4096)

# Load DSL weights
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

# Precompute wte norms for cosine similarity
wte_normed = F.normalize(wte, dim=-1)


def classify_error(gold_id, pred_id):
    """分类一个错误：gold_id 是正确 token，pred_id 是预测/替换的 token。
    返回: 'correct', 'semantic_similar', 'random'
    """
    if pred_id == gold_id:
        return 'correct'
    # Cosine similarity in wte space
    cos = F.cosine_similarity(wte[gold_id].unsqueeze(0), wte[pred_id].unsqueeze(0)).item()
    if cos > 0.5:
        return 'semantic_similar'
    else:
        return 'random'


# ============================================================
# Phase 1: 推理错误分析
# ============================================================
print("=" * 60)
print("Phase 1: 推理时错误模式分析")
print("=" * 60)

# 用 10 个 GSM8K 问题，跟踪每步 commit 的 token
ds = __import__('datasets').load_dataset('gsm8k', 'main', split='test')
GEN = 256
STEPS = 64
N_PROBLEMS = 10

infer_errors = defaultdict(list)  # step → list of (gold_id, pred_id, error_type)
infer_error_cos = []  # 所有推理错误的 cosine similarity

for pi in range(N_PROBLEMS):
    row = ds[pi]
    gold_m = re.search(r'####\s*([\-\d,\.]+)', row['answer'])
    gold_ans = gold_m.group(1).replace(',', '').strip() if gold_m else ''
    prompt = f"Question: {row['question']}\nAnswer: Let's think step by step.\n"
    iids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]

    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=device)
    x[0, :pl] = iids[0]
    base = GEN // STEPS; rem = GEN % STEPS
    nt = [base] * STEPS
    for i in range(rem): nt[i] += 1

    torch.manual_seed(SEED)
    prev_committed = {}  # pos → token_id (已 commit 的)

    with torch.no_grad():
        for s in range(STEPS):
            mi = (x == MASK_ID)
            logits = model(x).logits
            x0 = logits.argmax(dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[~mi] = -float('inf')
            _, idx = x0_p.sort(dim=-1, descending=True)

            # 这一步要 commit 哪些位置
            commit_positions = idx[0, :nt[s]].tolist()
            for pos in commit_positions:
                if pos >= pl:  # 只看生成区域
                    pred_id = x0[0, pos].item()
                    prev_committed[pos] = pred_id

            x[0, idx[0, :nt[s]]] = x0[0, idx[0, :nt[s]]]

    # 最终生成的 text
    final_text = tokenizer.decode(x[0, pl:pl+GEN], skip_special_tokens=True)

    # 现在看：最终 commit 的 token 在后续步骤有没有被改变？
    # remasking 不会改已 commit 的 token，但我们可以看模型 argmax 是否跟 commit 一致
    # 实际上，committed token 不会被改。推理错误 = committed token 本身就是错的。
    # 我们没有 ground truth（生成任务）。但可以看：
    # 在 partial generation 过程中，新 commit 的 token 跟 context 是否一致。

    # 改用另一个方法：拿 clean text（eval_texts），mask 30%，看模型预测错误
    pass

# 用 eval_texts 做推理错误分析（有 ground truth）
print("\n用 eval_texts 分析推理错误模式...")
texts = json.loads(open('results/paper/eval_texts.json').read())[:50]

infer_error_types = Counter()
infer_cos_sims = []

for ti, text in enumerate(texts):
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    L = ids.shape[1]
    if L < 20: continue

    # Mask 30%
    torch.manual_seed(SEED + ti)
    n_mask = max(1, int(L * 0.3))
    mask_pos = torch.randperm(L)[:n_mask]

    x = ids.clone()
    x[0, mask_pos] = MASK_ID

    with torch.no_grad():
        logits = model(x).logits
        preds = logits.argmax(dim=-1)

    for p in mask_pos:
        gold_id = ids[0, p].item()
        pred_id = preds[0, p].item()
        etype = classify_error(gold_id, pred_id)
        infer_error_types[etype] += 1
        if etype != 'correct':
            cos = F.cosine_similarity(wte[gold_id].unsqueeze(0), wte[pred_id].unsqueeze(0)).item()
            infer_cos_sims.append(cos)

total_infer = sum(infer_error_types.values())
print(f"\n推理错误分布 (50 texts, 30% mask):")
for etype in ['correct', 'semantic_similar', 'random']:
    n = infer_error_types[etype]
    print(f"  {etype:20s}: {n:5d} ({n/total_infer:.1%})")

if infer_cos_sims:
    print(f"\n推理错误的 cos(gold, pred) 分布:")
    print(f"  mean={np.mean(infer_cos_sims):.3f}, median={np.median(infer_cos_sims):.3f}")
    print(f"  std={np.std(infer_cos_sims):.3f}")
    # Histogram
    bins = [(-1, -0.5), (-0.5, 0), (0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print(f"  cos 分布:")
    for lo, hi in bins:
        n = sum(1 for c in infer_cos_sims if lo <= c < hi)
        print(f"    [{lo:+.1f}, {hi:+.1f}): {n:4d} ({n/len(infer_cos_sims):.1%})")


# ============================================================
# Phase 2: 训练噪声分析（converter 在不同 SNR 下的错误模式）
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: 训练噪声（converter）错误模式分析")
print("=" * 60)

# 取同样的 texts，对每个 token 加噪过 converter
snr_values = [0.5, 1, 2, 5, 10, 20, 50]
train_error_types = {snr: Counter() for snr in snr_values}
train_cos_sims = {snr: [] for snr in snr_values}

for ti, text in enumerate(texts[:20]):  # 20 texts 足够
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    L = ids.shape[1]
    if L < 10: continue

    for snr_val in snr_values:
        torch.manual_seed(SEED + ti)
        snr = torch.tensor(float(snr_val), device=device)

        # 对每个 token 加噪过 converter
        e = ne[ids[0]]  # (L, d_noise)
        eps = torch.randn(1, L, ne.shape[1], device=device)
        z = snr * e.unsqueeze(0) + torch.sqrt(snr) * eps

        # Converter softmax → argmax
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1).squeeze()  # (L, V+1)
        converter_preds = probs[:, :ne.shape[0]].argmax(dim=-1)  # (L,)

        for p in range(L):
            gold_id = ids[0, p].item()
            pred_id = converter_preds[p].item()
            etype = classify_error(gold_id, pred_id)
            train_error_types[snr_val][etype] += 1
            if etype != 'correct':
                cos = F.cosine_similarity(wte[gold_id].unsqueeze(0), wte[pred_id].unsqueeze(0)).item()
                train_cos_sims[snr_val].append(cos)

print(f"\n训练噪声（converter）错误分布:")
print(f"{'SNR':>6s} | {'correct':>10s} | {'semantic':>10s} | {'random':>10s} | {'error cos mean':>15s}")
print("-" * 65)
for snr_val in snr_values:
    total = sum(train_error_types[snr_val].values())
    c = train_error_types[snr_val]['correct']
    s = train_error_types[snr_val]['semantic_similar']
    r = train_error_types[snr_val]['random']
    cos_mean = np.mean(train_cos_sims[snr_val]) if train_cos_sims[snr_val] else 0
    print(f"{snr_val:6.1f} | {c/total:10.1%} | {s/total:10.1%} | {r/total:10.1%} | {cos_mean:+15.3f}")


# ============================================================
# Phase 3: 对比
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: 推理错误 vs 训练噪声 对比")
print("=" * 60)

# 推理错误的 cos 分布
infer_cos_arr = np.array(infer_cos_sims)
print(f"\n推理错误 cos 分布: mean={infer_cos_arr.mean():.3f}, std={infer_cos_arr.std():.3f}")

# 找哪个 SNR 的训练噪声最接近推理错误
print(f"\n各 SNR 训练噪声 cos 分布 vs 推理错误:")
for snr_val in snr_values:
    if not train_cos_sims[snr_val]:
        print(f"  SNR={snr_val}: no errors (all correct)")
        continue
    t_arr = np.array(train_cos_sims[snr_val])
    # KS test
    from scipy import stats
    ks_stat, ks_p = stats.ks_2samp(infer_cos_arr, t_arr)
    print(f"  SNR={snr_val:5.1f}: cos_mean={t_arr.mean():+.3f} (infer={infer_cos_arr.mean():+.3f})  "
          f"KS={ks_stat:.3f} p={ks_p:.4f}")

# 保存结果
results = {
    'inference': {
        'error_types': dict(infer_error_types),
        'cos_mean': float(np.mean(infer_cos_sims)),
        'cos_std': float(np.std(infer_cos_sims)),
    },
    'training_noise': {
        str(snr): {
            'error_types': dict(train_error_types[snr]),
            'cos_mean': float(np.mean(train_cos_sims[snr])) if train_cos_sims[snr] else None,
        }
        for snr in snr_values
    }
}
os.makedirs('results', exist_ok=True)
with open('results/train_vs_infer_noise.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/train_vs_infer_noise.json")
