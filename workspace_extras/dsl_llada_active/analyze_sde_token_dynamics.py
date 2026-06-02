"""分析 SDE 每一步每个 token 的变化动态。

核心问题：beta=2 SNR 跳变时到底发生了什么？
- 每步记录所有 position 的 argmax token
- 哪些 token 在哪一步 "crystallize"（不再变化）
- 跳变前后 token 的语义关系
- 对比 correct vs wrong 的 problem

Usage: CUDA_VISIBLE_DEVICES=0 python analyze_sde_token_dynamics.py
"""
import torch, torch.nn.functional as F
import sys, os, math, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

# Load Beta2 model + DSL weights
ckpt = 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000'
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

STEPS = 32
GEN = 64  # 短一点方便分析
SEED = 42

# SDE 参数（跟 repo 里一致）
BI = bv  # 用训练的 beta
NS = 0.05
SNR_MAX = 50.0

# Sensitive schedule
snr_min = 3.0
n1 = max(1, int(STEPS * 0.05)); n2 = max(1, int(STEPS * 0.90)); n3 = STEPS - n1 - n2
snrs = torch.cat([
    torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
    torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
    torch.exp(torch.linspace(math.log(74), math.log(SNR_MAX), n3 + 1))[1:],
]).to(device)

prompts = [
    "Write a short paragraph about the history of France.",
    "Explain why water boils at 100 degrees Celsius.",
    "Solve step by step: If a train travels at 60 mph for 2.5 hours, how far does it travel?",
]

for pi, prompt in enumerate(prompts):
    msgs = [{'role': 'user', 'content': prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)

    torch.manual_seed(SEED + pi)
    y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)

    def get_xhat_and_tokens(yy, ss):
        z = ss * yy
        cl = BI * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        bp = F.softmax(logits, dim=-1)
        tokens = bp[0].argmax(dim=-1)  # (GEN,)
        conf = bp[0].max(dim=-1).values  # (GEN,)
        # converter confidence
        conv_conf = probs[0, :, :-1].max(dim=-1).values.mean().item()
        xh = bp[:, :, :ne.shape[0]] @ ne
        return xh, tokens, conf, conv_conf

    # 记录每步的 token
    all_tokens = []  # list of (GEN,) tensors
    all_confs = []
    all_conv_confs = []
    all_snr_vals = []

    for i in range(len(snrs) - 1):
        sc, sn = snrs[i], snrs[i + 1]
        ds = sn - sc
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        xh, tokens, conf, conv_conf = get_xhat_and_tokens(y, sc)
        all_tokens.append(tokens.cpu())
        all_confs.append(conf.cpu())
        all_conv_confs.append(conv_conf)
        all_snr_vals.append(sc.item())

        f = (xh - y) / sc; g = NS / sc
        ye = y + f * ds + g * dW
        xhe, _, _, _ = get_xhat_and_tokens(ye, sn)
        fe = (xhe - ye) / sn; ge = NS / sn
        y = y + 0.5 * (f + fe) * ds + 0.5 * (g + ge) * dW

    # Final step
    xh, tokens, conf, conv_conf = get_xhat_and_tokens(y, snrs[-1])
    all_tokens.append(tokens.cpu())
    all_confs.append(conf.cpu())
    all_conv_confs.append(conv_conf)
    all_snr_vals.append(snrs[-1].item())

    # 分析
    n_steps = len(all_tokens)
    print(f"\n{'='*70}")
    print(f"Prompt {pi}: {prompt[:60]}...")
    print(f"{'='*70}")
    print(f"Final text: {tokenizer.decode(all_tokens[-1], skip_special_tokens=True)[:200]}")

    # 1. 每步有多少 token 变了
    print(f"\n--- Token Changes Per Step ---")
    print(f"{'step':>4s} {'SNR':>8s} {'changed':>8s} {'conv_conf':>10s} {'backbone_conf':>14s}")
    for s in range(1, n_steps):
        changed = (all_tokens[s] != all_tokens[s-1]).sum().item()
        snr = all_snr_vals[s]
        cc = all_conv_confs[s]
        bc = all_confs[s].mean().item()
        print(f"{s:4d} {snr:8.2f} {changed:8d}/{GEN} {cc:10.3f} {bc:14.3f}")

    # 2. 每个 position 的 crystallization step
    lock_steps = []
    for p in range(GEN):
        final_tok = all_tokens[-1][p].item()
        lock = n_steps - 1
        for s in range(n_steps - 1, 0, -1):
            if all_tokens[s-1][p].item() != final_tok:
                lock = s
                break
        lock_steps.append(lock)

    # 3. 跳变分析：找最大变化的步骤
    changes_per_step = []
    for s in range(1, n_steps):
        c = (all_tokens[s] != all_tokens[s-1]).sum().item()
        changes_per_step.append((s, c, all_snr_vals[s]))
    changes_per_step.sort(key=lambda x: -x[1])

    print(f"\n--- Top 5 Most Volatile Steps ---")
    for s, c, snr in changes_per_step[:5]:
        print(f"  step {s}: {c}/{GEN} tokens changed, SNR={snr:.2f}")
        # 显示变化的前 5 个 position
        changed_pos = (all_tokens[s] != all_tokens[s-1]).nonzero().squeeze(-1).tolist()[:5]
        for p in changed_pos:
            old_tok = tokenizer.decode([all_tokens[s-1][p].item()]).strip()
            new_tok = tokenizer.decode([all_tokens[s][p].item()]).strip()
            print(f"    pos {p}: '{old_tok}' → '{new_tok}'")

    # 4. 跳变前后的对比（SNR transition zone 附近）
    print(f"\n--- Token Sequence at Key SNR Points ---")
    key_steps = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    for s in key_steps:
        text = tokenizer.decode(all_tokens[s][:32], skip_special_tokens=True)
        print(f"  step {s:2d} (SNR={all_snr_vals[s]:6.2f}): {text[:100]}")

print(f"\n{'='*70}")
print("DONE")
