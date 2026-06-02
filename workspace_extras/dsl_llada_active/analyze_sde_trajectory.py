"""SDE 生成的 per-token 轨迹深度分析。

分析维度：
1. Token 变化类型分类（语义替换/格式调整/完全跳变/重复锁定）
2. 变化方向性（convergent vs oscillating）
3. 空间相关性（clustered changes, wave patterns）
4. Phase transition 微观结构（哪些 token 先锁定）

Usage: CUDA_VISIBLE_DEVICES=5 python dsl_llada/analyze_sde_trajectory.py
"""
import torch, torch.nn.functional as F
import sys, os, math, json
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
device = 'cuda:0'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

print("Loading model...")
ckpt = 'checkpoints/beta1_d100_1k/checkpoint-1000'
model = AutoModelForCausalLM.from_pretrained(
    ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device).eval()
wte = model.model.transformer.wte.weight.detach().float()  # (V, 4096)

print("Loading DSL weights...")
for fn in sorted(os.listdir(ckpt)):
    if fn.endswith('.safetensors'):
        st = load_file(os.path.join(ckpt, fn), device=device)
        if 'noise_embed.weight' in st:
            ne = st['noise_embed.weight'].float()
            lb = st['converter.logit_bias'].float()
            bw = st['converter.backbone_embedding.weight'].float()
            bb = st['converter.backbone_embedding.bias'].float()
            bv = st['converter.beta'].item()
            print(f"  beta={bv:.4f}, noise_dim={ne.shape[1]}, V={ne.shape[0]}")
            break
K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0)
nd = ne.shape[1]

# Precompute normalized wte for cosine similarity
wte_norm = F.normalize(wte[:ne.shape[0]], dim=-1)  # (V, 4096)

# ====== SDE 参数 ======
STEPS = 32
GEN = 128
SEED = 42
BI = 3.0  # Beta1 model, bi=3.0
NS = 0.05
SNR_MAX = 80.0

# Sensitive schedule
snr_min = 3.0
n1 = max(1, int(STEPS * 0.05)); n2 = max(1, int(STEPS * 0.90)); n3 = STEPS - n1 - n2
snrs = torch.cat([
    torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
    torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
    torch.exp(torch.linspace(math.log(74), math.log(SNR_MAX), n3 + 1))[1:],
]).to(device)

# Norm init
def norm_init(B, L, dim, device):
    y = torch.randn(B, L, dim, device=device)
    return F.normalize(y, dim=-1)

prompts = {
    "Arithmetic": "Question: What is 27 + 58? Show the reasoning briefly, then give the final answer.",
    "Logic": "There are three boxes: red, blue, green. Exactly one contains a coin. Red is empty. If blue has the coin, green is empty. Where can the coin be?",
    "Creative": "Write a short story about a robot hearing rain for the first time. Make it vivid but not cheesy.",
}


def classify_token_change(old_id, new_id, prev_id, next_id):
    """分类 token 变化类型。
    Returns: 'semantic', 'format', 'jump', 'repeat_lock', 'same'
    """
    if old_id == new_id:
        return 'same'

    # 重复锁定：变成跟相邻 token 一样
    if new_id == prev_id or new_id == next_id:
        return 'repeat_lock'

    # 计算 wte 空间的 cosine similarity
    old_tok_str = tokenizer.decode([old_id]).strip()
    new_tok_str = tokenizer.decode([new_id]).strip()

    # 格式调整：标点/空格/大小写
    if old_tok_str.lower() == new_tok_str.lower():
        return 'format'
    if set(old_tok_str) <= set(' \t\n.,;:!?-()[]{}"\'/') or set(new_tok_str) <= set(' \t\n.,;:!?-()[]{}"\'/'):
        if len(old_tok_str) <= 2 and len(new_tok_str) <= 2:
            return 'format'

    # cosine similarity in wte space
    if old_id < wte_norm.shape[0] and new_id < wte_norm.shape[0]:
        cos = (wte_norm[old_id] * wte_norm[new_id]).sum().item()
        if cos > 0.3:
            return 'semantic'
        elif cos < 0.1:
            return 'jump'

    return 'semantic'  # default


def detect_oscillation(token_seq):
    """检测 token 序列的振荡模式。
    Returns: (pattern_type, details)
    - 'stable': 始终不变
    - 'converge': A→B→C→...→Final，渐进收敛
    - 'oscillate': A→B→A→B 模式
    - 'mixed': 先振荡后稳定
    """
    # 找出所有变化点
    changes = []
    for i in range(1, len(token_seq)):
        if token_seq[i] != token_seq[i-1]:
            changes.append(i)

    if len(changes) == 0:
        return 'stable', {'num_changes': 0}
    if len(changes) == 1:
        return 'converge', {'num_changes': 1, 'change_step': changes[0]}

    # 检查是否有重复 token（振荡）
    unique_tokens = list(set(token_seq))
    token_counts = Counter(token_seq)

    # 检测 A→B→A→B 模式
    oscillation_count = 0
    for i in range(2, len(token_seq)):
        if token_seq[i] == token_seq[i-2] and token_seq[i] != token_seq[i-1]:
            oscillation_count += 1

    if oscillation_count >= 2:
        # 检查是否后半部分稳定了
        last_change = changes[-1]
        if last_change < len(token_seq) * 0.7:
            return 'mixed', {
                'num_changes': len(changes),
                'oscillations': oscillation_count,
                'stable_from': last_change,
                'unique_tokens': len(unique_tokens)
            }
        return 'oscillate', {
            'num_changes': len(changes),
            'oscillations': oscillation_count,
            'unique_tokens': len(unique_tokens)
        }

    return 'converge', {
        'num_changes': len(changes),
        'unique_tokens': len(unique_tokens)
    }


def analyze_spatial_correlation(token_matrix, step):
    """分析同一步里变化 token 的空间相关性。
    token_matrix: list of (GEN,) token tensors
    step: 要分析的步骤
    Returns: cluster info
    """
    if step < 1 or step >= len(token_matrix):
        return {}
    changed = (token_matrix[step] != token_matrix[step-1]).numpy()
    positions = np.where(changed)[0]
    if len(positions) < 2:
        return {'n_changed': len(positions), 'clusters': [], 'cluster_sizes': []}

    # 找连续 cluster
    clusters = []
    current_cluster = [positions[0]]
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= 2:  # 允许间隔1
            current_cluster.append(positions[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [positions[i]]
    clusters.append(current_cluster)

    return {
        'n_changed': len(positions),
        'n_clusters': len(clusters),
        'cluster_sizes': [len(c) for c in clusters],
        'max_cluster': max(len(c) for c in clusters),
        'avg_cluster': np.mean([len(c) for c in clusters]),
        'positions': positions.tolist(),
    }


def get_xhat_and_tokens(model, y, snr, pe, K, BI, lb, bw, bb, ne, pl, dummy):
    """SDE 一步：获取预测 token 和置信度。"""
    z = snr * y
    cl = BI * (z.float() @ K.T) + lb
    probs = F.softmax(cl.float(), dim=-1)
    h = F.linear(probs, bw, bb).to(torch.bfloat16)
    embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
    with torch.no_grad():
        logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
    bp = F.softmax(logits, dim=-1)
    tokens = bp[0].argmax(dim=-1)
    conf = bp[0].max(dim=-1).values
    conv_probs = probs[0, :, :-1]
    conv_conf = conv_probs.max(dim=-1).values
    conv_tokens = conv_probs.argmax(dim=-1)
    xh = bp[:, :, :ne.shape[0]] @ ne
    return xh, tokens, conf, conv_conf, conv_tokens


def analyze_crystallization_order(all_tokens, tokenizer):
    """分析 crystallization 顺序：哪些 token 先锁定。"""
    n_steps = len(all_tokens)
    gen_len = all_tokens[0].shape[0]
    final_tokens = all_tokens[-1]

    # 对每个 position，找到它最后一次变化的步骤
    lock_step = []
    for p in range(gen_len):
        final_tok = final_tokens[p].item()
        last_change = 0
        for s in range(n_steps - 1, 0, -1):
            if all_tokens[s-1][p].item() != final_tok:
                last_change = s
                break
        lock_step.append(last_change)

    # 按 lock_step 排序
    positions_by_lock = sorted(range(gen_len), key=lambda p: lock_step[p])

    # 先锁定 vs 后锁定的 token 特征
    early_locked = [p for p in range(gen_len) if lock_step[p] <= n_steps * 0.3]
    late_locked = [p for p in range(gen_len) if lock_step[p] >= n_steps * 0.7]

    def categorize_tokens(positions):
        cats = Counter()
        for p in positions:
            tok_id = final_tokens[p].item()
            tok_str = tokenizer.decode([tok_id])
            # 分类
            if tok_str.strip() in '.,;:!?-()[]{}"\'/':
                cats['punctuation'] += 1
            elif tok_str.strip() in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on',
                                      'at', 'to', 'for', 'of', 'and', 'or', 'but', 'The', 'A']:
                cats['function_word'] += 1
            elif tok_str.strip().isdigit():
                cats['digit'] += 1
            elif tok_str.startswith(' ') and len(tok_str.strip()) <= 3:
                cats['short_word'] += 1
            elif tok_str == '\n' or tok_str == '\n\n':
                cats['newline'] += 1
            else:
                cats['content_word'] += 1
        return dict(cats)

    return {
        'lock_steps': lock_step,
        'early_locked': {
            'count': len(early_locked),
            'positions': early_locked[:20],
            'categories': categorize_tokens(early_locked),
            'tokens': [tokenizer.decode([final_tokens[p].item()]) for p in early_locked[:15]],
        },
        'late_locked': {
            'count': len(late_locked),
            'positions': late_locked[:20],
            'categories': categorize_tokens(late_locked),
            'tokens': [tokenizer.decode([final_tokens[p].item()]) for p in late_locked[:15]],
        },
    }


def detect_wave_pattern(token_matrix):
    """检测变化是否有波状传播模式。
    看每步变化的 position 的 median 是否单调移动。
    """
    n_steps = len(token_matrix)
    medians = []
    for s in range(1, n_steps):
        changed = (token_matrix[s] != token_matrix[s-1]).numpy()
        positions = np.where(changed)[0]
        if len(positions) > 0:
            medians.append((s, float(np.median(positions)), len(positions)))
        else:
            medians.append((s, None, 0))

    # 检查 median position 是否单调递增/递减
    valid_medians = [(s, m, n) for s, m, n in medians if m is not None and n >= 3]
    if len(valid_medians) < 5:
        return {'has_wave': False, 'medians': medians}

    # 简单的趋势检测
    ms = [m for _, m, _ in valid_medians]
    diffs = [ms[i+1] - ms[i] for i in range(len(ms)-1)]
    pos_frac = sum(1 for d in diffs if d > 0) / len(diffs)

    return {
        'has_wave': pos_frac > 0.7 or pos_frac < 0.3,
        'direction': 'left_to_right' if pos_frac > 0.7 else ('right_to_left' if pos_frac < 0.3 else 'none'),
        'position_trend_frac': pos_frac,
        'medians': [(s, m, n) for s, m, n in medians[:10]],
    }


# ====== Main ======
results = {}

for case_name, prompt in prompts.items():
    print(f"\n{'='*70}")
    print(f"Case: {case_name}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"{'='*70}")

    msgs = [{'role': 'user', 'content': prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(fmt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)

    torch.manual_seed(SEED)
    y = norm_init(1, GEN, nd, device)

    # 记录每步数据
    all_tokens = []
    all_confs = []
    all_conv_confs = []
    all_conv_tokens = []
    all_snr_vals = []

    for i in range(len(snrs) - 1):
        sc, sn = snrs[i], snrs[i + 1]
        ds = sn - sc
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        xh, tokens, conf, conv_conf, conv_tok = get_xhat_and_tokens(
            model, y, sc, pe, K, BI, lb, bw, bb, ne, pl, dummy)
        all_tokens.append(tokens.cpu())
        all_confs.append(conf.cpu())
        all_conv_confs.append(conv_conf.cpu())
        all_conv_tokens.append(conv_tok.cpu())
        all_snr_vals.append(sc.item())

        # Heun SDE step
        f = (xh - y) / sc; g = NS / sc
        ye = y + f * ds + g * dW
        xhe, _, _, _, _ = get_xhat_and_tokens(
            model, ye, sn, pe, K, BI, lb, bw, bb, ne, pl, dummy)
        fe = (xhe - ye) / sn; ge = NS / sn
        y = y + 0.5 * (f + fe) * ds + 0.5 * (g + ge) * dW

    # Final step
    xh, tokens, conf, conv_conf, conv_tok = get_xhat_and_tokens(
        model, y, snrs[-1], pe, K, BI, lb, bw, bb, ne, pl, dummy)
    all_tokens.append(tokens.cpu())
    all_confs.append(conf.cpu())
    all_conv_confs.append(conv_conf.cpu())
    all_conv_tokens.append(conv_tok.cpu())
    all_snr_vals.append(snrs[-1].item())

    final_text = tokenizer.decode(all_tokens[-1], skip_special_tokens=True)
    print(f"Final text: {final_text[:300]}")

    n_steps = len(all_tokens)
    token_matrix = all_tokens  # list of (GEN,) tensors

    # ====== 分析 1: Token 变化类型分类 ======
    print("\n--- Analysis 1: Token Change Classification ---")
    change_types_all = defaultdict(int)
    change_types_per_step = []

    for s in range(1, n_steps):
        step_types = defaultdict(int)
        for p in range(GEN):
            old_id = token_matrix[s-1][p].item()
            new_id = token_matrix[s][p].item()
            prev_id = token_matrix[s][max(0, p-1)].item()
            next_id = token_matrix[s][min(GEN-1, p+1)].item()
            ctype = classify_token_change(old_id, new_id, prev_id, next_id)
            if ctype != 'same':
                step_types[ctype] += 1
                change_types_all[ctype] += 1
        change_types_per_step.append(dict(step_types))

    print(f"  Total changes: {dict(change_types_all)}")
    total_changes = sum(change_types_all.values())
    if total_changes > 0:
        for ct, count in sorted(change_types_all.items(), key=lambda x: -x[1]):
            print(f"    {ct}: {count} ({100*count/total_changes:.1f}%)")

    # ====== 分析 2: 变化方向性 ======
    print("\n--- Analysis 2: Convergence vs Oscillation ---")
    direction_stats = Counter()
    oscillation_details = []

    for p in range(GEN):
        tok_seq = [token_matrix[s][p].item() for s in range(n_steps)]
        pattern, details = detect_oscillation(tok_seq)
        direction_stats[pattern] += 1
        if pattern == 'oscillate':
            oscillation_details.append({
                'position': p,
                'token_text': tokenizer.decode([tok_seq[-1]]),
                **details
            })

    print(f"  Pattern distribution: {dict(direction_stats)}")
    if oscillation_details:
        print(f"  Top oscillating positions:")
        for d in sorted(oscillation_details, key=lambda x: -x['oscillations'])[:5]:
            print(f"    pos {d['position']}: '{d['token_text'].strip()}' "
                  f"({d['oscillations']} oscillations, {d['unique_tokens']} unique tokens)")

    # ====== 分析 3: 空间相关性 ======
    print("\n--- Analysis 3: Spatial Correlation ---")
    spatial_data = []
    for s in range(1, n_steps):
        sp = analyze_spatial_correlation(token_matrix, s)
        sp['step'] = s
        sp['snr'] = all_snr_vals[s]
        spatial_data.append(sp)

    # 报告有 cluster 的步骤
    clustered_steps = [sp for sp in spatial_data if sp.get('max_cluster', 0) >= 4]
    print(f"  Steps with large clusters (>=4 consecutive): {len(clustered_steps)}")
    for sp in clustered_steps[:5]:
        print(f"    step {sp['step']} (SNR={sp['snr']:.1f}): "
              f"{sp['n_changed']} changed, {sp['n_clusters']} clusters, "
              f"max_cluster={sp['max_cluster']}, sizes={sp['cluster_sizes'][:8]}")

    # Wave detection
    wave = detect_wave_pattern(token_matrix)
    print(f"  Wave pattern: {wave['has_wave']} "
          f"(direction={wave.get('direction', 'N/A')}, "
          f"trend_frac={wave.get('position_trend_frac', 'N/A')})")

    # ====== 分析 4: Phase transition 微观结构 ======
    print("\n--- Analysis 4: Crystallization Order ---")
    crystal = analyze_crystallization_order(all_tokens, tokenizer)

    print(f"  Early locked (<30% steps): {crystal['early_locked']['count']} positions")
    print(f"    Categories: {crystal['early_locked']['categories']}")
    print(f"    Sample tokens: {crystal['early_locked']['tokens'][:10]}")
    print(f"  Late locked (>70% steps): {crystal['late_locked']['count']} positions")
    print(f"    Categories: {crystal['late_locked']['categories']}")
    print(f"    Sample tokens: {crystal['late_locked']['tokens'][:10]}")

    # ====== 补充: Per-step summary ======
    print("\n--- Per-Step Summary ---")
    print(f"{'step':>4s} {'SNR':>8s} {'changed':>8s} {'semantic':>9s} {'jump':>6s} {'repeat':>7s} {'format':>7s} {'avg_conf':>9s}")
    for s in range(1, n_steps):
        changed = (token_matrix[s] != token_matrix[s-1]).sum().item()
        ct = change_types_per_step[s-1]
        avg_conf = all_confs[s].mean().item()
        print(f"{s:4d} {all_snr_vals[s]:8.2f} {changed:8d}/{GEN} "
              f"{ct.get('semantic',0):9d} {ct.get('jump',0):6d} "
              f"{ct.get('repeat_lock',0):7d} {ct.get('format',0):7d} "
              f"{avg_conf:9.4f}")

    # ====== 补充: 详细振荡轨迹 ======
    print("\n--- Detailed Oscillation Traces ---")
    for p in range(GEN):
        tok_seq = [token_matrix[s][p].item() for s in range(n_steps)]
        pattern, _ = detect_oscillation(tok_seq)
        if pattern == 'oscillate':
            # 找变化的 token 和步骤
            changes_str = []
            for s in range(n_steps):
                if s == 0 or tok_seq[s] != tok_seq[s-1]:
                    tok_text = tokenizer.decode([tok_seq[s]]).strip()
                    changes_str.append(f"s{s}:'{tok_text}'")
            if len(changes_str) <= 15:
                print(f"  pos {p}: {' → '.join(changes_str)}")

    # ====== 补充: Converter vs Backbone agreement ======
    print("\n--- Converter vs Backbone Agreement ---")
    for s in [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]:
        agree = (all_tokens[s] == all_conv_tokens[s]).sum().item()
        print(f"  step {s} (SNR={all_snr_vals[s]:.1f}): "
              f"agree={agree}/{GEN} ({100*agree/GEN:.1f}%), "
              f"conv_conf={all_conv_confs[s].mean().item():.4f}, "
              f"backbone_conf={all_confs[s].mean().item():.4f}")

    # ====== 存储结果 ======
    case_result = {
        'prompt': prompt,
        'final_text': final_text,
        'gen_length': GEN,
        'steps': n_steps,
        'snr_values': all_snr_vals,
        'change_types_total': dict(change_types_all),
        'change_types_per_step': change_types_per_step,
        'direction_stats': dict(direction_stats),
        'oscillation_positions': [d['position'] for d in oscillation_details],
        'spatial_correlation': [
            {k: v for k, v in sp.items() if k != 'positions'}
            for sp in spatial_data
        ],
        'wave_pattern': {
            'has_wave': wave['has_wave'],
            'direction': wave.get('direction', 'none'),
            'trend_frac': wave.get('position_trend_frac', None),
        },
        'crystallization': {
            'lock_steps': crystal['lock_steps'],
            'early_locked': {
                'count': crystal['early_locked']['count'],
                'categories': crystal['early_locked']['categories'],
            },
            'late_locked': {
                'count': crystal['late_locked']['count'],
                'categories': crystal['late_locked']['categories'],
            },
        },
        'changes_per_step': [(token_matrix[s] != token_matrix[s-1]).sum().item()
                            for s in range(1, n_steps)],
        'avg_confidence_per_step': [all_confs[s].mean().item() for s in range(n_steps)],
        'converter_backbone_agreement': [
            (all_tokens[s] == all_conv_tokens[s]).sum().item() / GEN
            for s in range(n_steps)
        ],
        # Per-position token trajectory (compact: only record changes)
        'position_trajectories': {},
    }

    # 只记录有变化的 position 的轨迹
    for p in range(GEN):
        tok_seq = [token_matrix[s][p].item() for s in range(n_steps)]
        changes = [(s, tok_seq[s]) for s in range(n_steps) if s == 0 or tok_seq[s] != tok_seq[s-1]]
        if len(changes) > 1:
            case_result['position_trajectories'][str(p)] = {
                'token_ids': [c[1] for c in changes],
                'steps': [c[0] for c in changes],
                'token_texts': [tokenizer.decode([c[1]]).replace('\n', '\\n') for c in changes],
                'pattern': detect_oscillation(tok_seq)[0],
            }

    results[case_name] = case_result

# ====== 跨 case 对比总结 ======
print(f"\n{'='*70}")
print("CROSS-CASE COMPARISON")
print(f"{'='*70}")

for case_name in results:
    r = results[case_name]
    n_osc = r['direction_stats'].get('oscillate', 0)
    n_conv = r['direction_stats'].get('converge', 0)
    n_stable = r['direction_stats'].get('stable', 0)
    n_mixed = r['direction_stats'].get('mixed', 0)
    total_ch = sum(r['change_types_total'].values())
    jump_pct = 100 * r['change_types_total'].get('jump', 0) / max(1, total_ch)
    sem_pct = 100 * r['change_types_total'].get('semantic', 0) / max(1, total_ch)
    rep_pct = 100 * r['change_types_total'].get('repeat_lock', 0) / max(1, total_ch)

    # 找 crystallization step (最后一步有变化的步骤)
    cps = r['changes_per_step']
    crystal_step = 0
    for i in range(len(cps)-1, -1, -1):
        if cps[i] > 0:
            crystal_step = i + 1
            break

    print(f"\n{case_name}:")
    print(f"  Crystallization: step {crystal_step}/{len(cps)} "
          f"(SNR≈{r['snr_values'][min(crystal_step, len(r['snr_values'])-1)]:.1f})")
    print(f"  Patterns: stable={n_stable}, converge={n_conv}, oscillate={n_osc}, mixed={n_mixed}")
    print(f"  Change types: semantic={sem_pct:.0f}%, jump={jump_pct:.0f}%, repeat={rep_pct:.0f}%")
    print(f"  Wave: {r['wave_pattern']['direction']}")
    print(f"  Early lock categories: {r['crystallization']['early_locked']['categories']}")
    print(f"  Late lock categories: {r['crystallization']['late_locked']['categories']}")
    print(f"  Final text: {r['final_text'][:150]}...")

# ====== 保存 ======
out_path = 'results/sde_trajectory_analysis.json'
os.makedirs('results', exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nResults saved to {out_path}")

# ====== 关键发现总结 ======
print(f"\n{'='*70}")
print("关键发现总结")
print(f"{'='*70}")

arith = results.get('Arithmetic', {})
logic = results.get('Logic', {})
creative = results.get('Creative', {})

print("""
1. TOKEN 变化类型:
   - 各 case 的 semantic/jump/repeat 比例差异揭示生成质量差别
   - repeat_lock（重复锁定）比例高 → 退化性重复的微观证据

2. 振荡 vs 收敛:
   - 数学题: 预期 converge 为主（答案明确）
   - Logic: 预期 oscillate 较多（模型对答案不确定）
   - Creative: 预期 mixed（先尝试不同方向，后锁定但可能锁在重复上）

3. 空间相关性:
   - 如果变化是 clustered → 模型按"短语"为单位更新
   - 如果是随机散布 → 模型按独立 token 更新

4. Crystallization 顺序:
   - 结构词（the, is, of）先锁定 → 模型先确定句子骨架
   - 内容词后锁定 → 模型先搭框架再填内容
   - 数字最后锁定 → 计算结果需要最多 iteration

5. 实用价值:
   - 振荡比例可作为生成质量的早期预警信号
   - 如果 >20% position 在振荡，early stop 可能更好
   - cluster 变化模式可用于自适应 step size
""")
