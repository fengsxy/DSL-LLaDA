"""t-SNE with SEMANTIC noise embedding: PCA(wte, 100d) + normalize.
Compare impostor tokens: do they become synonyms?

Generates: paper_figures/tsne_interactive_semantics.html

Usage: CUDA_VISIBLE_DEVICES=0 python exp_tsne_semantic.py
"""
import torch, torch.nn.functional as F
import numpy as np, os, sys, time, math, json
from pathlib import Path
from safetensors.torch import load_file
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

t0 = time.time()
MASK_ID = 126336
NOISE_DIM = 100
TOP_K = 64

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

def get_id(word):
    ids = tokenizer(word, add_special_tokens=False)['input_ids']
    return ids[0] if len(ids) == 1 else None

CATEGORIES = {
    'Animal':  {w: get_id(w) for w in [' cat', ' dog', ' bird', ' fish', ' horse']},
    'Number':  {w: get_id(w) for w in [' one', ' two', ' three', ' four', ' five']},
    'Color':   {w: get_id(w) for w in [' red', ' blue', ' green', ' black', ' white']},
    'City':    {w: get_id(w) for w in [' Paris', ' London', ' Tokyo', ' Berlin', ' Rome']},
    'Digit':   {w: get_id(w) for w in ['0', '1', '2', '3', '7']},
}
for cat in CATEGORIES:
    CATEGORIES[cat] = {k: v for k, v in CATEGORIES[cat].items() if v is not None}

CAT_COLORS = {
    'Animal': '#1f77b4', 'Number': '#2ca02c', 'Color': '#9467bd',
    'City': '#d62728', 'Digit': '#ff7f0e',
}

print(f"[{time.time()-t0:.1f}s] Categories:")
for cat, tokens in CATEGORIES.items():
    print(f"  {cat}: {tokens}")

# SNR schedule
_steps = 32; _snr_min = 0.01; _snr_max = 80
_n1 = max(1, int(_steps * 0.05)); _n2 = max(1, int(_steps * 0.90)); _n3 = _steps - _n1 - _n2
_snrs = torch.cat([
    torch.exp(torch.linspace(math.log(_snr_min), math.log(7), _n1 + 1)),
    torch.exp(torch.linspace(math.log(7), math.log(74), _n2 + 1))[1:],
    torch.exp(torch.linspace(math.log(74), math.log(_snr_max), _n3 + 1))[1:],
])
snr_values = [round(s.item(), 2) for s in _snrs]

CKPTS = {
    'β=0.3': 'checkpoints/pertoken_b03_d100_1k/checkpoint-1000',
    'β=1.0': 'checkpoints/beta1_d100_1k/checkpoint-1000',
    'β=2.0': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
}


def load_dsl_weights(ckpt_path):
    for fn in sorted(os.listdir(ckpt_path)):
        if fn.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt_path, fn), device='cpu')
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bv = st['converter.beta'].item()
                return ne, lb, bv
    return None, None, None


def load_wte(ckpt_path):
    # Load only the shard that has wte (avoid loading all shards)
    wte_file = os.path.join(ckpt_path, 'model-00001-of-00004.safetensors')
    if os.path.exists(wte_file):
        st = load_file(wte_file, device='cpu')
        if 'model.transformer.wte.weight' in st:
            return st['model.transformer.wte.weight'].float()
    # Fallback: scan all
    for fn in sorted(os.listdir(ckpt_path)):
        if fn.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt_path, fn), device='cpu')
            if 'model.transformer.wte.weight' in st:
                return st['model.transformer.wte.weight'].float()
    return None


# ── Load semantic noise embedding ──
print(f"[{time.time()-t0:.1f}s] Loading wte...")
wte = load_wte(list(CKPTS.values())[0])
V_wte = wte.shape[0]  # 126464
print(f"[{time.time()-t0:.1f}s] wte shape: {wte.shape}")

# Try AE+Contrastive first, then AE, then PCA
ae_cont_path = Path('results/wte_ae_contrastive_embedding.pt')
ae_path = Path('results/wte_ae_embedding.pt')

if ae_cont_path.exists():
    print(f"[{time.time()-t0:.1f}s] Loading AE+Contrastive embedding...")
    data = torch.load(ae_cont_path, map_location='cpu', weights_only=True)
    key = 'latent_embeddings' if 'latent_embeddings' in data else 'embedding'
    ne_semantic = data[key].float()
    NOISE_DIM = ne_semantic.shape[1]
    embed_method = "AE+Contrastive"
elif ae_path.exists():
    print(f"[{time.time()-t0:.1f}s] Loading AE embedding...")
    data = torch.load(ae_path, map_location='cpu', weights_only=True)
    key = 'latent_embeddings' if 'latent_embeddings' in data else 'embedding'
    ne_semantic = data[key].float()
    NOISE_DIM = ne_semantic.shape[1]
    embed_method = "AE"
else:
    print(f"[{time.time()-t0:.1f}s] PCA {wte.shape[1]}d → {NOISE_DIM}d...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=NOISE_DIM, random_state=42)
    wte_pca = pca.fit_transform(wte.numpy())
    ne_semantic = F.normalize(torch.from_numpy(wte_pca).float(), dim=-1)
    embed_method = f"PCA (var={pca.explained_variance_ratio_.sum():.3f})"

print(f"[{time.time()-t0:.1f}s] Semantic embedding ready: {ne_semantic.shape} ({embed_method})")

# Check: cosine similarity between cat/dog vs cat/XCTAssertEqual
tid_cat = get_id(' cat')
tid_dog = get_id(' dog')
tid_paris = get_id(' Paris')
tid_london = get_id(' London')
tid_one = get_id(' one')
tid_two = get_id(' two')
cos = F.cosine_similarity
print(f"\n  Semantic cosine similarities:")
print(f"    cat-dog:     {cos(ne_semantic[tid_cat].unsqueeze(0), ne_semantic[tid_dog].unsqueeze(0)).item():.4f}")
print(f"    cat-Paris:   {cos(ne_semantic[tid_cat].unsqueeze(0), ne_semantic[tid_paris].unsqueeze(0)).item():.4f}")
print(f"    Paris-London:{cos(ne_semantic[tid_paris].unsqueeze(0), ne_semantic[tid_london].unsqueeze(0)).item():.4f}")
print(f"    one-two:     {cos(ne_semantic[tid_one].unsqueeze(0), ne_semantic[tid_two].unsqueeze(0)).item():.4f}")


def compute_trajectories_semantic(ne, bv, wte, categories, snr_values):
    """Same as original but with semantic embedding. Use simple uniform logit_bias."""
    V = ne.shape[0]
    K = torch.cat([ne, torch.zeros(1, ne.shape[1])], dim=0)  # (V+1, d)
    h_mask = wte[MASK_ID]

    # Simple logit_bias: mask slot gets log(V) advantage
    lb = torch.zeros(1, 1, V + 1)
    lb[0, 0, -1] = math.log(V)  # mask bias

    all_targets = {}
    for cat, tokens in categories.items():
        for tname, tid in tokens.items():
            all_targets[tname] = (tid, cat)

    embeddings = []
    labels = []
    meta = []
    transition_snr = {}

    N_NOISE_SEEDS = 5
    torch.manual_seed(42)

    for tname, (tid, cat) in all_targets.items():
        ne_tok = ne[tid]
        found = False
        for snr in snr_values:
            # Clean trajectory
            z = snr * ne_tok.unsqueeze(0).unsqueeze(0)
            cl = bv * (z.float() @ K.T) + lb
            probs = F.softmax(cl.float(), dim=-1).squeeze()
            real_probs = probs[:V]
            mask_prob = probs[V].item()

            topk_vals, topk_ids = real_probs.topk(TOP_K)
            h = mask_prob * h_mask
            for val, idx in zip(topk_vals, topk_ids.tolist()):
                h = h + val.item() * wte[idx]

            embeddings.append(h)
            labels.append(f"{tname}_{snr}")
            meta.append(("traj", tname, cat))

            if not found and real_probs[tid].item() > 0.5:
                transition_snr[tname] = snr
                found = True

            # Noisy samples
            for seed_i in range(N_NOISE_SEEDS):
                eps = torch.randn_like(ne_tok)
                z_noisy = snr * ne_tok + math.sqrt(max(snr, 0)) * eps
                cl_n = bv * (z_noisy.unsqueeze(0).unsqueeze(0).float() @ K.T) + lb
                probs_n = F.softmax(cl_n.float(), dim=-1).squeeze()
                real_probs_n = probs_n[:V]
                top1_id = real_probs_n.argmax().item()
                if top1_id != tid:
                    mask_prob_n = probs_n[V].item()
                    topk_vals_n, topk_ids_n = real_probs_n.topk(min(TOP_K, 8))
                    h_imp = mask_prob_n * h_mask
                    for val, idx in zip(topk_vals_n, topk_ids_n.tolist()):
                        h_imp = h_imp + val.item() * wte[idx]
                    embeddings.append(h_imp)
                    imp_word = tokenizer.decode([top1_id]).strip()
                    labels.append(f"imp_{tname}_{snr}_{imp_word}")
                    meta.append(("impostor", tname, cat))

    # MASK point
    embeddings.append(h_mask.clone())
    labels.append("MASK")
    meta.append(("mask", "mask", "mask"))

    # Endpoints
    for tname, (tid, cat) in all_targets.items():
        embeddings.append(wte[tid])
        labels.append(f"wte[{tname}]")
        meta.append(("endpoint", tname, cat))

    return torch.stack(embeddings).numpy(), meta, labels, transition_snr, all_targets


# ── Run for 3 beta values ──
# Use trained beta values but with semantic embedding
beta_values = {'β=0.3': 0.3, 'β=1.0': 1.0, 'β=2.0': 2.0}

all_X2d = []; all_meta = []; all_labels = []; all_trans = []; all_targets_list = []; all_bvs = []

for model_name, bv in beta_values.items():
    print(f"\n[{time.time()-t0:.1f}s] Processing {model_name} (semantic embed)...")

    X, meta, lbls, trans_snr, all_targets = compute_trajectories_semantic(
        ne_semantic, bv, wte, CATEGORIES, snr_values)

    n_imp = sum(1 for m in meta if m[0] == 'impostor')
    print(f"  Matrix: {X.shape}, {n_imp} impostor points")
    print(f"  Transitions: {trans_snr}")

    # Print impostor analysis
    impostors = {}
    for i, m in enumerate(meta):
        if m[0] == 'impostor':
            gold = m[1]
            parts = lbls[i].split('_', 3)
            if len(parts) >= 4:
                impostors.setdefault(gold, []).append(parts[3])
    print(f"  Impostor samples:")
    for gold, imps in sorted(impostors.items()):
        from collections import Counter
        top5 = Counter(imps).most_common(5)
        print(f"    {gold.strip():>8s} -> {top5}")

    # PCA + t-SNE
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    pca_tsne = PCA(n_components=min(50, X_norm.shape[0] - 1), random_state=42)
    X_pca = pca_tsne.fit_transform(X_norm)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_pca) - 1),
                random_state=42, max_iter=2000, metric='cosine')
    X2d = tsne.fit_transform(X_pca)

    all_X2d.append(X2d); all_meta.append(meta); all_labels.append(lbls)
    all_trans.append(trans_snr); all_targets_list.append(all_targets); all_bvs.append(bv)


# ── Generate interactive HTML ──
print(f"\n[{time.time()-t0:.1f}s] Generating interactive HTML...")

html_parts = []
html_parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>t-SNE Semantic Embedding ({embed_method})</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ font-family: sans-serif; background: #fff; margin: 20px; }}
h1 {{ font-size: 18px; }}
.info {{ color: #666; font-size: 13px; margin-bottom: 15px; }}
.panel {{ display: inline-block; vertical-align: top; }}
</style>
</head><body>
<h1>Converter Trajectory with Semantic Noise Embedding ({embed_method})</h1>
<p class="info">Noise embedding = {embed_method} (wte→{NOISE_DIM}d) + L2 normalize. Hover to see impostor details.<br>
Key question: do impostors become synonyms when embedding has semantic structure?</p>
""")

model_names_list = list(beta_values.keys())

for ax_idx in range(3):
    model_name = model_names_list[ax_idx]
    X2d = all_X2d[ax_idx]
    meta = all_meta[ax_idx]
    lbl_list = all_labels[ax_idx]
    all_targets = all_targets_list[ax_idx]
    bv = all_bvs[ax_idx]

    traces = []
    traj_data = []; imp_data = []; mask_data = None; ep_data = []

    for i, m in enumerate(meta):
        mtype, mname, mcat = m
        x, y = float(X2d[i][0]), float(X2d[i][1])
        lb_text = lbl_list[i] if i < len(lbl_list) else ""

        if mtype == "traj":
            traj_data.append((x, y, mname, mcat, lb_text))
        elif mtype == "impostor":
            parts = lb_text.split('_', 3)
            imp_word = parts[3] if len(parts) > 3 else '?'
            snr_str = parts[2] if len(parts) > 2 else '?'
            hover = f"gold={mname.strip()} SNR={snr_str}<br>impostor={imp_word}"
            imp_data.append((x, y, mname, mcat, hover))
        elif mtype == "mask":
            mask_data = (x, y)
        elif mtype == "endpoint":
            ep_data.append((x, y, mname, mcat))

    # Trajectory traces per category
    for cat in CATEGORIES:
        c = CAT_COLORS[cat]
        pts = [(d[0], d[1], d[4]) for d in traj_data if d[3] == cat]
        if pts:
            traces.append({
                'x': [p[0] for p in pts], 'y': [p[1] for p in pts],
                'text': [p[2] for p in pts],
                'mode': 'markers', 'type': 'scatter',
                'marker': {'size': 5, 'color': c, 'opacity': 0.7},
                'name': f'{cat} (traj)',
                'hovertemplate': '%{text}<extra></extra>',
            })

    # Impostor traces per category
    for cat in CATEGORIES:
        c = CAT_COLORS[cat]
        pts = [(d[0], d[1], d[4]) for d in imp_data if d[3] == cat]
        if pts:
            traces.append({
                'x': [p[0] for p in pts], 'y': [p[1] for p in pts],
                'text': [p[2] for p in pts],
                'mode': 'markers', 'type': 'scatter',
                'marker': {'size': 4, 'color': c, 'opacity': 0.3, 'symbol': 'x'},
                'name': f'{cat} impostor' if ax_idx == 2 else None,
                'showlegend': ax_idx == 2,
                'hovertemplate': '%{text}<extra></extra>',
            })

    if mask_data:
        traces.append({
            'x': [mask_data[0]], 'y': [mask_data[1]],
            'text': ['MASK'], 'mode': 'markers+text', 'type': 'scatter',
            'marker': {'size': 14, 'color': 'black', 'symbol': 'x'},
            'textposition': 'top center', 'name': 'MASK',
            'hovertemplate': 'MASK<extra></extra>',
        })

    if ep_data:
        traces.append({
            'x': [d[0] for d in ep_data], 'y': [d[1] for d in ep_data],
            'text': [d[2].strip() for d in ep_data],
            'mode': 'markers+text', 'type': 'scatter',
            'marker': {'size': 10, 'color': [CAT_COLORS[d[3]] for d in ep_data],
                       'symbol': 'star', 'line': {'width': 1, 'color': 'black'}},
            'textposition': 'top right', 'textfont': {'size': 9},
            'name': 'Endpoint (wte)',
            'hovertemplate': 'wte[%{text}]<extra></extra>',
        })

    div_id = f"plot{ax_idx}"
    html_parts.append(f'<div class="panel"><div id="{div_id}" style="width:550px;height:500px;"></div></div>')
    layout = {
        'title': {'text': f'{model_name} (semantic embed)', 'font': {'size': 14}},
        'xaxis': {'title': 't-SNE dim 1'},
        'yaxis': {'title': 't-SNE dim 2' if ax_idx == 0 else ''},
        'showlegend': ax_idx == 2,
        'legend': {'font': {'size': 9}},
        'margin': {'l': 50, 'r': 20, 't': 40, 'b': 50},
        'hovermode': 'closest',
    }
    html_parts.append(f"""<script>
Plotly.newPlot("{div_id}", {json.dumps(traces)}, {json.dumps(layout)});
</script>""")

html_parts.append("</body></html>")
out_dir = Path('paper_figures')
out_dir.mkdir(exist_ok=True)
html_path = out_dir / 'tsne_interactive_semantics.html'
html_path.write_text("\n".join(html_parts))
print(f"\n[{time.time()-t0:.1f}s] Saved {html_path}")
print(f"[{time.time()-t0:.1f}s] ALL DONE.")
