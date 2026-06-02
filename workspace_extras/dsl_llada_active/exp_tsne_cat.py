"""t-SNE in WTE space: 5 categories × 5 tokens, converter trajectory from MASK→token.
h(SNR) = Σ p_i(SNR) × wte[i].

Adapted for our LLaDA-8B checkpoints. Runs 3 models (beta=0.3, 1.0, 2.0) side by side.

Usage: CUDA_VISIBLE_DEVICES=0 python exp_tsne_cat.py
"""
import torch, torch.nn.functional as F
import numpy as np, os, sys, time, gc, math
from pathlib import Path
from safetensors.torch import load_file
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

t0 = time.time()
DPI = 300
matplotlib.rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif',
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7, 'legend.frameon': False,
    'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

MASK_ID = 126336

# Token categories - use LLaDA tokenizer IDs
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

# Filter out None
for cat in CATEGORIES:
    CATEGORIES[cat] = {k: v for k, v in CATEGORIES[cat].items() if v is not None}

CAT_COLORS = {
    'Animal': '#1f77b4', 'Number': '#2ca02c', 'Color': '#9467bd',
    'City': '#d62728', 'Digit': '#ff7f0e',
}

print(f"[{time.time()-t0:.1f}s] Categories:")
for cat, tokens in CATEGORIES.items():
    print(f"  {cat}: {tokens}")

# SNR schedule (sensitive, 32 steps)
_steps = 32; _snr_min = 0.01; _snr_max = 80
_n1 = max(1, int(_steps * 0.05)); _n2 = max(1, int(_steps * 0.90)); _n3 = _steps - _n1 - _n2
_snrs = torch.cat([
    torch.exp(torch.linspace(math.log(_snr_min), math.log(7), _n1 + 1)),
    torch.exp(torch.linspace(math.log(7), math.log(74), _n2 + 1))[1:],
    torch.exp(torch.linspace(math.log(74), math.log(_snr_max), _n3 + 1))[1:],
])
snr_values = [round(s.item(), 2) for s in _snrs]
TOP_K = 64

# 3 checkpoints
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
    for fn in sorted(os.listdir(ckpt_path)):
        if fn.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt_path, fn), device='cpu')
            if 'model.transformer.wte.weight' in st:
                return st['model.transformer.wte.weight'].float()
    return None


def compute_trajectories(ne, lb, bv, wte, categories, snr_values):
    """For each token, compute h(SNR) = Σ p_i(SNR) × wte[i] at each SNR.
    Also collect impostor points: when top-1 != gold, record the impostor's wte."""
    K = torch.cat([ne, torch.zeros(1, ne.shape[1])], dim=0)
    V = ne.shape[0]
    h_mask = wte[MASK_ID]

    all_targets = {}
    for cat, tokens in categories.items():
        for tname, tid in tokens.items():
            all_targets[tname] = (tid, cat)

    embeddings = []
    labels = []
    meta = []
    transition_snr = {}

    N_NOISE_SEEDS = 5  # number of noise samples per (token, SNR)
    torch.manual_seed(42)

    for tname, (tid, cat) in all_targets.items():
        ne_tok = ne[tid]
        found = False
        for snr in snr_values:
            # Clean trajectory point (no noise)
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

            # Noisy samples: z = SNR * e + sqrt(SNR) * eps
            for seed_i in range(N_NOISE_SEEDS):
                eps = torch.randn_like(ne_tok)
                z_noisy = snr * ne_tok + math.sqrt(max(snr, 0)) * eps
                cl_n = bv * (z_noisy.unsqueeze(0).unsqueeze(0).float() @ K.T) + lb
                probs_n = F.softmax(cl_n.float(), dim=-1).squeeze()
                real_probs_n = probs_n[:V]
                top1_id = real_probs_n.argmax().item()
                if top1_id != tid:
                    # Impostor: use the weighted-sum embedding (like trajectory point)
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

    # Endpoint (pure wte)
    for tname, (tid, cat) in all_targets.items():
        embeddings.append(wte[tid])
        labels.append(f"wte[{tname}]")
        meta.append(("endpoint", tname, cat))

    return torch.stack(embeddings).numpy(), meta, labels, transition_snr, all_targets


CACHE_FILE = Path('results/tsne_cache_v2.npz')
USE_CACHE = CACHE_FILE.exists() and '--no-cache' not in sys.argv

if USE_CACHE:
    print(f"[{time.time()-t0:.1f}s] Loading cached t-SNE from {CACHE_FILE}...")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    cached_X2d = cache['X2d_all']
    cached_meta = cache['meta_all']
    cached_labels = cache['labels_all']
    cached_trans = cache['trans_all']
    cached_targets = cache['targets_all']
    cached_bvs = cache['bvs']
else:
    print(f"[{time.time()-t0:.1f}s] Loading wte...")
    wte = load_wte(list(CKPTS.values())[0])
    print(f"[{time.time()-t0:.1f}s] wte shape: {wte.shape}")
    cached_X2d = []; cached_meta = []; cached_labels = []; cached_trans = []; cached_targets = []; cached_bvs = []

# Plot 3 panels
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax_idx, (model_name, ckpt_path) in enumerate(CKPTS.items()):
    ax = axes[ax_idx]

    if USE_CACHE:
        X2d = cached_X2d[ax_idx]
        meta = cached_meta[ax_idx]
        all_labels = cached_labels[ax_idx]
        trans_snr = cached_trans[ax_idx]
        all_targets = cached_targets[ax_idx]
        bv = cached_bvs[ax_idx]
        print(f"[{time.time()-t0:.1f}s] {model_name} (cached, β={bv:.4f})")
    else:
        print(f"\n[{time.time()-t0:.1f}s] Processing {model_name}...")
        ne, lb, bv = load_dsl_weights(ckpt_path)
        print(f"  β_trained={bv:.4f}")

        X, meta, all_labels, trans_snr, all_targets = compute_trajectories(ne, lb, bv, wte, CATEGORIES, snr_values)
        print(f"  Transitions: {trans_snr}")
        print(f"  Matrix: {X.shape}")

        # PCA + t-SNE
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        pca = PCA(n_components=min(50, X_norm.shape[0] - 1), random_state=42)
        X_pca = pca.fit_transform(X_norm)
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_pca) - 1),
                    random_state=42, max_iter=2000, metric='cosine')
        X2d = tsne.fit_transform(X_pca)

        cached_X2d.append(X2d); cached_meta.append(meta); cached_labels.append(all_labels)
        cached_trans.append(trans_snr); cached_targets.append(all_targets)
        cached_bvs.append(bv)

    # Split using meta tags (since impostors are interleaved)
    traj_pts = {}  # tname -> list of (x,y) with corresponding snr
    traj_snrs = {}
    impostor_pts = []  # list of (x, y, cat)
    mask_pt = None
    endpoint_pts = {}

    for i, m in enumerate(meta):
        mtype, mname, mcat = m
        if mtype == "traj":
            traj_pts.setdefault(mname, []).append(X2d[i])
            # figure out which snr index this is
            traj_snrs.setdefault(mname, []).append(len(traj_snrs.get(mname, [])))
        elif mtype == "impostor":
            impostor_pts.append((X2d[i], mcat))
        elif mtype == "mask":
            mask_pt = X2d[i]
        elif mtype == "endpoint":
            endpoint_pts[mname] = X2d[i]

    for tname in traj_pts:
        traj_pts[tname] = np.array(traj_pts[tname])

    # Draw
    snr_arr = np.array(snr_values, dtype=float)
    sc = None
    for tname, (tid, cat) in all_targets.items():
        pts = traj_pts[tname]
        c = CAT_COLORS[cat]
        n_pts = len(pts)
        # snr_arr might differ in length if some steps missing; use actual count
        s_arr = snr_arr[:n_pts]
        ax.plot(pts[:, 0], pts[:, 1], color=c, alpha=0.25, linewidth=0.8, zorder=2)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=s_arr, cmap='coolwarm',
                        s=12, zorder=3, edgecolors=c, linewidths=0.3,
                        vmin=snr_arr.min(), vmax=snr_arr.max())
        # Transition ring
        if tname in trans_snr and trans_snr[tname] in snr_values:
            ti = snr_values.index(trans_snr[tname])
            if ti < n_pts:
                ax.scatter(pts[ti, 0], pts[ti, 1], s=80, facecolors='none',
                           edgecolors=c, linewidths=1.5, zorder=5)

    # Impostor noise points — color by source category
    if impostor_pts:
        imp_x = np.array([p[0] for p in impostor_pts])
        imp_cats = [p[1] for p in impostor_pts]
        imp_colors = [CAT_COLORS.get(c, 'gray') for c in imp_cats]
        ax.scatter(imp_x[:, 0], imp_x[:, 1], marker='.', c=imp_colors,
                   s=4, alpha=0.15, zorder=1)
        n_imp = len(impostor_pts)
    else:
        n_imp = 0
    print(f"  {model_name}: {n_imp} impostor points")

    # MASK
    ax.scatter(mask_pt[0], mask_pt[1], marker='X', c='k', s=200, zorder=7)
    ax.annotate('MASK', (mask_pt[0], mask_pt[1]), fontsize=8, fontweight='bold',
                xytext=(5, -10), textcoords='offset points')

    # Endpoints
    for tname, (tid, cat) in all_targets.items():
        ep = endpoint_pts[tname]
        c = CAT_COLORS[cat]
        ax.scatter(ep[0], ep[1], marker='*', c=c, s=80, zorder=6, edgecolors='k', linewidths=0.5)
        ax.annotate(tname.strip(), (ep[0], ep[1]), fontsize=5.5, color=c,
                    xytext=(3, 3), textcoords='offset points')

    n_trans = len(trans_snr)
    median_trans = sorted(trans_snr.values())[len(trans_snr)//2] if trans_snr else 0
    ax.set_title(f"{model_name} (β={bv:.2f})\ntransition SNR≈{median_trans:.0f}, {n_trans}/{len(all_targets)} reach >50%")
    ax.set_xlabel("t-SNE dim 1")
    if ax_idx == 0:
        ax.set_ylabel("t-SNE dim 2")

# Colorbar
cbar = plt.colorbar(sc, ax=axes[-1], shrink=0.5, pad=0.08, aspect=25, location='right')
cbar.set_label('SNR', fontsize=9)

# Legend
legend_elements = [Line2D([0], [0], marker='X', color='w', markerfacecolor='k', markersize=8, label='MASK')]
for cat in CATEGORIES:
    legend_elements.append(Line2D([0], [0], color=CAT_COLORS[cat], linewidth=2, label=cat))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=8,
                               markeredgecolor='gray', markeredgewidth=1.5, label='Transition (p>50%)'))
legend_elements.append(Line2D([0], [0], marker='.', color='w', markerfacecolor='gray', markersize=6,
                               alpha=0.5, label='Impostor (top-1≠gold)'))
fig.legend(handles=legend_elements, loc='upper left', ncol=1, fontsize=7,
           bbox_to_anchor=(0.01, 0.98), borderaxespad=0)

plt.suptitle("Converter Output Trajectory in WTE Space: MASK $\\rightarrow$ Token\n"
             "Each dot = weighted sum of wte embeddings at a given SNR. Color = SNR level.",
             fontsize=13, y=1.02)
plt.tight_layout()

out_dir = Path('paper_figures')
out_dir.mkdir(exist_ok=True)
for fmt in ['png']:
    fig.savefig(out_dir / f'tsne_wte_3beta.{fmt}', format=fmt, dpi=DPI, bbox_inches='tight')
    print(f"\n[{time.time()-t0:.1f}s] Saved paper_figures/tsne_wte_3beta.{fmt}")
plt.close(fig)

# Save cache for fast re-rendering
if not USE_CACHE:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_FILE,
             X2d_all=np.array(cached_X2d, dtype=object),
             meta_all=np.array(cached_meta, dtype=object),
             labels_all=np.array(cached_labels, dtype=object),
             trans_all=np.array(cached_trans, dtype=object),
             targets_all=np.array(cached_targets, dtype=object),
             bvs=np.array(cached_bvs))
    print(f"[{time.time()-t0:.1f}s] Cached t-SNE data to {CACHE_FILE}")

# ── Interactive HTML with plotly ──
print(f"[{time.time()-t0:.1f}s] Generating interactive HTML...")
import json as _json

html_parts = []
html_parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>t-SNE Converter Trajectories (interactive)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body { font-family: sans-serif; background: #fff; margin: 20px; }
h1 { font-size: 18px; }
.panel { display: inline-block; vertical-align: top; }
</style>
</head><body>
<h1>Converter Output Trajectory in WTE Space: MASK → Token</h1>
<p style="color:#666;font-size:13px;">Hover over points to see token details. Gray dots = impostor tokens (noisy top-1 ≠ gold).</p>
""")

model_names_list = list(CKPTS.keys())

for ax_idx in range(3):
    model_name = model_names_list[ax_idx]
    X2d = cached_X2d[ax_idx]
    meta = cached_meta[ax_idx]
    lbl_list = cached_labels[ax_idx]
    all_targets = cached_targets[ax_idx]
    bv = cached_bvs[ax_idx]

    # Build traces for plotly
    traces = []
    traj_data = []; imp_data = []; mask_data = None; ep_data = []

    for i, m in enumerate(meta):
        mtype, mname, mcat = m
        x, y = float(X2d[i][0]), float(X2d[i][1])
        lb_text = lbl_list[i] if i < len(lbl_list) else ""

        if mtype == "traj":
            # Parse SNR from label like "cat_13.73"
            traj_data.append((x, y, mname, mcat, lb_text))
        elif mtype == "impostor":
            # Label like "imp_cat_7.0_dog" → parse impostor word
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

    # Impostor traces per category (colored by source)
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

    # MASK
    if mask_data:
        traces.append({
            'x': [mask_data[0]], 'y': [mask_data[1]],
            'text': ['MASK'], 'mode': 'markers+text', 'type': 'scatter',
            'marker': {'size': 14, 'color': 'black', 'symbol': 'x'},
            'textposition': 'top center',
            'name': 'MASK',
            'hovertemplate': 'MASK<extra></extra>',
        })

    # Endpoints
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
        'title': {'text': f'{model_name} (β={bv:.2f})', 'font': {'size': 14}},
        'xaxis': {'title': 't-SNE dim 1'},
        'yaxis': {'title': 't-SNE dim 2' if ax_idx == 0 else ''},
        'showlegend': ax_idx == 2,
        'legend': {'font': {'size': 9}},
        'margin': {'l': 50, 'r': 20, 't': 40, 'b': 50},
        'hovermode': 'closest',
    }
    html_parts.append(f"""<script>
Plotly.newPlot("{div_id}", {_json.dumps(traces)}, {_json.dumps(layout)});
</script>""")

html_parts.append("</body></html>")
html_path = out_dir / 'tsne_interactive.html'
html_path.write_text("\n".join(html_parts))
print(f"[{time.time()-t0:.1f}s] Saved {html_path}")

print(f"\n[{time.time()-t0:.1f}s] ALL DONE.")
