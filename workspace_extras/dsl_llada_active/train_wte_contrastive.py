"""Train contrastive + orthogonal noise embedding from wte.
Goal: (1) preserve cosine structure from wte, (2) maintain discriminability for converter.

Loss = cosine_preservation + λ_orth * soft_orthogonality

Usage: CUDA_VISIBLE_DEVICES=1 python train_wte_contrastive.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os, math
from pathlib import Path
from safetensors.torch import load_file

t0 = time.time()
LATENT_DIM = 100
BATCH_SIZE = 4096
N_PAIRS = 512  # pairs per batch for cosine loss
EPOCHS = 200
LR = 1e-3
LAMBDA_ORTH = 0.1  # orthogonality weight
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load wte
ckpt = 'checkpoints/beta1_d100_1k/checkpoint-1000'
print(f"[{time.time()-t0:.1f}s] Loading wte...")
wte = None
for fn in sorted(os.listdir(ckpt)):
    if fn.endswith('.safetensors'):
        st = load_file(os.path.join(ckpt, fn), device='cpu')
        if 'model.transformer.wte.weight' in st:
            wte = st['model.transformer.wte.weight'].float()
            break
assert wte is not None
V, D = wte.shape
print(f"  wte: {V} x {D}")

wte_norm = F.normalize(wte, dim=-1).to(DEVICE)

# Projection network: 4096 → 100
class ProjectionNet(nn.Module):
    def __init__(self, d_in=4096, d_out=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, d_out),
        )

    def forward(self, x):
        return self.net(x)

model = ProjectionNet(D, LATENT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"[{time.time()-t0:.1f}s] Training contrastive projection: {D}→{LATENT_DIM}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  λ_orth={LAMBDA_ORTH}")

# Monitor pairs
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

def get_id(w):
    ids = tokenizer(w, add_special_tokens=False)['input_ids']
    return ids[0] if len(ids) == 1 else None

monitor_pairs = [
    (' cat', ' dog'), (' cat', ' Paris'), (' Paris', ' London'),
    (' one', ' two'), (' red', ' blue'), (' bird', ' fish'),
    (' Tokyo', ' Berlin'), (' three', ' four'), (' black', ' white'),
]

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(V, device=DEVICE)
    total_cos = 0; total_orth = 0; n_batches = 0

    for i in range(0, V, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x = wte_norm[idx]  # (B, 4096)
        z = model(x)       # (B, 100)
        z_norm = F.normalize(z, dim=-1)
        B = z_norm.shape[0]

        # --- Loss 1: Cosine similarity preservation ---
        n_p = min(N_PAIRS, B * (B - 1) // 2)
        i1 = torch.randint(0, B, (n_p,), device=DEVICE)
        i2 = torch.randint(0, B, (n_p,), device=DEVICE)
        # Avoid self-pairs
        mask = i1 != i2
        i1, i2 = i1[mask], i2[mask]

        cos_orig = F.cosine_similarity(x[i1], x[i2], dim=-1)
        cos_proj = F.cosine_similarity(z_norm[i1], z_norm[i2], dim=-1)
        loss_cos = F.mse_loss(cos_proj, cos_orig)

        # --- Loss 2: Soft orthogonality within batch ---
        # Average absolute pairwise cosine should be small (but not zero — allow semantic clusters)
        # Use a sample to avoid O(B²) compute
        n_orth = min(256, B)
        oi = torch.randint(0, B, (n_orth,), device=DEVICE)
        oj = torch.randint(0, B, (n_orth,), device=DEVICE)
        mask_orth = oi != oj
        oi, oj = oi[mask_orth], oj[mask_orth]
        cos_orth = F.cosine_similarity(z_norm[oi], z_norm[oj], dim=-1)
        # Penalize high absolute cosine (want it small but allow some structure)
        # Use hinge: penalize |cos| > 0.3 (allow moderate similarity)
        threshold = 0.3
        loss_orth = F.relu(cos_orth.abs() - threshold).pow(2).mean()

        loss = loss_cos + LAMBDA_ORTH * loss_orth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cos += loss_cos.item()
        total_orth += loss_orth.item()
        n_batches += 1

    scheduler.step()
    avg_cos = total_cos / n_batches
    avg_orth = total_orth / n_batches

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            z_all = []
            for j in range(0, V, BATCH_SIZE):
                z_all.append(model(wte_norm[j:j+BATCH_SIZE]))
            z_all = F.normalize(torch.cat(z_all, dim=0), dim=-1)

            # Avg pairwise cosine (sample)
            sample_idx = torch.randint(0, V, (1000,), device=DEVICE)
            z_sample = z_all[sample_idx]
            gram = z_sample @ z_sample.T
            mask_diag = ~torch.eye(1000, dtype=torch.bool, device=DEVICE)
            avg_abs_cos = gram[mask_diag].abs().mean().item()

        print(f"\n  Epoch {epoch+1:3d} | cos_loss={avg_cos:.6f} orth_loss={avg_orth:.6f} avg|cos|={avg_abs_cos:.4f}")
        print(f"  {'Pair':<20s} {'orig_cos':>8s} {'proj_cos':>10s} {'diff':>8s}")
        for w1, w2 in monitor_pairs:
            id1, id2 = get_id(w1), get_id(w2)
            if id1 is None or id2 is None:
                continue
            c_orig = F.cosine_similarity(wte_norm[id1:id1+1], wte_norm[id2:id2+1]).item()
            c_proj = F.cosine_similarity(z_all[id1:id1+1], z_all[id2:id2+1]).item()
            print(f"  {w1.strip()+'-'+w2.strip():<20s} {c_orig:>8.4f} {c_proj:>10.4f} {abs(c_orig-c_proj):>8.4f}")

print(f"\n[{time.time()-t0:.1f}s] Training done.")

# Extract final embeddings
model.eval()
with torch.no_grad():
    z_all = []
    for i in range(0, V, BATCH_SIZE):
        z_all.append(model(wte_norm[i:i+BATCH_SIZE]))
    z_all = F.normalize(torch.cat(z_all, dim=0), dim=-1)

# Quality metrics
sample_idx = torch.randint(0, V, (2000,), device=DEVICE)
z_sample = z_all[sample_idx]
gram = z_sample @ z_sample.T
mask_diag = ~torch.eye(2000, dtype=torch.bool, device=DEVICE)
avg_abs_cos = gram[mask_diag].abs().mean().item()
max_cos = gram[mask_diag].abs().max().item()
print(f"\nQuality: avg|cos|={avg_abs_cos:.4f}, max|cos|={max_cos:.4f}")

# Save
out_dir = Path('results')
out_dir.mkdir(exist_ok=True)
torch.save({
    'embedding': z_all.cpu(),
    'projector_state': model.state_dict(),
    'latent_dim': LATENT_DIM,
    'lambda_orth': LAMBDA_ORTH,
}, out_dir / 'wte_contrastive_embedding.pt')
print(f"[{time.time()-t0:.1f}s] Saved results/wte_contrastive_embedding.pt")

# Final cosine table
print(f"\nFinal cosine similarity preservation:")
print(f"  {'Pair':<20s} {'orig_cos':>8s} {'proj_cos':>8s} {'diff':>8s}")
for w1, w2 in monitor_pairs:
    id1, id2 = get_id(w1), get_id(w2)
    if id1 is None or id2 is None:
        continue
    c_orig = F.cosine_similarity(wte_norm[id1:id1+1], wte_norm[id2:id2+1]).item()
    c_proj = F.cosine_similarity(z_all[id1:id1+1], z_all[id2:id2+1]).item()
    print(f"  {w1.strip()+'-'+w2.strip():<20s} {c_orig:>8.4f} {c_proj:>8.4f} {abs(c_orig-c_proj):>8.4f}")
