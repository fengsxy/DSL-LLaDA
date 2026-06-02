"""Train AE + Contrastive embedding from wte: 4096 → 100d → 4096.
Combined objectives:
  1. Reconstruction: decode(encode(wte)) ≈ wte
  2. Cosine preservation: cos(z_i, z_j) ≈ cos(wte_i, wte_j)
  3. Discriminability: different tokens should be distinguishable (soft orthogonality)

Usage: CUDA_VISIBLE_DEVICES=0 python train_wte_ae_contrastive.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, os
from pathlib import Path
from safetensors.torch import load_file

t0 = time.time()
LATENT_DIM = 100
BATCH_SIZE = 4096
EPOCHS = 300
LR = 1e-3
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Loss weights
W_RECON = 1.0       # reconstruction
W_COS = 1.0         # cosine preservation
W_DISCRIM = 0.3     # discriminability (soft orthogonality)
N_PAIRS = 512        # pairs per batch

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
print(f"  wte: {V} x {D}, device={DEVICE}")

wte_norm = F.normalize(wte, dim=-1).to(DEVICE)


class WteAEContrastive(nn.Module):
    def __init__(self, d_in=4096, d_latent=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, d_in),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


model = WteAEContrastive(D, LATENT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in model.parameters())
print(f"[{time.time()-t0:.1f}s] Model params: {n_params:,}")
print(f"  Weights: recon={W_RECON}, cos={W_COS}, discrim={W_DISCRIM}")

# Monitor pairs
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    def get_id(w):
        ids = tokenizer(w, add_special_tokens=False)['input_ids']
        return ids[0] if len(ids) == 1 else None
    HAS_TOK = True
except:
    HAS_TOK = False

monitor_pairs = [
    (' cat', ' dog'), (' cat', ' Paris'), (' Paris', ' London'),
    (' one', ' two'), (' red', ' blue'), (' bird', ' fish'),
    (' Tokyo', ' Berlin'), (' three', ' four'), (' black', ' white'),
]

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(V, device=DEVICE)
    ep_recon = 0; ep_cos = 0; ep_disc = 0; n_batches = 0

    for i in range(0, V, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x = wte_norm[idx]  # (B, 4096)
        B = x.shape[0]

        recon, z = model(x)
        z_norm = F.normalize(z, dim=-1)

        # ── Loss 1: Reconstruction ──
        loss_recon = F.mse_loss(recon, x)

        # ── Loss 2: Cosine similarity preservation ──
        n_p = min(N_PAIRS, B * (B - 1) // 2)
        i1 = torch.randint(0, B, (n_p,), device=DEVICE)
        i2 = torch.randint(0, B, (n_p,), device=DEVICE)
        mask = i1 != i2
        i1, i2 = i1[mask], i2[mask]
        if len(i1) > 0:
            cos_orig = F.cosine_similarity(x[i1], x[i2], dim=-1)
            cos_latent = F.cosine_similarity(z_norm[i1], z_norm[i2], dim=-1)
            loss_cos = F.mse_loss(cos_latent, cos_orig)
        else:
            loss_cos = torch.tensor(0.0, device=DEVICE)

        # ── Loss 3: Discriminability ──
        # Penalize |cos| > threshold between random pairs
        # This keeps tokens distinguishable for converter's argmax
        n_d = min(256, B)
        di = torch.randint(0, B, (n_d,), device=DEVICE)
        dj = torch.randint(0, B, (n_d,), device=DEVICE)
        dmask = di != dj
        di, dj = di[dmask], dj[dmask]
        if len(di) > 0:
            cos_d = F.cosine_similarity(z_norm[di], z_norm[dj], dim=-1)
            # Hinge at 0.5: allow moderate similarity for semantics, penalize too close
            loss_disc = F.relu(cos_d.abs() - 0.5).pow(2).mean()
        else:
            loss_disc = torch.tensor(0.0, device=DEVICE)

        loss = W_RECON * loss_recon + W_COS * loss_cos + W_DISCRIM * loss_disc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_recon += loss_recon.item()
        ep_cos += loss_cos.item()
        ep_disc += loss_disc.item()
        n_batches += 1

    scheduler.step()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        avg_r = ep_recon / n_batches
        avg_c = ep_cos / n_batches
        avg_d = ep_disc / n_batches

        model.eval()
        with torch.no_grad():
            z_all = []
            for j in range(0, V, BATCH_SIZE):
                z_all.append(model.encode(wte_norm[j:j+BATCH_SIZE]))
            z_all = F.normalize(torch.cat(z_all, dim=0), dim=-1)

            # Discriminability metrics
            sample_idx = torch.randint(0, V, (1000,), device=DEVICE)
            z_sample = z_all[sample_idx]
            gram = z_sample @ z_sample.T
            mask_diag = ~torch.eye(1000, dtype=torch.bool, device=DEVICE)
            avg_abs_cos = gram[mask_diag].abs().mean().item()
            max_cos = gram[mask_diag].abs().max().item()
            frac_above05 = (gram[mask_diag].abs() > 0.5).float().mean().item()

        print(f"\n  Epoch {epoch+1:3d}/{EPOCHS} | recon={avg_r:.6f} cos={avg_c:.6f} disc={avg_d:.6f}")
        print(f"    avg|cos|={avg_abs_cos:.4f}  max|cos|={max_cos:.4f}  frac>0.5={frac_above05:.4f}")

        if HAS_TOK:
            print(f"    {'Pair':<20s} {'orig':>6s} {'latent':>7s} {'diff':>6s}")
            for w1, w2 in monitor_pairs:
                id1, id2 = get_id(w1), get_id(w2)
                if id1 is None or id2 is None: continue
                c_o = F.cosine_similarity(wte_norm[id1:id1+1], wte_norm[id2:id2+1]).item()
                c_l = F.cosine_similarity(z_all[id1:id1+1], z_all[id2:id2+1]).item()
                print(f"    {w1.strip()+'-'+w2.strip():<20s} {c_o:>6.3f} {c_l:>7.3f} {abs(c_o-c_l):>6.3f}")

print(f"\n[{time.time()-t0:.1f}s] Training done.")

# Final embeddings
model.eval()
with torch.no_grad():
    z_all = []
    for i in range(0, V, BATCH_SIZE):
        z_all.append(model.encode(wte_norm[i:i+BATCH_SIZE]))
    z_all = F.normalize(torch.cat(z_all, dim=0), dim=-1)

# Save
out_dir = Path('results')
out_dir.mkdir(exist_ok=True)
save_path = out_dir / 'wte_ae_contrastive_embedding.pt'
torch.save({
    'embedding': z_all.cpu(),
    'encoder_state': model.encoder.state_dict(),
    'decoder_state': model.decoder.state_dict(),
    'latent_dim': LATENT_DIM,
    'weights': {'recon': W_RECON, 'cos': W_COS, 'discrim': W_DISCRIM},
}, save_path)
print(f"[{time.time()-t0:.1f}s] Saved {save_path}")

# Final report
sample_idx = torch.randint(0, V, (2000,), device=DEVICE)
z_sample = z_all[sample_idx]
gram = z_sample @ z_sample.T
mask_diag = ~torch.eye(2000, dtype=torch.bool, device=DEVICE)
avg_abs_cos = gram[mask_diag].abs().mean().item()

print(f"\nFinal quality: avg|cos|={avg_abs_cos:.4f}")
if HAS_TOK:
    print(f"\n  {'Pair':<20s} {'orig':>6s} {'final':>7s} {'diff':>6s}")
    for w1, w2 in monitor_pairs:
        id1, id2 = get_id(w1), get_id(w2)
        if id1 is None or id2 is None: continue
        c_o = F.cosine_similarity(wte_norm[id1:id1+1], wte_norm[id2:id2+1]).item()
        c_l = F.cosine_similarity(z_all[id1:id1+1], z_all[id2:id2+1]).item()
        print(f"  {w1.strip()+'-'+w2.strip():<20s} {c_o:>6.3f} {c_l:>7.3f} {abs(c_o-c_l):>6.3f}")
