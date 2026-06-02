"""Train autoencoder on wte embeddings: 4096 → 100d → 4096.
Goal: semantic noise embedding that preserves cosine structure.

Usage: python train_wte_autoencoder.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os
from pathlib import Path
from safetensors.torch import load_file

t0 = time.time()
LATENT_DIM = 100
BATCH_SIZE = 4096
EPOCHS = 200
LR = 1e-3
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

# Normalize wte for training (work in unit-norm space)
wte_norm = F.normalize(wte, dim=-1)
wte_data = wte_norm.to(DEVICE)


class WteAutoencoder(nn.Module):
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


model = WteAutoencoder(D, LATENT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"[{time.time()-t0:.1f}s] Training AE: {D}→{LATENT_DIM}→{D}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Device: {DEVICE}")

# Cosine similarity pairs for monitoring
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

best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(V, device=DEVICE)
    total_recon = 0; total_cos = 0; n_batches = 0

    for i in range(0, V, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x = wte_data[idx]

        recon, z = model(x)

        # Reconstruction loss (MSE on normalized vectors)
        loss_recon = F.mse_loss(recon, x)

        # Cosine preservation loss on random pairs within batch
        if len(idx) >= 64:
            n_pairs = 256
            i1 = torch.randint(0, len(idx), (n_pairs,), device=DEVICE)
            i2 = torch.randint(0, len(idx), (n_pairs,), device=DEVICE)
            cos_orig = F.cosine_similarity(x[i1], x[i2], dim=-1)
            z_norm = F.normalize(z, dim=-1)
            cos_latent = F.cosine_similarity(z_norm[i1], z_norm[i2], dim=-1)
            loss_cos = F.mse_loss(cos_latent, cos_orig)
        else:
            loss_cos = torch.tensor(0.0, device=DEVICE)

        loss = loss_recon + 0.5 * loss_cos

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += loss_recon.item()
        total_cos += loss_cos.item()
        n_batches += 1

    scheduler.step()
    avg_recon = total_recon / n_batches
    avg_cos = total_cos / n_batches

    if (epoch + 1) % 10 == 0 or epoch == 0:
        # Monitor cosine preservation
        model.eval()
        with torch.no_grad():
            z_all = []
            for i in range(0, V, BATCH_SIZE):
                z_all.append(model.encode(wte_data[i:i+BATCH_SIZE]))
            z_all = F.normalize(torch.cat(z_all, dim=0), dim=-1)

        print(f"\n  Epoch {epoch+1:3d} | recon={avg_recon:.6f} cos={avg_cos:.6f} lr={scheduler.get_last_lr()[0]:.6f}")
        print(f"  {'Pair':<20s} {'orig_cos':>8s} {'latent_cos':>10s} {'diff':>8s}")
        for w1, w2 in monitor_pairs:
            id1, id2 = get_id(w1), get_id(w2)
            if id1 is None or id2 is None:
                continue
            c_orig = F.cosine_similarity(wte_data[id1:id1+1], wte_data[id2:id2+1]).item()
            c_lat = F.cosine_similarity(z_all[id1:id1+1], z_all[id2:id2+1]).item()
            print(f"  {w1.strip()+'-'+w2.strip():<20s} {c_orig:>8.4f} {c_lat:>10.4f} {abs(c_orig-c_lat):>8.4f}")

    if avg_recon < best_loss:
        best_loss = avg_recon

print(f"\n[{time.time()-t0:.1f}s] Training done. Best recon loss: {best_loss:.6f}")

# Extract final latent embeddings
model.eval()
with torch.no_grad():
    z_all = []
    for i in range(0, V, BATCH_SIZE):
        z_all.append(model.encode(wte_data[i:i+BATCH_SIZE]))
    z_all = F.normalize(torch.cat(z_all, dim=0), dim=-1)

# Save
out_dir = Path('results')
out_dir.mkdir(exist_ok=True)
torch.save({
    'embedding': z_all.cpu(),
    'encoder_state': model.encoder.state_dict(),
    'decoder_state': model.decoder.state_dict(),
    'latent_dim': LATENT_DIM,
    'pca_variance': None,
}, out_dir / 'wte_ae_embedding.pt')
print(f"[{time.time()-t0:.1f}s] Saved results/wte_ae_embedding.pt")

# Final quality check
print(f"\nFinal cosine similarity preservation:")
print(f"  {'Pair':<20s} {'orig_cos':>8s} {'AE_cos':>8s} {'diff':>8s}")
for w1, w2 in monitor_pairs:
    id1, id2 = get_id(w1), get_id(w2)
    if id1 is None or id2 is None:
        continue
    c_orig = F.cosine_similarity(wte_data[id1:id1+1], wte_data[id2:id2+1]).item()
    c_ae = F.cosine_similarity(z_all[id1:id1+1], z_all[id2:id2+1]).item()
    print(f"  {w1.strip()+'-'+w2.strip():<20s} {c_orig:>8.4f} {c_ae:>8.4f} {abs(c_orig-c_ae):>8.4f}")
