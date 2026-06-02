"""Train a contrastive autoencoder to compress LLaDA wte embeddings into
a compact noise embedding space for DSL models.

Based on reference implementation. Adapted for LLaDA-8B (126464 tokens, 4096d → 100d).

Usage:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python dsl_llada/train_ae_contrastive_embed.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os
from pathlib import Path
from safetensors.torch import load_file


class ContrastiveAutoEncoder(nn.Module):
    def __init__(self, d_input: int, d_latent: int, d_hidden: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_input),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def main():
    t0 = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    LATENT_DIM = 100
    EPOCHS = 500
    BATCH_SIZE = 2048
    LR = 1e-3
    CONTRASTIVE_WEIGHT = 0.5

    # Load wte from LLaDA checkpoint
    ckpt = 'checkpoints/beta1_d100_1k/checkpoint-1000'
    wte_file = os.path.join(ckpt, 'model-00001-of-00004.safetensors')
    print(f"[{time.time()-t0:.1f}s] Loading wte from {wte_file}...")
    st = load_file(wte_file, device='cpu')
    embeddings = st['model.transformer.wte.weight'].float().to(device)
    del st
    V, D = embeddings.shape
    print(f"[{time.time()-t0:.1f}s] wte: {V} x {D}, device={device}")

    # Train
    ae = ContrastiveAutoEncoder(D, LATENT_DIM).to(device)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=LR, weight_decay=1e-4)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"  Params: {n_params:,}, epochs={EPOCHS}, λ_contrastive={CONTRASTIVE_WEIGHT}")

    for epoch in range(EPOCHS):
        idx = torch.randperm(V, device=device)
        total_recon, total_contrast, n_batches = 0.0, 0.0, 0

        for i in range(0, V, BATCH_SIZE):
            batch = embeddings[idx[i:i + BATCH_SIZE]]
            recon, z = ae(batch)

            loss_recon = F.mse_loss(recon, batch)

            z_norm = F.normalize(z, dim=1)
            gram = z_norm @ z_norm.T
            mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=device)
            loss_contrast = (gram[mask] ** 2).mean()

            loss = loss_recon + CONTRASTIVE_WEIGHT * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += loss_recon.item()
            total_contrast += loss_contrast.item()
            n_batches += 1

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:4d}/{EPOCHS} ({time.time()-t0:.0f}s): "
                  f"recon={total_recon/n_batches:.6f}  contrast={total_contrast/n_batches:.6f}")

    # Extract and normalize
    ae.eval()
    with torch.no_grad():
        latent = []
        for i in range(0, V, BATCH_SIZE):
            latent.append(ae.encode(embeddings[i:i+BATCH_SIZE]))
        latent = torch.cat(latent, dim=0)
    latent_norm = F.normalize(latent, dim=1)

    # Evaluate
    sample_idx = torch.randperm(V)[:2000]
    gram = latent_norm[sample_idx] @ latent_norm[sample_idx].T
    mask = ~torch.eye(2000, dtype=torch.bool, device=device)
    off = gram[mask]
    print(f"\nOrthogonality (2000 tokens): mean|cos|={off.abs().mean():.4f}, max|cos|={off.abs().max():.4f}")

    # Nearest neighbors
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
        test_words = [' cat', ' dog', ' red', ' blue', ' Paris', ' London', ' one', ' two', ' three']
        print(f"\nNearest neighbors in latent space:")
        for word in test_words:
            ids = tokenizer(word, add_special_tokens=False)['input_ids']
            if len(ids) == 1:
                tid = ids[0]
                sims = latent_norm[tid] @ latent_norm.T
                sims[tid] = -2
                top5v, top5i = sims.topk(5)
                neighbors = [f"{tokenizer.decode([i.item()]).strip()}({v.item():.3f})"
                             for v, i in zip(top5v, top5i)]
                print(f"  {word.strip():>8s}: {'  '.join(neighbors)}")
    except Exception as e:
        print(f"Tokenizer not available: {e}")

    # Save
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / 'wte_ae_contrastive_embedding.pt'
    torch.save({
        'latent_embeddings': latent_norm.cpu(),
        'latent_raw': latent.cpu(),
        'ae_state_dict': ae.cpu().state_dict(),
        'config': {
            'd_input': D, 'd_latent': LATENT_DIM, 'vocab_size': V,
            'contrastive_weight': CONTRASTIVE_WEIGHT, 'epochs': EPOCHS,
        },
    }, save_path)
    print(f"\n[{time.time()-t0:.1f}s] Saved {save_path}")
    print(f"  latent_embeddings: {latent_norm.shape} (unit-norm)")


if __name__ == '__main__':
    main()
