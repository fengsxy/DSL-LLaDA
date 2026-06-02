"""DSL-style modules for LLaDA-8B finetuning.

Contains:
- SoftmaxConvertBias: converts noisy embeddings to backbone input space
- sample_mixed_snr: samples mixed ROAR + LogNormal SNRs
- noisy_embedding: adds noise to clean embeddings at a given SNR
- DSLLaDA: wrapper combining LLaDA with DSL-style noisy embedding
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SoftmaxConvertBias
# ---------------------------------------------------------------------------

class SoftmaxConvertBias(nn.Module):
    def __init__(self, d_backbone, vocab_size, embed):
        """
        Convert noisy embedding z (B, T, z_dim) to backbone input (B, T, d_backbone)
        using softmax attention over V+1 slots (V tokens + 1 mask slot),
        then a trainable linear (V+1 -> d_backbone).
        """
        super().__init__()
        self.embed = embed  # (V, d_noise) The embedding where noise is added
        V, d_noise = embed.weight.shape
        assert vocab_size == V, f"vocab_size {vocab_size} != embed.weight.shape[0] {V}"

        # goes from V+1 -> dim of transformer backbone, learned
        # bias=True is critical: without it the two embedding spaces lack a degree of
        # freedom to align (some tokens won't train properly)
        self.backbone_embedding = nn.Linear(vocab_size + 1, d_backbone, bias=True)

        # Scaling: need beta*SNR > log(V)≈11.75 for softmax to be discriminative.
        # With typical LogNormal SNR~4, beta=5 gives 20 → P(correct)≈99.97%.
        # Init at 5.0 (not 1/√d=0.088): lr=2e-5 can't learn from small init, but can fine-tune around 5.
        beta_init = float(os.environ.get('DSL_BETA_INIT', '5.0'))
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # Learnable bias over logits (favor extra slot initially)
        self.logit_bias = nn.Parameter(torch.zeros(V + 1))
        with torch.no_grad():
            self.logit_bias[-1] = math.log(V)  # lean toward +1 slot with prob 1/2, representing mask

    def get_token_probs(self, z):
        """Convert noisy z into probability of different tokens, with bias toward mask if z is too noisy.

        All computation is forced to float32: the 128k-dim softmax is numerically
        fragile in bf16 (only ~3 decimal digits of precision → logit ordering errors
        corrupt the probability distribution before it even reaches the backbone).
        XDLM doesn't have this problem because it uses discrete tokens, no softmax converter.
        """
        B, T, d_noise = z.shape
        device = z.device

        # Force float32 for the entire softmax computation
        z_f32 = z.float()
        embed_w = self.embed.weight.to(device=device, dtype=torch.float32, non_blocking=True)
        zeros_row = torch.zeros(1, d_noise, dtype=torch.float32, device=device)
        K = torch.cat([embed_w, zeros_row], dim=0)  # (V+1, z_dim)

        logits = self.beta.float() * (z_f32 @ K.transpose(0, 1))
        logits = logits + self.logit_bias.to(device=device, dtype=torch.float32)
        probs = F.softmax(logits, dim=-1)  # (B, T, V+1) in float32
        return probs

    def forward(self, z):
        """Use backbone embedding as values to convert probabilities to vector embedding for backbone."""
        probs = self.get_token_probs(z)  # (B, T, V+1) in float32 (numerically stable)
        # Cast to backbone_embedding weight dtype (bf16 in mixed-precision training)
        return self.backbone_embedding(probs.to(self.backbone_embedding.weight.dtype))  # (B, T, d_backbone)


# ---------------------------------------------------------------------------
# SNR sampling
# ---------------------------------------------------------------------------

def sample_mixed_snr(
    batch_size,
    seq_len,
    device,
    mu=None,
    sigma=None,
    snr_max=100.0,
    snr_max_ln=None,
    roar_frac=10,
):
    # Defaults: mu=1.69, sigma=0.9 → median SNR ≈ 5.4, 66% in learning sweet-spot SNR 2-15.
    # ROAR handles extremes; continuous LogNormal focuses on mid-range where backbone learns.
    if mu is None:
        mu = float(os.environ.get('DSL_SNR_MU', '1.69'))
    if sigma is None:
        sigma = float(os.environ.get('DSL_SNR_SIGMA', '0.9'))
    if snr_max_ln is None:
        snr_max_ln = float(os.environ.get('DSL_SNR_MAX_LN', '40.0'))
    """Sample mixed SNRs: 1/roar_frac ROAR (smoothed per-token) + rest LogNormal.

    Each sample is independently assigned ROAR with probability 1/roar_frac,
    so this works even with batch_size=1 (probabilistic, not integer split).

      - ROAR "masked" tokens:   SNR ~ Uniform[0, 1)
      - ROAR "unmasked" tokens: SNR ~ Uniform[0.8*snr_max, snr_max]
      - LogNormal tokens:       SNR ~ LogNormal(mu, sigma), clamped to [0, snr_max_ln], per-sample scalar

    Args:
        roar_frac: denominator — 1/roar_frac probability per sample (default 10 → 10%)

    Returns:
        snrs: (batch_size, seq_len) float32 tensor
    """
    # Per-sample probabilistic ROAR assignment (works with any batch_size)
    is_roar = torch.rand(batch_size, device=device) < (1.0 / roar_frac)
    n_roar = is_roar.sum().item()
    n_logn = batch_size - n_roar

    snrs = torch.empty(batch_size, seq_len, device=device)

    # --- Smoothed ROAR ---
    if n_roar > 0:
        unmask_sizes = torch.randint(0, seq_len, (n_roar,), device=device)
        rand_order = torch.argsort(torch.rand(n_roar, seq_len, device=device), dim=1)
        is_masked = rand_order >= unmask_sizes.unsqueeze(1)

        low  = torch.rand(n_roar, seq_len, device=device)                              # [0, 1)
        high = snr_max * (0.8 + 0.2 * torch.rand(n_roar, seq_len, device=device))     # [0.8*snr_max, snr_max]
        snrs[is_roar] = torch.where(is_masked, low, high)

    # --- LogNormal (per-sample scalar, broadcast) ---
    if n_logn > 0:
        log_snrs = torch.randn(n_logn, device=device) * sigma + mu
        snrs[~is_roar] = log_snrs.exp().clamp(max=snr_max_ln).unsqueeze(1).expand(n_logn, seq_len)

    return snrs  # (batch_size, seq_len)


# ---------------------------------------------------------------------------
# Noisy embedding
# ---------------------------------------------------------------------------

def noisy_embedding(embed, input_ids, snr):
    """Embed input_ids and add noise at the given SNR level.

    Formula: z_noisy = snr * z + sqrt(snr) * eps

    Args:
        embed: nn.Embedding (or weight-normed equivalent) with .weight shape (V, D)
        input_ids: (B, L) integer token ids
        snr: scalar, (B,), or (B, L) SNR values

    Returns:
        z_noisy: (B, L, D) noisy embeddings
    """
    z = embed(input_ids)  # clean embedding (B, L, D)
    B, L, D = z.shape

    if not torch.is_tensor(snr):
        snr = torch.tensor(snr, device=z.device, dtype=torch.float32)

    if snr.dim() == 0:  # scalar
        t_used = snr.view(1, 1, 1).expand(B, L, 1)
    elif snr.dim() == 1:  # (B,)
        t_used = snr.view(B, 1, 1).expand(B, L, 1)
    elif snr.dim() == 2:  # (B, L)
        t_used = snr.view(B, L, 1)
    else:
        raise ValueError(f"snr must be scalar, (B,), or (B,L); got shape {tuple(snr.shape)}")

    # Keep t_used in float32: z_noisy = float32 * bf16 → float32, giving better numerical
    # stability for the V+1=126k softmax attention in SoftmaxConvertBias (matches dsl2 reference).
    t_used = t_used.to(dtype=torch.float32)
    eps = torch.randn(B, L, D, dtype=torch.float32, device=z.device)
    z_noisy = t_used * z.float() + torch.sqrt(torch.clamp(t_used, min=0.0)) * eps
    return z_noisy


# ---------------------------------------------------------------------------
# LoRA for ff_out
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Lightweight LoRA wrapper: keeps base frozen, trains low-rank A·B."""

    def __init__(self, base_linear, r=64, alpha=128):
        super().__init__()
        self.base = base_linear
        # Freeze base weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(d_in, r) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        self.scale = alpha / r

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scale
        return base_out + lora_out


# ---------------------------------------------------------------------------
# Attach DSL modules to an existing LLaDA model (no wrapper)
# ---------------------------------------------------------------------------

LOG2 = math.log(2)
MASK_TOKEN_ID = 126336
NOISE_DIM = int(os.environ.get('DSL_NOISE_DIM', '48'))
NOISE_INIT = os.environ.get('DSL_NOISE_INIT', 'random')  # 'random', 'pca', or 'ae_contrastive'
AE_EMBED_PATH = os.environ.get('DSL_AE_EMBED_PATH', 'results/wte_ae_contrastive_embedding.pt')


def _make_pca_noise_embed(wte_weight, noise_dim):
    """Create noise embedding from PCA of wte — preserves semantic structure.

    Tokens that are semantically similar in wte (4096-dim) stay close in the
    noise_embed (noise_dim), so adding Gaussian noise drifts through semantic
    neighbors rather than random tokens.
    """
    V, d_backbone = wte_weight.shape
    wte_f = wte_weight.detach().float()
    mean = wte_f.mean(dim=0, keepdim=True)
    wte_centered = wte_f - mean
    U, S, Vh = torch.pca_lowrank(wte_centered, q=noise_dim)
    # U: (V, noise_dim) — projection onto top principal components
    U_norm = F.normalize(U, dim=-1)

    total_var = (wte_centered ** 2).sum()
    explained_var = (S[:noise_dim] ** 2).sum()
    pct = explained_var / total_var * 100
    print(f"  PCA noise_embed: {noise_dim}d, variance explained = {pct:.1f}%")

    embed = nn.Embedding(V, noise_dim)
    with torch.no_grad():
        embed.weight.data = U_norm
        embed.weight.requires_grad_(False)
    return embed


def attach_dsl_modules(model, noise_dim=NOISE_DIM, mask_token_id=MASK_TOKEN_ID,
                       freeze_ff_out=True, noise_init=NOISE_INIT):
    """Attach DSL components (noise_embed, converter) directly to a LLaDA model.

    This avoids wrapping the model in a separate nn.Module, so it stays a
    PreTrainedModel with full HF Trainer compatibility (gradient checkpointing,
    save/load, LoRA, etc.) and identical DeepSpeed parameter partitioning.

    Args:
        noise_init: 'random' (default, DSL standard) or 'pca' (PCA of wte,
                    preserves semantic structure so noise drifts through
                    semantic neighbors instead of random tokens).
    """
    wte_weight = model.model.transformer.wte.weight  # (V, d_backbone)
    vocab_size, d_backbone = wte_weight.shape

    # Frozen noise embedding (unit-norm)
    if noise_init == 'ae_contrastive':
        ae_path = os.environ.get('DSL_AE_EMBED_PATH', 'results/wte_ae_contrastive_embedding.pt')
        print(f"  Loading AE+Contrastive noise embedding from {ae_path}")
        ae_data = torch.load(ae_path, map_location='cpu', weights_only=True)
        ae_emb = ae_data['latent_embeddings'].float()  # (V, d_latent), already unit-norm
        ae_dim = ae_emb.shape[1]
        if ae_dim != noise_dim:
            print(f"  WARNING: AE latent dim={ae_dim} != noise_dim={noise_dim}, using ae_dim={ae_dim}")
            noise_dim = ae_dim
        noise_embed = nn.Embedding(vocab_size, noise_dim)
        with torch.no_grad():
            noise_embed.weight.data[:ae_emb.shape[0]] = ae_emb[:vocab_size]
            noise_embed.weight.requires_grad_(False)
        print(f"  AE+Contrastive noise_embed: {noise_embed.weight.shape}, frozen")
    elif noise_init == 'pca':
        noise_embed = _make_pca_noise_embed(wte_weight, noise_dim)
    else:
        # Standard: random unit-norm directions
        noise_embed = nn.Embedding(vocab_size, noise_dim)
        with torch.no_grad():
            noise_embed.weight.data = F.normalize(noise_embed.weight.data, dim=-1)
            noise_embed.weight.requires_grad_(False)
    model.noise_embed = noise_embed

    # Converter: noisy embeddings -> backbone space
    converter = SoftmaxConvertBias(d_backbone, vocab_size, noise_embed)
    bbemb_init = os.environ.get('DSL_BBEMB_INIT', 'wte')  # 'wte' or 'random'
    if bbemb_init == 'random':
        # Random init (Xavier) — forces converter to learn from scratch
        nn.init.xavier_normal_(converter.backbone_embedding.weight)
        nn.init.zeros_(converter.backbone_embedding.bias)
        print(f"  backbone_embedding: RANDOM init (Xavier)")
    else:
        with torch.no_grad():
            converter.backbone_embedding.weight[:, :vocab_size] = wte_weight.T
            converter.backbone_embedding.weight[:, vocab_size] = wte_weight[mask_token_id]
        print(f"  backbone_embedding: wte.T init")
    # Optionally freeze entire converter
    freeze_converter = os.environ.get('DSL_FREEZE_CONVERTER', '0') == '1'
    if freeze_converter:
        for p in converter.parameters():
            p.requires_grad_(False)
        print(f"  converter: FROZEN (all params)")
    model.converter = converter

    # Learnable log(snr_max) for ROAR and LogNormal clamping
    # snr_max = exp(log_snr_max), init from env var (default: log(100)≈4.6)
    init_snr_max = float(os.environ.get('DSL_SNR_MAX', '100.0'))
    learnable_snr_max = os.environ.get('DSL_LEARNABLE_SNR_MAX', '0') == '1'
    model.log_snr_max = nn.Parameter(
        torch.tensor(math.log(init_snr_max)),
        requires_grad=learnable_snr_max,
    )

    # ff_out (output projection, separate from wte since weight_tying=false).
    # Full-parameter training of ff_out with bf16+ZeRO-2 causes NaN (DeepSpeed
    # buffer init issue). Options: freeze entirely, or wrap with LoRA.
    ffout_lora_r = int(os.environ.get('DSL_FFOUT_LORA_R', '0'))
    if ffout_lora_r > 0 and hasattr(model.model.transformer, 'ff_out'):
        # LoRA: base weights frozen, train low-rank adapter
        lora = LoRALinear(model.model.transformer.ff_out, r=ffout_lora_r)
        model.model.transformer.ff_out = lora
    elif freeze_ff_out and hasattr(model.model.transformer, 'ff_out'):
        model.model.transformer.ff_out.weight.requires_grad_(False)
        if model.model.transformer.ff_out.bias is not None:
            model.model.transformer.ff_out.bias.requires_grad_(False)

    return model


@torch.no_grad()
def nll_integral_quick(model, input_ids, int_steps=20, snr_max=30.0):
    """Lightweight NLL integral bound for training monitoring.

    Computes DSL NLL upper bound (bits/token) using trapezoidal integration
    over a linear SNR grid. Designed to be cheap enough to call every ~50 steps.

    Returns dict with: nll_bpt, nll_diff_bpt, nll_recon_bpt, best_snr
    """
    B, T = input_ids.shape
    device = input_ids.device
    vocab_size = model.noise_embed.weight.shape[0]

    snr_max_scaled = snr_max * math.log(vocab_size) / math.log(27)
    snrs = torch.linspace(0., snr_max_scaled, int_steps, device=device)
    if snrs[0].item() != 0.0:
        snrs = torch.cat([torch.zeros(1, dtype=snrs.dtype, device=device), snrs])
    K = snrs.size(0)

    x_embed = model.noise_embed(input_ids).float()  # (B, T, noise_dim)
    embed_weight = model.noise_embed.weight.float()  # (V, noise_dim)

    nll_integrand = torch.zeros(B, K)
    nll_reconst = torch.zeros(B, K)

    for i, snr_val in enumerate(snrs):
        snr_t = torch.full((B, T), snr_val.item(), device=device)
        z = noisy_embedding(model.noise_embed, input_ids, snr_t)
        h = model.converter(z)
        # Match backbone dtype (bf16 in production, float32 in tests)
        wte_dtype = model.model.transformer.wte.weight.dtype
        logits = model(input_ids=input_ids, inputs_embeds=h.to(dtype=wte_dtype)).logits

        # Recon CE
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            input_ids.view(-1), reduction='none'
        ).view(B, T).mean(dim=-1)
        nll_reconst[:, i] = (ce / LOG2).cpu()

        # Diffusion MSE: x_hat = softmax(logits) @ embed_weight
        p = F.softmax(logits.float(), dim=-1)
        x_hat = torch.matmul(p, embed_weight.to(p.device))
        mse = F.mse_loss(x_hat, x_embed, reduction='none').sum(dim=-1).mean(dim=-1)
        nll_integrand[:, i] = (0.5 * mse / LOG2).cpu()

    # Trapezoidal integration
    deltas = (snrs[1:] - snrs[:-1]).cpu()
    avg_vals = 0.5 * (nll_integrand[:, :-1] + nll_integrand[:, 1:])
    nll_diffusion = torch.cumsum(avg_vals * deltas.unsqueeze(0), dim=1)
    nll_total = nll_diffusion + nll_reconst[:, 1:]

    best_idx = nll_total.argmin(dim=1)
    all_b = torch.arange(B)
    best_snr = snrs[1:][best_idx]

    return {
        'nll_bpt': nll_total[all_b, best_idx].mean().item(),
        'nll_diff_bpt': nll_diffusion[all_b, best_idx].mean().item(),
        'nll_recon_bpt': nll_reconst[:, 1:][all_b, best_idx].mean().item(),
        'best_snr': best_snr.mean().item(),
    }


def embed_health_metrics(model):
    """Compute diagnostics on noise embedding and converter embedding quality.

    Monitors for identity collapse: if embeddings become identical (all rows
    converge to the same vector), avg_cosine_sim → 1.0 and effective_rank → 1.
    Healthy embeddings: avg_cosine_sim ≈ 0, effective_rank ≈ noise_dim.
    """
    metrics = {}

    # --- noise_embed (frozen, shouldn't change but verify) ---
    W = model.noise_embed.weight  # (V, noise_dim)
    idx = torch.randperm(W.shape[0], device=W.device)[:1000]
    W_sample = F.normalize(W[idx].float(), dim=-1)
    cos_sim = (W_sample @ W_sample.T).fill_diagonal_(0)
    metrics["embed/avg_cosine_sim"] = cos_sim.abs().mean().item()
    _, s, _ = torch.linalg.svd(W_sample[:256].float(), full_matrices=False)
    s = s / s.sum()
    metrics["embed/effective_rank"] = torch.exp(-(s * torch.log(s + 1e-10)).sum()).item()

    # --- converter backbone_embedding (trainable, critical to monitor) ---
    if hasattr(model, 'converter'):
        # backbone_embedding.weight is (d_backbone, V+1) — each column is a token's embedding
        be_W = model.converter.backbone_embedding.weight.float()  # (d_backbone, V+1)
        # Sample 1000 random token columns and compute pairwise cosine similarity
        n_tokens = be_W.shape[1]
        cidx = torch.randperm(n_tokens, device=be_W.device)[:1000]
        cols = F.normalize(be_W[:, cidx].T, dim=-1)  # (1000, d_backbone)
        be_cos = (cols @ cols.T).fill_diagonal_(0)
        metrics["converter/avg_cosine_sim"] = be_cos.abs().mean().item()
        # Effective rank of converter embedding
        _, s_be, _ = torch.linalg.svd(cols[:256].float(), full_matrices=False)
        s_be = s_be / s_be.sum()
        metrics["converter/effective_rank"] = torch.exp(-(s_be * torch.log(s_be + 1e-10)).sum()).item()

    return metrics


# ---------------------------------------------------------------------------
# DSLLaDA wrapper (kept for standalone testing / sanity checks only)
# ---------------------------------------------------------------------------

class DSLLaDA(nn.Module):
    """DSL-style wrapper around LLaDA — used only for CPU sanity checks.
    Production training uses attach_dsl_modules() instead.
    """

    MASK_TOKEN_ID = MASK_TOKEN_ID
    NOISE_DIM = NOISE_DIM

    def __init__(self, llada_model):
        super().__init__()
        self.llada = llada_model
        attach_dsl_modules(llada_model, self.NOISE_DIM, self.MASK_TOKEN_ID)
        # Alias for backward compat with sanity checks
        self.noise_embed = llada_model.noise_embed
        self.converter = llada_model.converter

    def forward(self, input_ids, snrs=None, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        if snrs is None:
            snrs = sample_mixed_snr(B, L, input_ids.device)
        z_noisy = noisy_embedding(self.noise_embed, input_ids, snrs)
        h = self.converter(z_noisy)
        return self.llada(input_ids=input_ids, inputs_embeds=h,
                          attention_mask=attention_mask, **kwargs)
