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

        # Beta controls softmax sharpness. Two regimes:
        # - beta=5.0: converter locks onto correct token at SNR>2 (original setting)
        # - beta=1/√d: smooth distribution, matches dsl2_ref (recommended for DSL benefits)
        # Configurable via DSL_BETA_INIT env var.
        beta_init = float(os.environ.get('DSL_BETA_INIT', '5.0'))
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # Learnable bias over logits (favor extra slot initially)
        self.logit_bias = nn.Parameter(torch.zeros(V + 1))
        with torch.no_grad():
            self.logit_bias[-1] = math.log(V)  # lean toward +1 slot with prob 1/2, representing mask

        # Residual projection: bypass softmax to pass high-frequency info directly to backbone.
        # The softmax converter acts as a low-pass filter (soft retrieval over tokens).
        # This residual path preserves continuous/high-freq details from z for the transformer.
        # DSL_RESIDUAL=1: always-on high-pass residual
        # DSL_RESIDUAL=2: SNR-gated high-pass (on when uncertain, off when confident)
        residual_mode = os.environ.get('DSL_RESIDUAL', '0')
        self.use_residual = residual_mode in ('1', '2')
        self.snr_gated = residual_mode == '2'
        if self.use_residual:
            self.residual_proj = nn.Linear(d_noise, d_backbone, bias=True)
            nn.init.normal_(self.residual_proj.weight, std=0.01)
            nn.init.zeros_(self.residual_proj.bias)
        if self.snr_gated:
            # Learnable gate temperature and threshold
            # gate = sigmoid(-snr / temp + bias)
            # temp controls transition sharpness, bias controls threshold SNR
            self.gate_temp = nn.Parameter(torch.tensor(5.0))   # transition width
            self.gate_bias = nn.Parameter(torch.tensor(2.0))   # ~sigmoid(2)=0.88 at SNR=0

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
        # Stop-grad on K matrix only when DSL_OUTPUT_EMBED is active.
        # Otherwise, let gradients flow normally for e2e training.
        use_detach = os.environ.get('DSL_OUTPUT_EMBED', '0') == '1'
        if hasattr(self, '_embed_weight_override') and self._embed_weight_override is not None:
            embed_w = self._embed_weight_override.detach().to(device=device, dtype=torch.float32) if use_detach else self._embed_weight_override.to(device=device, dtype=torch.float32)
        else:
            embed_w = self.embed.weight.detach().to(device=device, dtype=torch.float32) if use_detach else self.embed.weight.to(device=device, dtype=torch.float32, non_blocking=True)
        zeros_row = torch.zeros(1, d_noise, dtype=torch.float32, device=device)
        K = torch.cat([embed_w, zeros_row], dim=0)  # (V+1, z_dim)

        logits = self.beta.float() * (z_f32 @ K.transpose(0, 1))
        logits = logits + self.logit_bias.to(device=device, dtype=torch.float32)
        probs = F.softmax(logits, dim=-1)  # (B, T, V+1) in float32
        return probs

    def forward(self, z, snr=None):
        """Use backbone embedding as values to convert probabilities to vector embedding for backbone.
        Args:
            z: noisy embedding (B, T, d_noise)
            snr: per-token SNR (B, T) or None. Used for SNR-gated residual (DSL_RESIDUAL=2).
        """
        probs = self.get_token_probs(z)  # (B, T, V+1) in float32 (numerically stable)
        # When backbone_embedding is trained via separate fp32 optimizer (bypassing
        # DeepSpeed to avoid ZeRO-2 partition NaN), use the fp32 weight/bias override
        # so gradients flow to the separate optimizer's parameters.
        if hasattr(self, '_bbemb_weight_override') and self._bbemb_weight_override is not None:
            w = self._bbemb_weight_override
            b = getattr(self, '_bbemb_bias_override', None)
            h_low = F.linear(probs.to(w.dtype), w, b)
        else:
            # Cast to backbone_embedding weight dtype (bf16 in mixed-precision training)
            h_low = self.backbone_embedding(probs.to(self.backbone_embedding.weight.dtype))  # (B, T, d_backbone)

        # Residual path: HIGH-PASS FILTER — orthogonal projection
        # Remove the soft-retrieval direction from z, keep only the perpendicular component.
        if self.use_residual:
            embed_w = self.embed.weight.to(z.dtype)  # (V, d_noise)
            z_low = probs[:, :, :embed_w.shape[0]].to(z.dtype) @ embed_w  # (B, T, d_noise)
            z_low_norm = z_low / (z_low.norm(dim=-1, keepdim=True) + 1e-8)
            proj_scalar = (z * z_low_norm).sum(dim=-1, keepdim=True)
            z_high = z - proj_scalar * z_low_norm  # high-freq only
            h_high = self.residual_proj(z_high.to(self.residual_proj.weight.dtype))

            if self.snr_gated and snr is not None:
                # gate = sigmoid(-snr / temp + bias)
                # SNR高 → gate≈0（确定，不需要residual）
                # SNR低 → gate≈1（不确定，需要高频补充）
                gate = torch.sigmoid(-snr / self.gate_temp.abs().clamp(min=0.1) + self.gate_bias)
                gate = gate.unsqueeze(-1)  # (B, T, 1)
                return h_low + gate.to(h_high.dtype) * h_high
            return h_low + h_high
        return h_low


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
    roar_frac=None,
):
    """Sample mixed SNRs: 1/roar_frac ROAR (per-token) + rest LogNormal (per-token).

    Following dsl2_ref/dsls/snrs.py, ROAR uses per-token SNR (some tokens get high
    SNR as clear context, others get low SNR as prediction targets — like MASK training).

    LogNormal is also per-token: each token independently samples its own SNR from
    LogNormal(mu, sigma). This gives natural variation within each sequence — some
    tokens happen to get high SNR (clear context) and others low SNR (noisy targets).
    This is critical for 126k vocab where the converter has a sharp threshold:
    per-sample SNR puts ALL tokens on the same side of the threshold, destroying
    the context/target structure that the backbone needs for learning.

    Controlled by DSL_SNR_PER_TOKEN env var (default '1' = per-token).
    Set to '0' for legacy per-sample LogNormal behavior.

    Each sample is independently assigned ROAR with probability 1/roar_frac,
    so this works even with batch_size=1 (probabilistic, not integer split).

      - ROAR "masked" tokens:   SNR ~ Uniform[0, 1)
      - ROAR "unmasked" tokens: SNR ~ Uniform[0.8*snr_max, snr_max]
      - LogNormal tokens:       SNR ~ LogNormal(mu, sigma), clamped to [0, snr_max_ln]

    Args:
        roar_frac: denominator — 1/roar_frac probability per sample (default 10 → 10%)

    Returns:
        snrs: (batch_size, seq_len) float32 tensor
    """
    if mu is None:
        mu = float(os.environ.get('DSL_SNR_MU', '1.69'))
    if sigma is None:
        sigma = float(os.environ.get('DSL_SNR_SIGMA', '0.9'))
    if snr_max_ln is None:
        snr_max_ln = float(os.environ.get('DSL_SNR_MAX_LN', '40.0'))
    if roar_frac is None:
        roar_frac = int(os.environ.get('DSL_ROAR_FRAC', '10'))
    per_token = os.environ.get('DSL_SNR_PER_TOKEN', '1') == '1'

    # Per-sample probabilistic ROAR assignment (works with any batch_size)
    is_roar = torch.rand(batch_size, device=device) < (1.0 / roar_frac)
    n_roar = is_roar.sum().item()
    n_logn = batch_size - n_roar

    snrs = torch.empty(batch_size, seq_len, device=device)

    # --- Smoothed ROAR (always per-token, matching dsl2_ref) ---
    if n_roar > 0:
        unmask_sizes = torch.randint(0, seq_len, (n_roar,), device=device)
        rand_order = torch.argsort(torch.rand(n_roar, seq_len, device=device), dim=1)
        is_masked = rand_order >= unmask_sizes.unsqueeze(1)

        low  = torch.rand(n_roar, seq_len, device=device)                              # [0, 1)
        high = snr_max * (0.8 + 0.2 * torch.rand(n_roar, seq_len, device=device))     # [0.8*snr_max, snr_max]
        snrs[is_roar] = torch.where(is_masked, low, high)

    # --- LogNormal ---
    if n_logn > 0:
        if per_token:
            # Per-token: each token independently samples its own SNR.
            # This gives natural context/target variation within each sequence.
            log_snrs = torch.randn(n_logn, seq_len, device=device) * sigma + mu
            snrs[~is_roar] = log_snrs.exp().clamp(max=snr_max_ln)
        else:
            # Legacy per-sample: all tokens in a sequence share the same SNR.
            log_snrs = torch.randn(n_logn, device=device) * sigma + mu
            snrs[~is_roar] = log_snrs.exp().clamp(max=snr_max_ln).unsqueeze(1).expand(n_logn, seq_len)

    return snrs  # (batch_size, seq_len)


# ---------------------------------------------------------------------------
# Noisy embedding
# ---------------------------------------------------------------------------

def noisy_embedding(embed, input_ids, snr, weight_override=None):
    """Embed input_ids and add noise at the given SNR level.

    Formula: z_noisy = snr * z + sqrt(snr) * eps

    Args:
        embed: nn.Embedding (or weight-normed equivalent) with .weight shape (V, D)
        input_ids: (B, L) integer token ids
        snr: scalar, (B,), or (B, L) SNR values
        weight_override: optional tensor (V, D) to use instead of embed.weight
                         (for separate optimizer with grad tracking)

    Returns:
        z_noisy: (B, L, D) noisy embeddings
    """
    # Stop-grad on input side only when DSL_OUTPUT_EMBED is active.
    use_detach = os.environ.get('DSL_OUTPUT_EMBED', '0') == '1'
    if weight_override is not None:
        z = F.embedding(input_ids, weight_override.detach() if use_detach else weight_override)
    else:
        z = embed(input_ids).detach() if use_detach else embed(input_ids)
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
NOISE_INIT = os.environ.get('DSL_NOISE_INIT', 'random')  # 'random' or 'pca'


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

    # Noise embedding (unit-norm)
    freeze_embed = os.environ.get('DSL_FREEZE_EMBED', '1') != '0'
    if noise_init == 'pca':
        noise_embed = _make_pca_noise_embed(wte_weight, noise_dim)
    else:
        # Standard: random unit-norm directions
        # Use fixed seed so all ranks get identical init (critical for DDP/ZeRO)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        noise_embed = nn.Embedding(vocab_size, noise_dim)
        torch.random.set_rng_state(rng_state)
        with torch.no_grad():
            noise_embed.weight.data = F.normalize(noise_embed.weight.data, dim=-1)
    # Always freeze for DeepSpeed (avoids ZeRO partition NaN).
    # When DSL_FREEZE_EMBED=0, training is done via separate fp32 optimizer
    # managed by dllm_Trainer_DSL (same approach as ff_out).
    noise_embed.weight.requires_grad_(False)
    model.noise_embed = noise_embed
    model._dsl_train_embed = not freeze_embed  # flag for trainer

    # Converter: noisy embeddings -> backbone space
    converter = SoftmaxConvertBias(d_backbone, vocab_size, noise_embed)
    with torch.no_grad():
        converter.backbone_embedding.weight[:, :vocab_size] = wte_weight.T
        converter.backbone_embedding.weight[:, vocab_size] = wte_weight[mask_token_id]
    model.converter = converter

    # Output projection: when DSL_OUTPUT_EMBED=1, replace ff_out with
    # (hidden @ output_proj) @ E.t() so noise_embed learns from output side.
    # This is the stop-grad pattern: input side uses E.detach(), output side uses E.
    use_output_embed = os.environ.get('DSL_OUTPUT_EMBED', '0') == '1'
    if use_output_embed:
        output_proj = nn.Linear(d_backbone, noise_dim, bias=False)
        # Initialize from PCA of wte (project backbone space to noise space)
        with torch.no_grad():
            # Use SVD to find best projection from d_backbone→noise_dim
            U, S, Vh = torch.linalg.svd(wte_weight.float(), full_matrices=False)
            # Vh[:noise_dim] is the top-noise_dim right singular vectors
            output_proj.weight.data = Vh[:noise_dim].to(wte_weight.dtype)
        # Don't register output_proj on model (avoids DeepSpeed hang).
        # Store init weight as plain tensor; trainer will create fp32 copy.
        model._output_proj_init_weight = output_proj.weight.data.clone().cpu()
        model._dsl_output_embed = True
        # When using output embed, always freeze ff_out (it's replaced)
        if hasattr(model.model.transformer, 'ff_out'):
            model.model.transformer.ff_out.weight.requires_grad_(False)
            if model.model.transformer.ff_out.bias is not None:
                model.model.transformer.ff_out.bias.requires_grad_(False)
        freeze_ff_out = True  # override
        if int(os.environ.get('LOCAL_RANK', '-1')) in (-1, 0):
            print(f"  [DSL] Output embed enabled: output_proj {d_backbone}→{noise_dim}, "
                  f"logits = (h @ W_proj) @ E.t()", flush=True)
    else:
        model._dsl_output_embed = False

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


def embed_regularization_loss(noise_embed, lambda_rank=0.01, n_sample=512, weight_override=None):
    """Anti-collapse regularization via pairwise cosine similarity.

    Penalizes high pairwise cosine similarity between embedding rows,
    preventing collapse to a low-dimensional subspace. More numerically
    stable than SVD-based effective rank (avoids convergence failures
    with ill-conditioned matrices under ZeRO partitioning).

    Args:
        noise_embed: nn.Embedding with unit-norm rows
        lambda_rank: regularization strength (default 0.01)
        n_sample: number of rows to sample for pairwise computation (default 512)
        weight_override: optional tensor to use instead of noise_embed.weight

    Returns:
        loss_reg: scalar tensor (lambda * mean_squared_cosine_sim)
    """
    W = weight_override if weight_override is not None else noise_embed.weight  # (V, d)
    V = W.shape[0]
    # Sample subset for efficiency: O(n_sample^2) not O(V^2)
    if V > n_sample:
        idx = torch.randperm(V, device=W.device)[:n_sample]
        W_sub = W[idx]
    else:
        W_sub = W
    # Normalize (should already be unit-norm, but ensure for gradient flow)
    W_norm = F.normalize(W_sub.float(), dim=-1)
    # Pairwise cosine similarity matrix
    cos_mat = W_norm @ W_norm.T  # (n, n)
    # Zero out diagonal (self-similarity = 1, not informative)
    n = cos_mat.shape[0]
    mask = ~torch.eye(n, device=cos_mat.device, dtype=torch.bool)
    # Mean squared cosine similarity (penalizes any high similarity)
    loss_reg = lambda_rank * (cos_mat[mask] ** 2).mean()
    return loss_reg


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
