import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from dsl.convert import SoftmaxConvert, SoftmaxConvertBias, Gumbel
from dsl.backbone import DIT, DITS, DITY, BERT, ModernBERT
CONVERTS = {'softmax': SoftmaxConvert, 'softmaxbias': SoftmaxConvertBias, 'gumbel': Gumbel}
BACKBONES = {'dit': DIT, 'dit-sahoo': DITS, 'bert': BERT, 'dity': DITY, 'modernbert': ModernBERT}


def build_dsl_from_cfg(cfg, vocab_size: int):
    d_backbone = cfg.backbone.dim_h                      # transformer hidden size (output dim of convert)
    noisy_dim  = cfg.data.dim_embed                     # Noisy embedding is in this dim space

    # Shared unit norm embedding used by main class to get noisy embeddings, and maybe by convert
    token_embed = nn.Embedding(vocab_size, noisy_dim)
    token_embed = weight_norm(token_embed, name='weight', dim=0)
    with torch.no_grad():
        # Reparameterize to direction (weight_v) and norm (weight_g); fix norm to 1.0
        token_embed.weight_g.fill_(1.)
        token_embed.weight_g.requires_grad_(False)

    # Build backbone, converter, then model
    backbone = BACKBONES[cfg.backbone.name](cfg, vocab_size)  # Get backbone and init
    convert  = CONVERTS[cfg.convert.name](d_backbone, vocab_size, token_embed)  # Get convert and init

    model = DSL(embed=token_embed, convert=convert, backbone=backbone, cfg=cfg)
    return model


class DSL(nn.Module):
    def __init__(self, embed, convert, backbone, **_):
        super().__init__()
        self.embed = embed  # Trainable, but fixed-norm token embedding (vocab_size, noisy_dim)
        self.backbone = backbone  # Takes noisy input and outputs logits (B, T, vocab_size)
        self.convert = convert  # Converts noisy embedding (B, T, noisy_dim) to backbone input (B, T, d_model)
        self.vocab_size, self.noise_dim = embed.num_embeddings, embed.embedding_dim

    def norm_weight_like(self, ref: torch.Tensor) -> torch.Tensor:
        """Embedding weight on same device/dtype as `ref`."""
        W = self.embed.weight
        if W.device != ref.device or W.dtype != ref.dtype:
            W = W.to(device=ref.device, dtype=ref.dtype, non_blocking=True)
        return W

    def forward(self, z):
        """Noisy embeddings in, logits over discrete tokens out."""
        mags = z.norm(dim=-1)
        h = self.convert(z)      # (B, T, d==backbone hidden size)
        return self.backbone(h, mags)  # Backbone produces logits (B, T, vocab_size)

    def x_hat(self, z):
        logits = self.forward(z)
        p = logits.softmax(dim=-1).float()
        x_hat = torch.matmul(p, self.norm_weight_like(p))
        return x_hat

    def noisy_embedding(self, input_ids: torch.Tensor, snr):
        """Take input token ids, embed, then add noise with desired SNR.
          - snr can be scalar, (B,), or (B, L)
          - output uses: z_noisy = snr * z + sqrt(snr) * eps
        Returns:
          z_noisy: (B, L, D)
        """
        z = self.embed(input_ids)  # clean embedding z (B, L, D)
        B, L, D = z.shape

        if not torch.is_tensor(snr):
            snr = torch.tensor(snr, device=z.device, dtype=torch.float32)

        if snr.dim() == 0:  # scalar
            t_used = snr.view(1, 1, 1).expand(B, L, 1)  # (B,L,1)
        elif snr.dim() == 1:  # (B,)
            t_used = snr.view(B, 1, 1).expand(B, L, 1)  # (B,L,1)
        elif snr.dim() == 2:  # (B, L)
            t_used = snr.view(B, L, 1)  # (B,L,1)
        else:
            raise ValueError(f"t must be scalar, (B,), or (B,L); got shape {tuple(snr.shape)}")

        eps = torch.randn_like(z)
        z_noisy = t_used * z + torch.sqrt(torch.clamp(t_used, min=0.0)) * eps
        return z_noisy