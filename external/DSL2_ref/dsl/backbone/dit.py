import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# function overload
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        # 添加检查
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError(f"Invalid timestep values detected: {t}")
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# NOTE: for t of shape = (batch, seq_length), a per-token timestep embedder
class TimestepEmbedder_pertoken(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """Generate time embedding, where timestep is of shape (batch, seq_length), per-token noise level.
        :param hidden_size: time embedding size
        :param frequency_embedding_size: frequency for sinusoidal embedding generation,
                Yunshu: as I remember this has a lower bound to guarantee the learned transformer to have orth latent space
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Creat sinusoidal timestep embeddings.
        :param t: a Tensor of shape (batch, seq_length), one per token. NOTE different from classic TimeEmbedder which is per-sample noise level t
        :param dim: the dimension of the output
        :param max_period: controls the minimum frequency  of the embeddings
        :return: a Tensor of shpae (batch, seq_length, dim), positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        # expand dims to handel the addtional seq_length dimension
        args = t[..., None].float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[..., :1])], dim=-1)
        # print(f'\ntime embedding = {embedding.shape}, should be ({t.shape[0]}, {t.shape[1]}, {dim})\n')
        return embedding

    def forward(self, t):
        """
        :param t: input tensor of shape (batch, seq_length)
        :return: timestep embeddings of shape (batch, seq_length, t_dim)
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # flatten the first two dimensions to pass through MLP
        batch, seq_length, _ = t_freq.shape
        t_freq_flat = t_freq.view(batch * seq_length, -1)
        t_emb_flat = self.mlp(t_freq_flat)
        # reshape back to (batch, seq_length, t_dim)
        t_emb = t_emb_flat.view(batch, seq_length, -1)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, bias=None, seqlens=None):
        """
        :param x: batch of sentences, tensor of shape (batch, seq_length, d)
        :param rotary_cos_sin:
        :param c: batch of conditions, tensor of shape (batch, seq_length),
        NOTE the current method is very computational and memory heavy, modify the DiT conditioning to enable efficient training / sampling.
        """
        # print(f'\n')
        # print(f'DiT Block, x = {x.shape}')
        # print(f'DiT Block, c = {c.shape}')
        # print(f'DiT Block, adaLN output = {self.adaLN_modulation(c).shape}')

        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # NOTE Original adaLN, for per-image noise level, where we use timestep as condition c
        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # # NOTE: shift_msa and other conditions should be of shape (batch, 1, d), since originally in image case uses per-image noise level t
        # # TODO: Now I need to modify this to a per-token noise level
        # (shift_msa, scale_msa, gate_msa, shift_mlp,
        #  scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=2)
        # print(f'DiT Block, shift_msa = {shift_msa.shape}, scale_msa = {scale_msa.shape} ...')

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv,
                        'b s (three h d) -> b s three h d',
                        three=3,
                        h=self.n_heads)
        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)
        if bias is not None:
            ones = torch.ones_like(qkv[..., :1])
            qkv = torch.cat((qkv, ones), dim=-1)
            bias = rearrange(bias, 'b s -> (b s) ()').to(qkv.dtype)
            qkv[:, 1, :, -1] = bias  # add bias to k, in added embedding dimension
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)
        if bias is not None:
            x = x[..., :-1]  # remove added bias dimension

        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x),
                                  None,
                                  gate_msa,
                                  x_skip,
                                  self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(
                self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim,
                                          2 * hidden_size,
                                          bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                                 Gate Model                                    #
#################################################################################
class GatingNet(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

        with torch.no_grad():
            self.linear.weight.zero_()  # set weight to zero
            self.linear.bias.fill_(-10.0)  # set bias to -10, so sigmoid(-10) ~ 1e-5

    def forward(self, t):
        x = self.linear(t)
        return torch.sigmoid(x)  # gating value between (0,1)


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size, mlp_ratio=4):
        super().__init__()
        self.config = config

        # ----- read from new config layout -----
        # backbone hyperparams live under cfg.backbone
        bb = config.backbone
        # data bits (vocab_size) live under cfg.data
        self.vocab_size = vocab_size

        # method flags
        self.dit_cond = bool(bb.dit_cond)
        self.dit_attn_bias = bool(bb.dit_attn_bias)

        dim_h   = int(bb.dim_h)
        n_heads = int(bb.n_heads)
        n_blocks = int(bb.n_blocks)
        cond_dim = int(bb.cond_dim)
        dropout  = float(bb.dropout)

        # ----- modules -----
        self.sigma_map = TimestepEmbedder(cond_dim)
        # If you later switch to per-token conditioning:
        # self.sigma_map = TimestepEmbedder_pertoken(cond_dim)

        self.rotary_emb = Rotary(dim_h // n_heads)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                DDiTBlock(
                    dim=dim_h,
                    n_heads=n_heads,
                    cond_dim=cond_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            hidden_size=dim_h,
            out_channels=self.vocab_size,
            cond_dim=cond_dim,
        )

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, z, noise_mags):
        mags = z.norm(dim=-1)
        # Input is sequence of vectors. "mags" is magnitude of noisy embedding
        assert mags.dim() == 2, "WRONG Magnitude Inputs..."

        # 更强的异常值处理
        if torch.isnan(mags).any() or torch.isinf(mags).any():
            print(f"Warning: mags contains NaN or Inf before clamp, replacing with safe values")
            mags = torch.nan_to_num(mags, nan=1.0, posinf=10.0, neginf=1e-4)
        # 确保合理范围
        mags = torch.clamp(mags, min=1e-4, max=10.0)
        # 检查输入 z
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(f"Warning: input z contains NaN or Inf values")
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        # c shape (batch, t_dim), assign per-sentence noise level t

        t_input = noise_mags.mean(dim=-1)

        if torch.isnan(t_input).any() or torch.isinf(t_input).any():
            print(f"Warning: t_input contains NaN or Inf values")
            t_input = torch.nan_to_num(t_input, nan=1.0, posinf=10.0, neginf=1e-4)
        c = F.silu(self.sigma_map(t_input))

        if not self.dit_cond:
            c = torch.zeros_like(c)
        if self.dit_attn_bias:
            bias = (-1.) / (mags + 1e-10)
        else:
            bias = None

        rotary_cos_sin = self.rotary_emb(z)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                z = self.blocks[i](z, rotary_cos_sin, c, bias=bias, seqlens=None)
        with torch.autocast(device_type='cuda', enabled=False):
            z = self.output_layer(z, c)
        return z


#################################################################################
#                             D I T  (per-token)                                #
#                         New variant: class DITY                                #
#################################################################################

class DDiTBlockY(nn.Module):
    """DDiT block variant that expects per-token conditioning `c` with shape (B, S, cond_dim).

    Matches the attention/MLP structure of DDiTBlock but *does not* insert an
    extra singleton sequence dimension for conditioning. This mirrors the code
    you provided for per-token conditioning.
    """
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, bias=None, seqlens=None):
        """
        x: (B, S, D)
        c: (B, S, cond_dim)  # per-token conditioning
        bias: optional (B, S) additive attention bias (larger -> more attention)
        """
        B, S, _ = x.shape
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Per-token modulation params (no extra singleton dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=2)
        )

        # Attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (t h d) -> b s t h d', t=3, h=self.n_heads)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')

        if seqlens is None:
            cu_seqlens = torch.arange(0, (B + 1) * S, step=S, dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)

        if bias is not None:
            ones = torch.ones_like(qkv[..., :1])
            qkv = torch.cat((qkv, ones), dim=-1)
            bias_flat = rearrange(bias, 'b s -> (b s) ()').to(qkv.dtype)
            qkv[:, 1, :, -1] = bias_flat  # add bias to k in the appended dim

        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, S, 0.0, causal=False
        )
        if bias is not None:
            x = x[..., :-1]  # drop the appended bias dim

        x = rearrange(x, '(b s) h d -> b s (h d)', b=B)
        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # MLP
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout
        )
        return x


class DDitFinalLayerY(nn.Module):
    """Final projection layer matching per-token conditioning shape (B, S, cond_dim)."""
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        # Per-token: no [:, None] broadcast
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        return self.linear(x)


class DITY(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """DiT-style backbone with **per-token** conditioning and the same external
    interface as `DIT`:

    __init__(cfg, vocab_size, mlp_ratio=4)
    forward(z, noise_mags)

    Interface details mirrored from `dsl/backbone/dit.py`:
    - Hyperparameters read from `cfg.backbone` (dim_h, n_heads, n_blocks, cond_dim, dropout).
    - Flags: `bb.dit_cond`, `bb.dit_attn_bias`.
    - `vocab_size` passed as an argument (not read from cfg.data).
    """

    def __init__(self, config, vocab_size: int, mlp_ratio: int = 4):
        super().__init__()
        self.config = config
        bb = config.backbone
        self.vocab_size = int(vocab_size)

        # Method flags
        self.dit_cond = bool(getattr(bb, 'dit_cond'))
        self.dit_attn_bias = bool(getattr(bb, 'dit_attn_bias'))

        dim_h   = int(getattr(bb, 'dim_h'))
        n_heads = int(getattr(bb, 'n_heads'))
        n_blocks = int(getattr(bb, 'n_blocks'))
        cond_dim = int(getattr(bb, 'cond_dim'))
        dropout  = float(getattr(bb, 'dropout'))

        # Per-token timestep embedder and rotary embeddings
        self.sigma_map = TimestepEmbedder_pertoken(cond_dim)
        self.rotary_emb = Rotary(dim_h // n_heads)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlockY(dim=dim_h, n_heads=n_heads, cond_dim=cond_dim,
                       mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.output_layer = DDitFinalLayerY(hidden_size=dim_h,
                                            out_channels=self.vocab_size,
                                            cond_dim=cond_dim)

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, z: torch.Tensor, noise_mags: torch.Tensor) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        z : (B, S, D)
            Noisy token embeddings.
        noise_mags : (B,) or (B, S)
            per-token conditioning. 1D inputs will be broadcast across sequence.
        """
        B, S, _ = z.shape

        if noise_mags.dim() == 1:
            t_input = noise_mags[:, None].expand(B, S)
        else:
            t_input = noise_mags

        # Build per-token condition embedding
        c = F.silu(self.sigma_map(t_input))  # (B, S, cond_dim)
        if not self.dit_cond:
            c = torch.zeros_like(c)

        # Optional per-token attention bias
        if self.dit_attn_bias:
            bias = (-1.0) / (noise_mags + 1e-10)  # (B, S)
        else:
            bias = None

        rotary_cos_sin = self.rotary_emb(z)

        # Main stack
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for blk in self.blocks:
                z = blk(z, rotary_cos_sin, c, bias=bias, seqlens=None)
            z = self.output_layer(z, c)

        return z
