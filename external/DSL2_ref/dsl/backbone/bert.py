"""
Simple BERT-style encoder backbone for CPU / Apple MPS.
- No CUDA-specific deps, no flash-attn, no hub mixin.
- Uses standard sinusoidal positional encoding (parameter-free).
- API compatible with previous backbone: class `DITS`, forward(x, sigma=None, ...).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Parameter-free sinusoidal positions (Vaswani et al. 2017).

    Precomputes a (max_len, d_model) table on the correct device/dtype when
    first used, then slices to sequence length and adds to inputs.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.register_buffer("pe", None, persistent=False)

    def _maybe_build(self, device, dtype):
        if self.pe is not None and self.pe.device == device and self.pe.dtype == dtype:
            return
        position = torch.arange(self.max_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-(math.log(10000.0) / self.d_model))
        )
        pe = torch.zeros(self.max_len, self.d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        assert D == self.d_model, f"pos-enc dim {self.d_model} != input dim {D}"
        self._maybe_build(device=x.device, dtype=x.dtype)
        return x + self.pe[:T].unsqueeze(0)


class OutputHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class BERT(nn.Module):
    """Minimal BERT-style encoder.

    Notes
    -----
    * Non-causal. Provide `attn_mask` if you want to restrict attention.
    * `sigma` is ignored (kept for API compatibility with DSL stack).
    """

    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.config = cfg
        bb = cfg.backbone
        self.vocab_size = int(vocab_size)

        d_model = int(getattr(bb, "dim_h"))
        n_heads = int(getattr(bb, "n_heads"))
        n_layers = int(getattr(bb, "n_blocks"))
        dropout = float(getattr(bb, "dropout", 0.1))
        mlp_ratio = float(getattr(bb, "mlp_ratio", 4.0))
        max_seq_len = int(getattr(bb, "max_seq_len", 1024))

        self.pos = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_seq_len)

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(mlp_ratio * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.enc_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.head = OutputHead(hidden_size=d_model, vocab_size=self.vocab_size)

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        x = self.pos(x)
        x = self.encoder(x, is_causal=False)
        logits = self.head(x)
        return logits