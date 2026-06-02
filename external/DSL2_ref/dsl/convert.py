"""A few variations for converting from noisy embedding space to backbone (transformer) space.
Ideas to try:
- VQVAE, RVQ. Learn discrete reps from continous (problem, need extra losses)      (haven't tried)
- Gumbel softmax to pick discrete code, trainable, can output discrete vectors. No extra loss (poor in first try)
- Gating - produce mask token for very noisy input              (first try works poorly)
- Add one mask token - add learnable attention bias toward it.  (easy, helps a little in text8, will test lm1b)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxConvert(nn.Module):
    def __init__(self, d, vocab_size, embed):
        """
        Convert noisy embedding z (B, T, z_dim) to model input (B, T, d)
        using softmax over token similarities to `embed.weight` (V, z_dim),
        then a trainable linear (V -> d).
        """
        super().__init__()
        self.embed = embed                               # nn.Embedding or a module with .weight (V, z_dim)
        self.embedding = nn.Linear(vocab_size, d, bias=False)  # (V -> d)
        self.beta = nn.Parameter(1. / torch.sqrt(torch.tensor(d, dtype=torch.float32)))

    def forward(self, z):
        # z: (B, T, z_dim); embed.weight: (V, z_dim)
        # scores: (B, T, V)
        W = self.embed.weight.to(device=z.device, dtype=z.dtype, non_blocking=True)
        scores = self.beta * (z @ W.T)
        softmaxed = F.softmax(scores, dim=-1)
        # out: (B, T, d)
        return self.embedding(softmaxed)


class SoftmaxConvertBias(nn.Module):
    def __init__(self, d_backbone, vocab_size, embed):
        """

        """
        super().__init__()
        self.embed = embed  # (V, d_noise) The embedding where noise is added
        V, d_noise = embed.weight.shape
        assert vocab_size == V, f"vocab_size {vocab_size} != embed.weight.shape[0] {V}"

        # goes from V+1 -> dim of transformer backbone, learned
        self.backbone_embedding = nn.Linear(vocab_size + 1, d_backbone, bias=False)

        # Learnable scaling of dot product attention, init to 1/sqrt(d_noise) (d_noise is dimension of dot)
        self.beta = nn.Parameter(1. / torch.sqrt(torch.tensor(float(d_noise))))

        # Learnable bias over logits (favor extra slot initially)
        self.logit_bias = nn.Parameter(torch.zeros(V + 1))
        with torch.no_grad():
            self.logit_bias[-1] = math.log(V)  # lean toward +1 slot with prob 1/2, representing mask

    def get_token_probs(self, z):
        """Convert noisy z into probability of different tokens, with bias toward mask if z is too noisy"""
        B, T, d_noise = z.shape
        device, dtype = z.device, z.dtype

        # Keys K: concat embedding keys with the extra zero key for mask -> (V+1, z_dim)
        zeros_row = torch.zeros(1, d_noise, dtype=self.embed.weight.dtype, device=device)
        embed_w = self.embed.weight.to(device=device, dtype=dtype, non_blocking=True)  # avoid bugs at eval. Weight norm doesn't register and move to right device
        K = torch.cat([embed_w, zeros_row], dim=0)  # (V+1, z_dim)

        # softmax attention: logits are scaled dot product (B, T, V+1)
        logits = self.beta * (z @ K.transpose(0, 1))
        logits = logits + self.logit_bias.to(device=device, dtype=dtype)
        probs = F.softmax(logits, dim=-1)  # Attention over V+1, shape (B, T, V+1)
        return probs

    def forward(self, z):
        """Use backbone embedding as values to convert probabilities to vector embedding for backbone."""
        return self.backbone_embedding(self.get_token_probs(z))  # (B, T, d)


class Gumbel(nn.Module):
    def __init__(self, d, vocab_size, embed, noisy_dim):
        """Use Gumbel to always output one token.
         Use magnitude to add a "mask" signal to each token.
         """
        super().__init__()
        self.embed = embed                               # nn.Embedding or a module with .weight (V, z_dim)
        self.embedding = nn.Linear(vocab_size, d, bias=False)  # (V -> d)
        self.beta = nn.Parameter(1. / torch.sqrt(torch.tensor(d, dtype=torch.float32)))
        self.mask_token = torch.nn.Parameter(torch.randn(1, d))

    def forward(self, z):
        # z: (B, T, z_dim); embed.weight: (V, z_dim)
        # scores: (B, T, V)
        logits = self.beta * (z @ self.embed.weight.T)
        probs = F.softmax(logits, dim=-1)

        # Use gumbel to get single token, differentiably
        # one_hot = F.gumbel_softmax(logits, tau=1., hard=True, dim=-1)
        # best_embedding = self.embedding(one_hot)  # out: (B, T, d)

        # Use argmax to get single token
        idx = torch.argmax(logits, dim=-1)  # (batch, T)
        one_hot = F.one_hot(idx, num_classes=logits.size(-1)).float()  # (batch, T, vocab_size)
        best_embedding = self.embedding(one_hot)  # out: (B, T, d)

        # Get avg embedding
        avg_embedding = self.embedding(probs)

        # Confidence gating: if max prob < 0.9, use mask token
        max_prob, _ = torch.max(probs, dim=-1, keepdim=True)
        use_mask = max_prob < 0.1
        mask_token_expanded = self.mask_token.expand(z.shape[0], z.shape[1], -1)
        final_output = torch.where(use_mask, mask_token_expanded, avg_embedding)
        return best_embedding  # test
        # Notes: different variations all fail: best embed or gumbel sample best embed, with or wihtout mask
        # And avg embedding (previous case) plus gate/mask also does poorly
        # return