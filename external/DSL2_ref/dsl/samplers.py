"""Samplers for DSL.
TODO:
1. Add adaptive sampler
2. Use adaptive sampler or curvature to record SNR paths, then incorporate them into training.
3. Special random order AR path sampler (corresponds to special SNR paths, trained with ROAR NLL)
"""
import torch

@torch.no_grad()
def sample_roar(model, batch_size, seq_length, causal=False, low_precision_sampling=False, snr_max=100.):
    """Sample in a random order autoregressive way, using our model as an MDM
    snr_max is used to indicate a high SNR to be considered "unmasked".
    causal: if True, sample left to right.
    low_precision_sampling: if True, use half precision for logits during sampling. https://arxiv.org/pdf/2409.02908
    argues that Sahoo et al got better gen ppl but lower diversity by using inaccurate low precision sampling.
    """
    device = next(model.parameters()).device
    z = torch.zeros((batch_size, seq_length, model.noise_dim), device=device)
    order = torch.randperm(seq_length) if not causal else torch.arange(seq_length)
    for t in order:
        logits = model(z)[:, t]
        if low_precision_sampling:
            logits = logits.float()
        else:
            logits = logits.double()
        probs = logits.softmax(dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).long()
        z[:, t] = snr_max * model.embed(token_ids)
    W = model.norm_weight_like(z)
    return (z @ W.T).argmax(dim=-1)


@torch.no_grad()
def sample_path(model, batch_size, seq_length, snrs, history=False, small_init=1e-4):
    """Simulate the model, dz = dsnr x_hat dt + sqrt(dsnr) dW,
    SNRs is a discrete SNR path. It should have shape (K,) where K is seq_length.
    OR it can be a tensor of shape (T, K) to have per-sample SNR paths.
    """
    # Normalize snrs shape and device
    device = next(model.parameters()).device
    snrs = torch.as_tensor(snrs, dtype=torch.float32, device=device)
    if snrs.ndim == 1:
        snrs = snrs.unsqueeze(0)
    assert snrs.ndim == 2, "size: (number of tokens, K steps) or just (K,) if snrs are same for each token."

    # Set initial condition. With SNR=1e-8, then z=1e-8 * x + 1e-4 * noise. x part is negligible.
    z = small_init * torch.randn((batch_size, seq_length, model.noise_dim), device=device)
    if history:
        zs = torch.zeros((batch_size, seq_length, model.noise_dim, snrs.shape[1]), device='cpu')
        zs[..., 0] = z.cpu()
    for i, dsnr in enumerate(snrs[:,1:].T - snrs[:,:-1].T):
        x_hat = model.x_hat(z)
        delta = dsnr.unsqueeze(-1).clamp_min(0.)
        z = z + x_hat * delta + torch.sqrt(delta) * torch.randn_like(z)
        if history:
            zs[..., i + 1] = z.cpu()
    token_ids = model(z).argmax(dim=-1)
    if history:
        return token_ids, zs
    return token_ids