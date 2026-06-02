import math
import torch
import torch.nn.functional as F
import os
import typing
import transformers

LOG2 = math.log(2)


@torch.inference_mode()
def compute_metrics(
    model,
    input_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    snr_path,
    *,
    nll_n_rep: int = 1,
    roar_n_rep: int = 1,
    nll_int_steps: int = 30
):
    """
    Compute a compact set of metrics on a mini-batch.

    Returns dict:
      - "ce": mean cross-entropy (float)
      - "ntok": token count used in CE averaging (int)
      - "nll", "nll_diff", "nll_recon": NLL components (floats, mean over batch)
      - "nll_roar_bpt": ROAR estimator (float, mean over batch)
    """
    # CE (sample SNRs like training).
    snr = snr_path.sample(input_ids.size(0), input_ids.size(1))
    logits = model(model.noisy_embedding(input_ids, snr))
    ce = step_ce_loss(logits, input_ids, valid_tokens)
    ntok = int(valid_tokens.sum().item())

    nll_total, nll_diff, nll_recon, nll_last = nll(model, input_ids, valid_tokens,
                                                   n_rep=nll_n_rep, int_steps=nll_int_steps)
    roar = nll_roar(model, input_ids, valid_tokens, n_rep=roar_n_rep).mean().item()

    return {
        "ce": float(ce.item()),
        "ntok": int(ntok),
        "nll": float(nll_total.mean().item()),
        "nll_diff": float(nll_diff.mean().item()),
        "nll_recon": float(nll_recon.mean().item()),
        "nll_roar": float(roar),
        "nll_max_snr": float(nll_last.mean().item()),
    }


def step_ce_loss(logits: torch.Tensor,
                 targets: torch.Tensor,
                 valid_tokens):
    """
    Use cross entropy loss to predict tokens from noisy embedding for training
    logits: (B, L, V)
    targets: (B, L)
    Returns: (mean_ce, ntok) where ntok excludes PAD using valid_tokens.
    """
    # CE expects (N, C, L) or (N, C) — we use (B, V, L)
    ce_per_tok = F.cross_entropy(
        logits.transpose(1, 2).float(),  # (B, V, L)
        targets,
        reduction="none"
    )  # (B, L)

    ntok = int(valid_tokens.sum().item())
    loss = (ce_per_tok * valid_tokens).sum() / max(1, ntok)
    return loss


def nll(model, x, valid_tokens, n_rep=5, int_steps=100, snr_max=30., return_curves: bool = False):
    """
    Upper bound on NLL using the SNR path invariant bound with diffusion path.
    # TODO: Did YW use float64 for these metrics? Is it necessary? It's quite expensive.
    # I see, it's only for the recon term at high snr. It makes sense if you set 0 prob to wrong token you get high loss
    # Maybe it's enough that I put backbone logits to float, and train more at high SNR?
    # Also, because I pick the best bound, it may mitigate this issue.
    # https://arxiv.org/abs/2409.02908 identifies precision in sampling as a separate issue.

    Args:
      model: DSL model
      x: (B, T) token ids
      snr_max: 30.  Max SNR to use for integration.
      int_steps: 100  Number of integration steps to use.
      n_rep: number of samples per x.
    Returns:
      (nll, nll_reconst, nll_diffusion) — all in **bits per token**
    NOTES: for NLL bounds, simple linear snr schedules work fine.
    We don't care about snr_max from training - we just estimate bounds with various snr_max and take the best.
    Caveat: because we take the best one, per x, this may underestimate unless n_rep is large enough.
    TODO: Generalize to allow per token snr paths
    """
    snr_max = snr_max * math.log(model.vocab_size) / math.log(35)  # heuristic scaling based on vocab size
    snrs = torch.linspace(0., snr_max, int_steps, device=x.device)
    B, T = x.shape
    log2 = math.log(2)
    assert snrs.ndim == 1 and snrs.size(0) >= 2
    if snrs[0].item() != 0.0:  # If the first element isn't zero, prepend a zero
        snrs = torch.cat([torch.zeros(1, dtype=snrs.dtype, device=snrs.device), snrs])
    K = snrs.size(0)

    with torch.no_grad():
        x_embed = model.embed(x)
        nll_integrand = torch.zeros(B, K)  # store output on cpu
        nll_reconst = torch.zeros(B, K)  # store output on cpu
        for _ in range(n_rep):
            for i, snr in enumerate(snrs):  # Compute MSE and CE at each snr
                # noise and denoise
                z = model.noisy_embedding(x, snr)
                logits = model(z)
                p = logits.softmax(dim=-1).float()
                x_hat = torch.matmul(p, model.norm_weight_like(p))

                # recon CE, bits per token, padding handled in step_ce_loss
                nll_reconst[:, i] += (step_ce_loss(logits, x, valid_tokens) / log2 / n_rep).cpu()

                # diffusion MSE term, bits per token
                mse_x = F.mse_loss(x_hat, x_embed, reduction="none").sum(dim=-1)  # sum embed dimension
                mse_x = (mse_x * valid_tokens).sum(dim=-1) / valid_tokens.sum(dim=-1).clamp_min(1)  # mean per non-pad token
                nll_integrand[:, i] += (0.5 * mse_x / log2 / n_rep).cpu()  # integrand, bits per token units

        deltas = (snrs[1:] - snrs[:-1]).cpu()
        avg_vals = 0.5 * (nll_integrand[:, :-1] + nll_integrand[:, 1:])
        nll_diffusion = torch.cumsum(avg_vals * deltas.unsqueeze(0), dim=1)  # cumulative integral
        nll = nll_diffusion + nll_reconst[:,1:]


        # Optional WANDB logging of curves vs SNR (bits per token)
        if return_curves:
            # Mean over batch for compact logging
            mean_integrand = nll_integrand.mean(dim=0)            # (K,)
            mean_reconst   = nll_reconst.mean(dim=0)              # (K,)
            mean_diffusion = nll_diffusion.mean(dim=0)            # (K-1,)
            mean_bound     = (nll_reconst[:, 1:] + nll_diffusion).mean(dim=0)  # (K-1,)

        best_bound = nll.argmin(dim=1)  # index of best bound for each x
        all_x = torch.arange(B)

        if return_curves:  # to viz curves versus SNR
            curves = {
                "snr_all": snrs.detach().cpu(),
                "integrand_bpt": mean_integrand.detach().cpu(),
                "reconst_bpt": mean_reconst.detach().cpu(),
                "snr_pos": snrs[1:].detach().cpu(),
                "diffusion_cum_bpt": mean_diffusion.detach().cpu(),
                "bound_curve_bpt": mean_bound.detach().cpu(),
            }
            recon_pos = nll_reconst[:, 1:]
            return nll[all_x, best_bound], nll_diffusion[all_x, best_bound], recon_pos[all_x, best_bound], curves
    nll_last_snr_bound = nll[:, -1]
    recon_pos = nll_reconst[:, 1:]
    return nll[all_x, best_bound], nll_diffusion[all_x, best_bound], recon_pos[all_x, best_bound], nll_last_snr_bound



def nll_roar(model, x, valid_tokens, n_rep=1, snr_max=100.):
    """ROAR = Random Order Autoregressive NLL bound (BPT). See notes for derivation.
    -log p(x) = AR, but we can average over any order.
    This leads to sampling cross entropy of unmasked tokens, given masked tokens,
    where the probability of unmasked set of size k is, p(k) =  1/k / Z.
    -log p(x) = Z  E_k[CE( p(x_unmask|x_mask)]
    model: DSL model
    x: input ids
    n_rep: number of samples to average over
    snr_max: 30.  Large SNR to use for "unmasked" variables
    """
    B, T = x.shape
    log2 = math.log(2)
    # Using snr_max for a token is considered "unmasked"
    snr_max = snr_max * math.log(model.vocab_size) / math.log(35)  # heuristic scaling based on vocab size

    nll = 0  # accumulate bits per token
    for _ in range(n_rep):
        # Sample number of unmasked tokens per sequence
        unmask_sizes = torch.randint(0, T, (B,), device=x.device)
        mask = make_mask_tensor(unmask_sizes, T)  # True for masked, False for unmasked

        # Add mask noise and predict all tokens
        snrs = snr_max * (~mask)  # high SNR for unmasked
        z = model.noisy_embedding(x, snrs)
        logits = model(z)

        # Get cross entropy of predictions on masked tokens (excluding PAD, following Sahoo)
        ce_per_tok = F.cross_entropy(logits.transpose(1, 2).float(), x, reduction="none")  # B, T
        mask = mask & valid_tokens  # only consider non-pad masked tokens
        n_pred_tok = mask.sum(dim=-1).clamp_min(1)
        ce_mask = (ce_per_tok * mask).sum(dim=1) / n_pred_tok  # mean CE over predicted masked tokens
        nll += ce_mask / log2 / n_rep  # bits per token
    return nll


def make_mask_tensor(unmask_sizes: torch.Tensor, T: int):
    """
    Returns a boolean (B, T) tensor where each row has:
      - 0 for UNMASKED positions (count = unmask_sizes[i])
      - 1 for MASKED positions   (count = T - unmask_sizes[i])
    Positions are chosen uniformly at random per row.

    unmask_sizes: (B,) integers in [1, T]
    """
    device = unmask_sizes.device
    B = unmask_sizes.size(0)

    # Random permutation per row via sorting random scores
    scores = torch.rand(B, T, device=device)
    perm = scores.argsort(dim=1)  # [B, T], columns are positions in increasing random order

    # Compute per-position ranks (inverse permutation): ranks[b, pos] = order index of pos in perm[b]
    ranks = torch.empty_like(perm)
    ranks.scatter_(1, perm, torch.arange(T, device=device).expand(B, -1))

    # Unmask the first k positions (smallest ranks); mask the rest
    k = unmask_sizes.view(B, 1)                # [B, 1]
    unmask = ranks < k                         # True for UNMASKED positions
    mask = (~unmask)                           # 0 for unmask, 1 for mask

    return mask


# --------------------------------------------------------------------------------------
# Generative Perplexity
# Warning: Unchecked ChatGPT code
# Based on https://github.com/s-sahoo/duo/blob/7bc5b056ca88d18b1865b277b056d7b8796a9602/metrics.py#L164
# TODO: minimal problem is sahoo uses llama 2, I don't see how to do it with this code.
# --------------------------------------------------------------------------------------

class _RunningAvg:
    """Simple running average accumulator for scalar or per-token tensors."""
    def __init__(self, dtype=torch.float64, device=None):
        self.dtype = dtype
        self.device = device
        self.reset()

    def reset(self):
        self._sum = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self._weight = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def update(self, value: torch.Tensor, weight: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        value = value.to(self.dtype)
        weight = weight.to(self.dtype)
        self._sum += value.sum()
        self._weight += weight.sum()

    def mean(self) -> torch.Tensor:
        return self._sum / self._weight.clamp_min(1.0)


@torch.no_grad()
def _genppl_retokenize(
    tokenizer: transformers.PreTrainedTokenizerBase,
    text_samples: typing.List[str],
    max_length: int,
    device: torch.device,
    model_name: str,
):
    """Tokenize text for the eval LM and infer a sensible context size.

    Returns (input_ids, attention_mask, ctx_len).
    """
    kwargs = dict(
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    # Heuristic context size: prefer model config if available
    eval_ctx = None
    try:
        # Some tokenizers carry model_max_length; guard huge sentinel values
        mlen = getattr(tokenizer, "model_max_length", None)
        if isinstance(mlen, int) and 0 < mlen < 100_000:
            eval_ctx = mlen
    except Exception:
        pass
    if eval_ctx is None:
        # Back-compat with older snippet behaviour
        if "llama2" in model_name.lower():
            eval_ctx = 4096
        else:
            eval_ctx = 1024

    toks = tokenizer(text_samples, **kwargs)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    return input_ids, attention_mask, int(eval_ctx)


def sentence_entropy(sequences: torch.Tensor) -> torch.Tensor:
    """
    Compute the empirical token entropy per sequence ("sentence entropy") in bits.
    This is the diversity metric used in Appendix H.1 of https://arxiv.org/pdf/2409.02908.

    Args:
        sequences: (B, L) LongTensor of token ids (batch of tokenized sequences).
    Returns:
        entropies: (B,) Tensor of entropy (in bits) for each sequence.
    """
    B = sequences.size(0)
    entropies = []
    for seq in sequences:
        # Get unique tokens and their counts
        uniq, counts = torch.unique(seq, return_counts=True)
        probs = counts.float() / counts.sum()
        entropy = -(probs * probs.log2()).sum()
        entropies.append(entropy)
    return torch.stack(entropies)


@torch.no_grad()
def generative_perplexity(
    text_samples: typing.List[str],
    *,
    eval_model_name_or_path: str = 'gpt2-large',
    max_length: int = 1024,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    retokenize: bool = True,
) -> dict:
    """Compute generative perplexity of generated text using a reference LM.

    Mirrors the behaviour of the provided reference code, but in a compact form.

    Args:
        text_samples: list of decoded strings to evaluate.
        eval_model_name_or_path: HF model id or local path for the eval LM.
        max_length: max sequence length when retokenizing samples.
        batch_size: micro-batch for evaluation.
        device: device for the eval LM and tensors.
        retokenize: if False, `text_samples` is assumed to be an int tensor of token ids.

    Returns:
        A dict with:
            {
              "ppl": float,
              "mean_nll": float,           # in nats per token
              "ntok": int,
            }
    """
    device = torch.device(device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Tokenizer setup (pad->eos if missing for batching)
    tokenizer = transformers.AutoTokenizer.from_pretrained(eval_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model once in eval mode
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        eval_model_name_or_path
    ).eval()
    eval_model = eval_model.to(device)

    # Prepare inputs
    if retokenize:
        input_ids, attention_mask, eval_ctx = _genppl_retokenize(
            tokenizer, text_samples, max_length=max_length, device=device, model_name=eval_model_name_or_path
        )
    else:
        # Expect already-tokenized tensor
        assert isinstance(text_samples, torch.Tensor), "When retokenize=False, provide a LongTensor of token ids"
        input_ids = text_samples.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        eval_ctx = input_ids.shape[-1]

    # Accumulators
    nll_meter = _RunningAvg(dtype=torch.float64, device=device)

    # Mini-batching over samples, and chunk further by context window if needed
    B = input_ids.size(0)
    batch_size = max(1, min(batch_size, B))

    for b_start in range(0, B, batch_size):
        b_end = min(B, b_start + batch_size)
        ids_batch = input_ids[b_start:b_end]
        attn_batch = attention_mask[b_start:b_end]

        # Use sliding window approach with stride to avoid double counting
        stride = 512
        seq_len = ids_batch.shape[-1]
        num_strides = math.ceil((seq_len - eval_ctx + stride) / stride)
        
        for i in range(max(1, num_strides)):
            if i == 0:
                start, end = 0, min(eval_ctx, seq_len)
            else:
                start = i * stride
                end = min(start + eval_ctx, seq_len)
            
            ids_chunk = ids_batch[..., start:end]
            attn_chunk = attn_batch[..., start:end]
            
            logits = eval_model(ids_chunk, attention_mask=attn_chunk).logits.transpose(-1, -2)  # (B, V, L)
            nll = F.cross_entropy(logits[..., :-1], ids_chunk[..., 1:], reduction='none')  # (B, L-1)
            
            # Exclude all EOS tokens from valid mask
            valid = (ids_chunk[..., 1:] != tokenizer.eos_token_id).to(nll.dtype)
            
            if i == 0:
                # First window: use all positions
                nll_meter.update(nll * valid, valid)
            else:
                # Subsequent windows: only update non-overlapping part
                update_start = start + eval_ctx - stride
                update_window = end - update_start
                nll_meter.update(nll[..., -update_window:] * valid[..., -update_window:],
                                 valid[..., -update_window:])

    mean_nll = nll_meter.mean().item()            # nats per token
    ppl = float(math.exp(mean_nll))               # perplexity
    ntok = int(nll_meter._weight.item())

    # Compute sentence entropy (average over batch) - important to verify low gen ppl with high entropy/diversity
    sent_entropy = sentence_entropy(input_ids).mean().item()

    return {"ppl": ppl, "mean_nll": mean_nll, "ntok": ntok, "sentence_entropy": sent_entropy}
