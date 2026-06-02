"""NLL evaluation metrics for DSL-LLaDA.

Adapted from dsl2_ref/dsls/metrics.py for our model interface:
  - model.noise_embed: frozen unit-norm nn.Embedding (V, noise_dim)
  - model.converter: SoftmaxConvertBias (noisy z → backbone input)
  - noisy_embedding(): add noise at given SNR
  - model(input_ids=x, inputs_embeds=h): LLaDA backbone forward

Metrics:
  1. NLL integral bound (bits/token): upper bound via SNR path integration
  2. ROAR NLL (bits/token): random-order autoregressive bound
  3. Fixed-SNR CE + accuracy: reconstruction quality at specific SNR points
  4. Clean CE: backbone degradation check (no DSL path)
  5. Mask infilling accuracy: original LLaDA capability

Usage:
    python dsl_llada/eval/eval_nll.py --checkpoint ./checkpoints/dsl_1000step/checkpoint-1000 --gpu 0
    python dsl_llada/eval/eval_nll.py --checkpoint GSAI-ML/LLaDA-8B-Instruct --gpu 1  # baseline (no DSL)
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from dsl_llada.core.dsl_modules import noisy_embedding, attach_dsl_modules

LOG2 = math.log(2)
MASK_ID = 126336


def load_model(path, device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    return model.to(device).eval()


def attach_dsl_if_needed(model, checkpoint, device):
    """Attach DSL modules from checkpoint if not already present."""
    if hasattr(model, 'noise_embed'):
        return True
    # Only attach if checkpoint has DSL weights
    import glob
    import safetensors.torch
    shard_files = sorted(glob.glob(os.path.join(checkpoint, 'model-*.safetensors')))
    has_dsl = False
    for sf in shard_files:
        sd = safetensors.torch.load_file(sf, device='cpu')
        if any(k.startswith('converter.') or k.startswith('noise_embed.') for k in sd.keys()):
            has_dsl = True
        del sd
        if has_dsl:
            break
    if not has_dsl:
        return False

    attach_dsl_modules(model, freeze_ff_out=True)
    for sf in shard_files:
        sd = safetensors.torch.load_file(sf, device=str(device))
        for k, v in sd.items():
            if k.startswith('converter.') or k.startswith('noise_embed.'):
                parts = k.split('.')
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                param = getattr(obj, parts[-1])
                if isinstance(param, torch.nn.Parameter):
                    param.data.copy_(v)
                else:
                    setattr(obj, parts[-1], v)
        del sd
    model.noise_embed = model.noise_embed.to(device)
    model.converter = model.converter.to(device)
    return True


def dsl_forward(model, input_ids, snr):
    """DSL forward: noise_embed → noisy z → converter → backbone → logits."""
    z = noisy_embedding(model.noise_embed, input_ids, snr)
    h = model.converter(z).to(dtype=torch.bfloat16)
    out = model(input_ids=input_ids, inputs_embeds=h)
    return out.logits


def make_mask_tensor(unmask_sizes, T):
    """Boolean (B, T) tensor: True=masked, False=unmasked."""
    device = unmask_sizes.device
    B = unmask_sizes.size(0)
    scores = torch.rand(B, T, device=device)
    perm = scores.argsort(dim=1)
    ranks = torch.empty_like(perm)
    ranks.scatter_(1, perm, torch.arange(T, device=device).expand(B, -1))
    k = unmask_sizes.view(B, 1)
    return ranks >= k  # True = masked


@torch.no_grad()
def nll_integral(model, x, valid_tokens, n_rep=3, int_steps=50, snr_max=30.0):
    """NLL upper bound via SNR path integration (bits/token).

    Adapted from dsl2_ref/dsls/metrics.py:nll().

    Returns:
        nll_best: (B,) best NLL bound per sample
        nll_diff_best: (B,) diffusion component at best SNR
        nll_recon_best: (B,) reconstruction CE at best SNR
        best_snr: (B,) SNR that minimizes the bound
    """
    vocab_size = model.noise_embed.weight.shape[0]
    snr_max_scaled = snr_max * math.log(vocab_size) / math.log(35)
    snrs = torch.linspace(0., snr_max_scaled, int_steps, device=x.device)
    B, T = x.shape

    if snrs[0].item() != 0.0:
        snrs = torch.cat([torch.zeros(1, dtype=snrs.dtype, device=snrs.device), snrs])
    K = snrs.size(0)

    # Get clean embeddings for MSE computation
    x_embed = model.noise_embed(x).float()  # (B, T, noise_dim)
    embed_weight = model.noise_embed.weight.float()  # (V, noise_dim)

    nll_integrand = torch.zeros(B, K)
    nll_reconst = torch.zeros(B, K)

    for _ in range(n_rep):
        for i, snr_val in enumerate(snrs):
            # Noisy embedding + forward
            snr_t = torch.full((B, T), snr_val.item(), device=x.device)
            logits = dsl_forward(model, x, snr_t)

            # Reconstruction CE (bits/token)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                x.view(-1), reduction='none'
            ).view(B, T)
            ce_masked = (ce * valid_tokens).sum(dim=-1) / valid_tokens.sum(dim=-1).clamp_min(1)
            nll_reconst[:, i] += (ce_masked / LOG2 / n_rep).cpu()

            # Diffusion MSE term: x_hat = p @ embed_weight
            logp = F.log_softmax(logits.float(), dim=-1)
            p = logp.exp()  # (B, T, V)
            x_hat = torch.matmul(p, embed_weight.to(p.device))  # (B, T, noise_dim)

            mse = F.mse_loss(x_hat, x_embed, reduction='none').sum(dim=-1)  # (B, T)
            mse_mean = (mse * valid_tokens).sum(dim=-1) / valid_tokens.sum(dim=-1).clamp_min(1)
            nll_integrand[:, i] += (0.5 * mse_mean / LOG2 / n_rep).cpu()

    # Trapezoidal integration
    deltas = (snrs[1:] - snrs[:-1]).cpu()
    avg_vals = 0.5 * (nll_integrand[:, :-1] + nll_integrand[:, 1:])
    nll_diffusion = torch.cumsum(avg_vals * deltas.unsqueeze(0), dim=1)  # (B, K-1)
    nll_total = nll_diffusion + nll_reconst[:, 1:]

    # Pick best bound per sample
    best_idx = nll_total.argmin(dim=1)
    all_b = torch.arange(B)
    best_snr = snrs[1:][best_idx].cpu()
    recon_pos = nll_reconst[:, 1:]

    return (
        nll_total[all_b, best_idx],
        nll_diffusion[all_b, best_idx],
        recon_pos[all_b, best_idx],
        best_snr,
    )


@torch.no_grad()
def nll_roar(model, x, valid_tokens, n_rep=3, snr_max=100.0):
    """ROAR NLL bound (bits/token).

    Random Order Autoregressive: mask random subset, predict masked tokens
    given unmasked tokens at high SNR.
    """
    vocab_size = model.noise_embed.weight.shape[0]
    snr_max_scaled = snr_max * math.log(vocab_size) / math.log(35)
    B, T = x.shape

    nll = torch.zeros(B)
    for _ in range(n_rep):
        unmask_sizes = torch.randint(0, T, (B,), device=x.device)
        mask = make_mask_tensor(unmask_sizes, T)  # True=masked

        snr_t = snr_max_scaled * (~mask).float()  # high SNR for unmasked
        logits = dsl_forward(model, x, snr_t)

        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            x.view(-1), reduction='none'
        ).view(B, T)

        valid_mask = mask & valid_tokens.bool()
        n_pred = valid_mask.sum(dim=-1).clamp_min(1).float()
        ce_masked = (ce * valid_mask.float()).sum(dim=1) / n_pred
        nll += (ce_masked / LOG2 / n_rep).cpu()

    return nll


@torch.no_grad()
def eval_fixed_snr(model, x, valid_tokens, snr_values=(1., 5., 10., 20., 50.)):
    """CE and accuracy at fixed SNR points."""
    B, T = x.shape
    results = {}
    for snr_val in snr_values:
        snr_t = torch.full((B, T), snr_val, device=x.device)
        logits = dsl_forward(model, x, snr_t)

        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            x.view(-1), reduction='none'
        ).view(B, T)
        ce_mean = (ce * valid_tokens).sum() / valid_tokens.sum().clamp_min(1)

        preds = logits.argmax(dim=-1)
        acc = ((preds == x).float() * valid_tokens).sum() / valid_tokens.sum().clamp_min(1)

        results[f'snr{int(snr_val)}'] = {
            'ce': ce_mean.item(),
            'acc': acc.item(),
        }
    return results


@torch.no_grad()
def nll_mc(model, x, mc_num=128, batch_size=16):
    """LLaDA official Monte Carlo NLL estimate (Appendix B.5).

    For unconditional NLL: treat entire sequence as 'answer' (no prompt).
    Randomly mask k tokens, compute CE on masked positions / p_mask,
    average over mc_num samples.

    Works for both original LLaDA and Softmasker (uses standard mask path).

    Returns: NLL estimate (nats, lower = better, negative value)
    """
    B_orig, T = x.shape
    assert B_orig == 1, "nll_mc expects single-sample input"
    seq = x.expand(batch_size, -1).clone()  # (batch_size, T)

    losses = []
    for _ in range(mc_num // batch_size):
        # Random mask ratio per sample in batch
        k = torch.randint(1, T + 1, (), device=x.device)
        # Spread k values across batch for variance reduction (from LLaDA code)
        x_vals = torch.round(
            torch.linspace(float(k), k + (batch_size - 1) * (T / batch_size),
                           steps=batch_size, device=x.device)
        ).long()
        x_vals = ((x_vals - 1) % T) + 1

        # Create masks
        indices = torch.arange(T, device=x.device).repeat(batch_size, 1)
        is_mask = indices < x_vals.unsqueeze(1)
        for j in range(batch_size):
            is_mask[j] = is_mask[j][torch.randperm(T, device=x.device)]

        noisy_seq = torch.where(is_mask, MASK_ID, seq)
        p_mask = (x_vals / T).unsqueeze(1).expand(batch_size, T)

        mask_index = (noisy_seq == MASK_ID)
        logits = model(noisy_seq).logits

        loss = F.cross_entropy(
            logits[mask_index].float(), seq[mask_index], reduction='none'
        ) / p_mask[mask_index]
        loss = loss.sum() / batch_size
        losses.append(loss.item())

    # Return per-token log-likelihood (nats): log p(x) / T
    return -sum(losses) / len(losses) / T


@torch.no_grad()
def eval_clean_ce(model, x, valid_tokens):
    """Clean CE (no DSL path, direct wte) — measures backbone degradation."""
    out = model(input_ids=x)
    logits = out.logits
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        x.view(-1), reduction='none'
    ).view(x.shape)
    ce_mean = (ce * valid_tokens).sum() / valid_tokens.sum().clamp_min(1)

    preds = logits.argmax(dim=-1)
    acc = ((preds == x).float() * valid_tokens).sum() / valid_tokens.sum().clamp_min(1)
    return ce_mean.item(), acc.item()


@torch.no_grad()
def eval_mask_infill(model, x, valid_tokens, mask_rate=0.5):
    """Mask infilling: mask some tokens, recover with standard LLaDA path."""
    B, T = x.shape
    n_mask = int(T * mask_rate)
    masked = x.clone()
    perm = torch.randperm(T, device=x.device)[:n_mask]
    masked[:, perm] = MASK_ID

    out = model(masked)
    preds = out.logits.argmax(dim=-1)
    acc = (preds[:, perm] == x[:, perm]).float().mean().item()
    return acc


def load_eval_texts(tokenizer, max_len=512, n_samples=200, device='cuda'):
    """Load eval texts from fineweb-edu for NLL evaluation."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT',
                      split='train', streaming=True)

    batches = []
    for item in ds:
        text = item['text']
        tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False,
                           truncation=True, max_length=max_len)
        ids = tokens['input_ids']
        if ids.shape[1] >= 64:  # skip very short texts
            batches.append(ids.to(device))
            if len(batches) >= n_samples:
                break

    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--nll_int_steps', type=int, default=50)
    parser.add_argument('--nll_n_rep', type=int, default=3)
    parser.add_argument('--roar_n_rep', type=int, default=3)
    parser.add_argument('--mc_num', type=int, default=128,
                        help='Monte Carlo samples for LLaDA NLL estimate')
    parser.add_argument('--mc_batch', type=int, default=16,
                        help='Batch size for MC NLL')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'

    if args.output is None:
        ckpt_name = args.checkpoint.replace('/', '_').replace('.', '_')
        os.makedirs('results', exist_ok=True)
        args.output = f'results/nll_{ckpt_name}.json'

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    has_dsl = attach_dsl_if_needed(model, args.checkpoint, device)
    print(f"  DSL modules: {'attached' if has_dsl else 'not found (baseline)'}")

    print(f"Loading eval texts (n={args.n_samples}, max_len={args.max_len})...")
    texts = load_eval_texts(tokenizer, max_len=args.max_len,
                            n_samples=args.n_samples, device=device)
    print(f"  Loaded {len(texts)} texts")

    # Accumulators
    all_mc_nll = []
    all_nll = []
    all_nll_diff = []
    all_nll_recon = []
    all_best_snr = []
    all_roar = []
    all_clean_ce = []
    all_clean_acc = []
    all_infill_acc = []
    snr_results_accum = {}

    t0 = time.time()
    for i, ids in enumerate(tqdm(texts, desc='NLL eval')):
        valid = torch.ones_like(ids, dtype=torch.float32)

        # MC NLL (LLaDA official, always available)
        mc_nll_val = nll_mc(model, ids, mc_num=args.mc_num, batch_size=args.mc_batch)
        all_mc_nll.append(mc_nll_val)

        # Clean CE (always available)
        ce_clean, acc_clean = eval_clean_ce(model, ids, valid)
        all_clean_ce.append(ce_clean)
        all_clean_acc.append(acc_clean)

        # Mask infilling (always available)
        infill_acc = eval_mask_infill(model, ids, valid, mask_rate=0.5)
        all_infill_acc.append(infill_acc)

        if has_dsl:
            # NLL integral bound
            nll_best, nll_diff, nll_recon, best_snr = nll_integral(
                model, ids, valid,
                n_rep=args.nll_n_rep, int_steps=args.nll_int_steps)
            all_nll.append(nll_best.mean().item())
            all_nll_diff.append(nll_diff.mean().item())
            all_nll_recon.append(nll_recon.mean().item())
            all_best_snr.append(best_snr.mean().item())

            # ROAR NLL
            roar = nll_roar(model, ids, valid, n_rep=args.roar_n_rep)
            all_roar.append(roar.mean().item())

            # Fixed-SNR eval
            snr_res = eval_fixed_snr(model, ids, valid)
            for k, v in snr_res.items():
                if k not in snr_results_accum:
                    snr_results_accum[k] = {'ce': [], 'acc': []}
                snr_results_accum[k]['ce'].append(v['ce'])
                snr_results_accum[k]['acc'].append(v['acc'])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"\n  Progress: {i+1}/{len(texts)} ({elapsed:.0f}s)")
            print(f"    MC NLL: {sum(all_mc_nll)/len(all_mc_nll):.4f}")
            print(f"    Clean CE: {sum(all_clean_ce)/len(all_clean_ce):.4f}, "
                  f"Clean Acc: {sum(all_clean_acc)/len(all_clean_acc):.4f}")
            print(f"    Infill@50%: {sum(all_infill_acc)/len(all_infill_acc):.4f}")
            if has_dsl and all_nll:
                print(f"    NLL bound: {sum(all_nll)/len(all_nll):.4f} bpt, "
                      f"ROAR: {sum(all_roar)/len(all_roar):.4f} bpt")
                print(f"    Best SNR: {sum(all_best_snr)/len(all_best_snr):.1f}")

    elapsed = time.time() - t0

    # Aggregate
    import numpy as np
    summary = {
        'checkpoint': args.checkpoint,
        'has_dsl': has_dsl,
        'n_samples': len(texts),
        'max_len': args.max_len,
        'elapsed_sec': elapsed,
        'mc_nll': float(np.mean(all_mc_nll)),
        'mc_nll_std': float(np.std(all_mc_nll)),
        'mc_nll_bpt': float(-np.mean(all_mc_nll) / LOG2),  # convert to bits/token (positive)
        'clean_ce': float(np.mean(all_clean_ce)),
        'clean_ce_std': float(np.std(all_clean_ce)),
        'clean_acc': float(np.mean(all_clean_acc)),
        'infill_acc_50pct': float(np.mean(all_infill_acc)),
        'infill_acc_std': float(np.std(all_infill_acc)),
    }

    if has_dsl and all_nll:
        summary.update({
            'nll_bpt': float(np.mean(all_nll)),
            'nll_bpt_std': float(np.std(all_nll)),
            'nll_diff_bpt': float(np.mean(all_nll_diff)),
            'nll_recon_bpt': float(np.mean(all_nll_recon)),
            'nll_roar_bpt': float(np.mean(all_roar)),
            'nll_roar_std': float(np.std(all_roar)),
            'best_snr': float(np.mean(all_best_snr)),
        })

        # Fixed-SNR summary
        fixed_snr = {}
        for k, v in snr_results_accum.items():
            fixed_snr[k] = {
                'ce': float(np.mean(v['ce'])),
                'acc': float(np.mean(v['acc'])),
            }
        summary['fixed_snr'] = fixed_snr

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"NLL Evaluation: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"  Samples: {len(texts)}, Time: {elapsed:.0f}s")
    print(f"  MC NLL:         {summary['mc_nll']:.4f} nats ({summary['mc_nll_bpt']:.4f} bpt)")
    print(f"  Clean CE:       {summary['clean_ce']:.4f} (acc={summary['clean_acc']:.4f})")
    print(f"  Infill@50%%:     {summary['infill_acc_50pct']:.4f}")
    if has_dsl:
        print(f"  NLL bound:      {summary['nll_bpt']:.4f} bpt")
        print(f"    Diffusion:    {summary['nll_diff_bpt']:.4f} bpt")
        print(f"    Recon:        {summary['nll_recon_bpt']:.4f} bpt")
        print(f"  ROAR NLL:       {summary['nll_roar_bpt']:.4f} bpt")
        print(f"  Best SNR:       {summary['best_snr']:.1f}")
        if 'fixed_snr' in summary:
            print(f"  Fixed-SNR:")
            for k, v in summary['fixed_snr'].items():
                print(f"    {k}: CE={v['ce']:.4f}, Acc={v['acc']:.4f}")
    print(f"\n  Saved to: {args.output}")


if __name__ == '__main__':
    main()
