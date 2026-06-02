"""
Soft Remasking: a sampling algorithm for masked diffusion LMs that allows
previously-committed tokens to be re-evaluated.

Instead of hard commit, confidence is used to probabilistically re-mask
low-confidence positions each step, giving the model a chance to correct
early mistakes. A cooling schedule ensures convergence.
"""

import torch
import numpy as np
import torch.nn.functional as F

# reuse Gumbel noise from LLaDA
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from generate import add_gumbel_noise


@torch.no_grad()
def generate_soft_remask(
    model,
    prompt,
    attention_mask=None,
    steps=64,
    gen_length=256,
    block_length=None,
    temperature=0.,
    cfg_scale=0.,
    mask_id=126336,
    # Soft remasking params
    remask_schedule='exponential',   # 'exponential', 'linear', 'cosine'
    remask_initial=0.3,              # initial remask probability scale
    remask_decay=0.92,               # per-step decay for exponential schedule
    protect_steps=4,                 # don't remask in first N steps (let model explore)
    freeze_steps=8,                  # stop remasking in last N steps (let it converge)
    confidence_eos_eot_inf=False,
):
    """
    Soft Remasking generation for masked diffusion LMs.

    Each step:
      1. Forward pass → logits
      2. For MASKED positions: unmask top-K by confidence (standard)
      3. For UNMASKED positions: re-mask with probability ~ (1 - confidence) * schedule_factor
      4. Cooling schedule reduces remask probability over time → convergence

    Args:
        remask_schedule: How remask probability decays. 'exponential', 'linear', or 'cosine'.
        remask_initial: Starting remask probability scale (0-1).
        remask_decay: Per-step multiplicative decay (for exponential schedule).
        protect_steps: Number of initial steps with no remasking (build initial structure).
        freeze_steps: Number of final steps with no remasking (ensure convergence).
    """
    if block_length is None:
        block_length = gen_length

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    prompt_index = (x != mask_id)
    prompt_len = prompt.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        # Compute reveal schedule: how many tokens should be unmasked by each step
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        mask_num = block_mask_index.sum(dim=1, keepdim=True)  # (B, 1)
        # Target: by step i, we should have revealed ~ (i+1)/steps_per_block fraction
        # of tokens. We use cumulative targets.
        target_revealed = torch.zeros(prompt.shape[0], steps_per_block, device=x.device, dtype=torch.long)
        for b_idx in range(prompt.shape[0]):
            total = mask_num[b_idx, 0].item()
            for s in range(steps_per_block):
                target_revealed[b_idx, s] = min(total, int(total * (s + 1) / steps_per_block))

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            # --- Forward pass ---
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

            # Compute confidence
            p = F.softmax(logits, dim=-1)  # (B, L, V)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

            if confidence_eos_eot_inf:
                x0_p[:, :] = torch.where(
                    (x0 == 126081) | (x0 == 126348),
                    torch.tensor(-float('inf'), device=x0_p.device),
                    x0_p
                )

            # --- Compute remask factor based on schedule ---
            remask_factor = _get_remask_factor(
                i, steps_per_block, remask_schedule,
                remask_initial, remask_decay,
                protect_steps, freeze_steps
            )

            # --- Phase 1: Soft Remask unmasked positions in block ---
            if remask_factor > 0 and i > 0:  # Don't remask on first step (nothing unmasked yet)
                block_unmasked = (~mask_index[:, block_start:block_end])
                if block_unmasked.any():
                    block_conf = x0_p[:, block_start:block_end]
                    # Remask probability: high when confidence is low
                    remask_prob = (1.0 - block_conf.clamp(0, 1)) * remask_factor
                    remask_roll = torch.rand_like(remask_prob)
                    remask_flags = (remask_roll < remask_prob) & block_unmasked
                    x[:, block_start:block_end] = torch.where(
                        remask_flags, torch.full_like(x[:, block_start:block_end], mask_id),
                        x[:, block_start:block_end]
                    )

            # --- Phase 2: Reveal tokens (unmask by confidence) ---
            # Recompute mask after remasking
            mask_index = (x == mask_id)
            # Only place predicted tokens on currently-masked positions
            x0_placed = torch.where(mask_index, x0, x)
            # Confidence only for masked positions
            confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=x0_p.device))

            # How many to reveal this step
            n_currently_masked_block = mask_index[:, block_start:block_end].sum(dim=1)  # (B,)

            if i < steps_per_block - 1:
                # Determine how many should be unmasked
                for b_idx in range(prompt.shape[0]):
                    n_target = target_revealed[b_idx, i].item()
                    n_already = block_length - n_currently_masked_block[b_idx].item()
                    n_reveal = max(0, n_target - n_already)
                    n_reveal = min(n_reveal, int(n_currently_masked_block[b_idx].item()))

                    if n_reveal > 0:
                        block_conf = confidence[b_idx, block_start:block_end]
                        _, top_idx = torch.topk(block_conf, k=n_reveal)
                        top_idx = top_idx + block_start
                        x[b_idx, top_idx] = x0_placed[b_idx, top_idx]
            else:
                # Last step: unmask everything remaining in block
                block_mask = mask_index[:, block_start:block_end]
                x[:, block_start:block_end] = torch.where(
                    block_mask, x0_placed[:, block_start:block_end],
                    x[:, block_start:block_end]
                )

    return x


def _get_remask_factor(step, total_steps, schedule, initial, decay, protect_steps, freeze_steps):
    """Compute the remask probability scaling factor at this step."""
    # No remasking in early or late steps
    if step < protect_steps or step >= total_steps - freeze_steps:
        return 0.0

    active_step = step - protect_steps
    active_total = total_steps - protect_steps - freeze_steps
    if active_total <= 0:
        return 0.0

    progress = active_step / active_total  # 0 → 1

    if schedule == 'exponential':
        return initial * (decay ** active_step)
    elif schedule == 'linear':
        return initial * (1.0 - progress)
    elif schedule == 'cosine':
        return initial * 0.5 * (1.0 + np.cos(np.pi * progress))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
