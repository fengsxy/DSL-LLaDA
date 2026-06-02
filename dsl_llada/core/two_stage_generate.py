# dsl_llada/eval/two_stage_generate.py
"""
Two-stage inference for DSLLaDA.

Stage 1 (steps < switch_step): fixed cap schedule (REMDM-style).
  - Unmask the top-k most confident masked positions each step.
Stage 2 (steps >= switch_step): confidence-based remasking.
  - Predict all positions, then re-mask low-confidence predictions.

Usage:
    from dsl_llada.core.two_stage_generate import two_stage_generate
    output_ids = two_stage_generate(model, prompt_ids, gen_length=128,
                                    total_steps=32, switch_ratio=0.5)
"""
import torch
import torch.nn.functional as F

MASK_ID = 126336


def two_stage_generate(model, prompt, gen_length=128, total_steps=32,
                       switch_ratio=0.5, snr_max=100.0):
    """
    Args:
        model:        DSLLaDA instance (on device, in eval mode).
        prompt:       (B, P) int64 tensor of prompt token ids.
        gen_length:   number of tokens to generate.
        total_steps:  total decoding steps.
        switch_ratio: fraction of steps using stage-1 (cap schedule).
                      E.g. 0.5 means steps 0..15 use cap, 16..31 use confidence.
        snr_max:      SNR used for unmasked positions.

    Returns:
        (B, P + gen_length) tensor of generated token ids.
    """
    B = prompt.shape[0]
    P = prompt.shape[1]
    seq_len = P + gen_length
    device = prompt.device

    # Start: prompt tokens fixed, generation region all masked
    x = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt

    prompt_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    prompt_mask[:, :P] = True

    switch_step = int(total_steps * switch_ratio)

    for step in range(total_steps):
        # Build per-token SNR: unmasked tokens get snr_max, masked get 0
        snrs = torch.zeros(B, seq_len, device=device)
        snrs[x != MASK_ID] = snr_max

        with torch.no_grad():
            out = model(input_ids=x, snrs=snrs)
            probs = F.softmax(out.logits.float(), dim=-1)  # (B, L, V)

        confidence, predicted = probs.max(dim=-1)  # (B, L)
        mask_positions = (x == MASK_ID) & (~prompt_mask)  # generation region only

        if step < switch_step:
            # Stage 1: cap schedule — unmask top-n_to_unmask confident positions
            n_masked = mask_positions[0].sum().item()
            n_to_unmask = max(1, int(n_masked * (step + 1) / switch_step))

            conf_masked = confidence.clone()
            conf_masked[~mask_positions] = -float('inf')
            n_to_unmask = min(n_to_unmask, int(mask_positions.sum(dim=-1).min().item()))
            if n_to_unmask > 0:
                _, top_idx = conf_masked.topk(n_to_unmask, dim=-1)
                x.scatter_(1, top_idx, predicted.gather(1, top_idx))
        else:
            # Stage 2: confidence remasking — accept high-confidence, re-mask rest
            progress = (step - switch_step) / max(1, total_steps - switch_step)
            threshold = 0.9 - 0.5 * progress  # linearly decrease: 0.9 → 0.4
            new_x = predicted.clone()
            new_x[confidence < threshold] = MASK_ID
            new_x[prompt_mask] = x[prompt_mask]  # never modify prompt
            x = new_x

    return x
