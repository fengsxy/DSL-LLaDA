"""Hybrid generation for DSL-LLaDA.

Reverse-hybrid approach:
  Phase 1: Standard remasking (few steps, e.g. 16) → rough draft tokens
  Phase 2: DSL ODE refinement from high SNR → polish using converter

The insight: converter accuracy is <4% at low SNR but >88% at SNR>20.
So use standard remasking (which doesn't need the converter) for the
hard low-SNR work, then DSL continuous refinement for the easy high-SNR
polishing where the converter excels.

Usage:
    from dsl_llada.core.dsl_generate import hybrid_generate
    output_ids = hybrid_generate(model, prompt_ids, gen_length=256)
"""
import math
import os
import sys
import torch
import torch.nn.functional as F

MASK_ID = 126336

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'LLaDA'))
from generate import generate as llada_generate


@torch.no_grad()
def hybrid_generate(model, prompt, gen_length=256,
                    phase1_steps=16, phase1_block_length=None,
                    phase2_steps=32, snr_start=20.0, snr_max=40.0,
                    renoise_scale=0.3,
                    temperature=None, remasking='low_confidence'):
    """
    Two-phase hybrid generation.

    Phase 1: Standard LLaDA remasking (fast, few steps) → rough draft
    Phase 2: Re-noise draft embeddings, then DSL ODE refinement

    The re-noising is key: at high SNR the converter is too confident about
    existing tokens to correct them. Adding noise (renoise_scale) gives the
    converter uncertainty, allowing it to propose corrections during ODE.

    Args:
        model:             LLaDA model with noise_embed and converter attached.
        prompt:            (B, P) int64 tensor of prompt token ids.
        gen_length:        number of tokens to generate.
        phase1_steps:      remasking steps for rough draft (default 16).
        phase1_block_length: block length for remasking (default=gen_length, single block).
        phase2_steps:      ODE refinement steps (default 32).
        snr_start:         starting SNR for refinement (default 20).
        snr_max:           target SNR (default 40).
        renoise_scale:     noise added to draft embeddings before refinement.
                           0 = no noise (just confirm draft), 1 = heavy noise.
        temperature:       sampling temperature (None = argmax).
        remasking:         'low_confidence' or 'random' for phase 1.

    Returns:
        (B, P + gen_length) tensor of token ids.
    """
    B, P = prompt.shape
    device = prompt.device

    if phase1_block_length is None:
        phase1_block_length = gen_length

    # ===================== Phase 1: Standard remasking =====================
    attention_mask = torch.ones_like(prompt)
    draft = llada_generate(
        model, prompt, attention_mask,
        steps=phase1_steps, gen_length=gen_length,
        block_length=phase1_block_length,
        temperature=0., cfg_scale=0., remasking=remasking)

    draft_ids = draft[:, P:]  # (B, gen_length)

    # ===================== Phase 2: DSL ODE refinement =====================
    noise_dim = model.noise_embed.weight.shape[1]
    L_gen = gen_length
    L_total = P + L_gen

    prompt_embed = model.noise_embed(prompt).float().double()
    draft_embed = model.noise_embed(draft_ids).float().double()

    # Re-noise: add Gaussian noise to draft embeddings so converter has
    # uncertainty and can propose corrections (not just confirm draft)
    if renoise_scale > 0:
        noise = renoise_scale * torch.randn_like(draft_embed)
        y_gen = draft_embed + noise
    else:
        y_gen = draft_embed

    # SNR schedule: snr_start → snr_max
    snrs = torch.linspace(snr_start, snr_max, steps=phase2_steps, device=device)

    for i in range(phase2_steps - 1):
        s = snrs[i].double()
        s_next = snrs[i + 1].double()
        ds = s_next - s
        if ds <= 0:
            continue

        z_gen = (s * y_gen).float()
        z_prompt = (snr_max * prompt_embed).float()
        z_full = torch.cat([z_prompt, z_gen], dim=1)
        h = model.converter(z_full).to(dtype=torch.bfloat16)

        # Use MASK_ID for gen region — backbone treats them as "to predict"
        input_ids = torch.full((B, L_total), MASK_ID, dtype=torch.long, device=device)
        input_ids[:, :P] = prompt

        out = model(input_ids=input_ids, inputs_embeds=h)
        logits = out.logits[:, P:, :].float()

        if temperature is not None and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            pred_ids = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, L_gen)
        else:
            pred_ids = logits.argmax(dim=-1)

        x_hat = model.noise_embed(pred_ids).float().double()

        # ODE step: dy = ((x_hat - y) / s) * ds
        f = (x_hat - y_gen) / s
        y_gen = y_gen + f * ds

    # ===================== Final prediction =====================
    z_final = (snrs[-1].double() * y_gen).float()
    z_full_final = torch.cat([(snr_max * prompt_embed).float(), z_final], dim=1)
    h_final = model.converter(z_full_final).to(dtype=torch.bfloat16)
    input_ids_final = torch.full((B, L_total), MASK_ID, dtype=torch.long, device=device)
    input_ids_final[:, :P] = prompt
    out_final = model(input_ids=input_ids_final, inputs_embeds=h_final)
    logits_final = out_final.logits[:, P:, :].float()

    if temperature is not None and temperature > 0:
        probs = F.softmax(logits_final / temperature, dim=-1)
        gen_ids = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, L_gen)
    else:
        gen_ids = logits_final.argmax(dim=-1)

    return torch.cat([prompt, gen_ids], dim=1)


# Keep original pure DSL generate for reference/comparison
@torch.no_grad()
def dsl_generate(model, prompt, gen_length=256, steps=64,
                 snr_max=40.0, snr_schedule='linear', solver='euler',
                 temperature=None, top_p=None):
    """Pure continuous DSL denoising (from noise). Kept for comparison."""
    B, P = prompt.shape
    L_gen = gen_length
    L_total = P + L_gen
    device = prompt.device
    noise_dim = model.noise_embed.weight.shape[1]

    snr_floor = 1e-4
    if snr_schedule == 'loglinear':
        log_min = math.log(snr_floor)
        log_max = math.log(snr_max)
        snrs = torch.exp(torch.linspace(log_min, log_max, steps=steps, device=device))
    else:
        snrs = torch.linspace(snr_floor, snr_max, steps=steps, device=device)

    prompt_embed = model.noise_embed(prompt).float().double()
    y_gen = torch.randn(B, L_gen, noise_dim, device=device, dtype=torch.float64)

    for i in range(steps - 1):
        s = snrs[i].double()
        s_next = snrs[i + 1].double()
        ds = (s_next - s).clamp_min(0.0)
        if ds == 0:
            continue

        z_gen = (s * y_gen).float()
        z_prompt = (snr_max * prompt_embed).float()
        z_full = torch.cat([z_prompt, z_gen], dim=1)
        h = model.converter(z_full).to(dtype=torch.bfloat16)

        input_ids = torch.full((B, L_total), MASK_ID, dtype=torch.long, device=device)
        input_ids[:, :P] = prompt
        out = model(input_ids=input_ids, inputs_embeds=h)
        logits = out.logits[:, P:, :].float()
        pred_ids = logits.argmax(dim=-1)
        x_hat = model.noise_embed(pred_ids).float().double()

        # ODE step (no noise)
        f = (x_hat - y_gen) / s
        y_gen = y_gen + f * ds

    z_final = (snrs[-1].double() * y_gen).float()
    z_full_final = torch.cat([(snr_max * prompt_embed).float(), z_final], dim=1)
    h_final = model.converter(z_full_final).to(dtype=torch.bfloat16)
    input_ids_final = torch.full((B, L_total), MASK_ID, dtype=torch.long, device=device)
    input_ids_final[:, :P] = prompt
    out_final = model(input_ids=input_ids_final, inputs_embeds=h_final)
    logits_final = out_final.logits[:, P:, :].float()
    gen_ids = logits_final.argmax(dim=-1)
    return torch.cat([prompt, gen_ids], dim=1)
