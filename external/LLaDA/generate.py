import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             eos_suppress_ratio=0.0, save_trajectory=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        save_trajectory: If True, return (x, trajectory) where trajectory is a dict with
            'tokens': (total_steps, gen_length) int tensor — top-1 prediction at each step
            'confidence': (total_steps, gen_length) float tensor — confidence of top-1
            'x_state': (total_steps, gen_length) int tensor — actual token state after remasking
    '''
    if remasking == 'dmax_soft':
        return generate_dmax_soft(model, prompt, attention_mask=attention_mask,
                                   steps=steps, gen_length=gen_length, block_length=block_length,
                                   mask_id=mask_id,
                                   confidence_eos_eot_inf=confidence_eos_eot_inf)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    prompt_len = prompt.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    total_steps = steps * num_blocks

    # Trajectory storage (only gen positions, on CPU to save GPU memory)
    if save_trajectory:
        traj_tokens = torch.full((total_steps, gen_length), mask_id, dtype=torch.long)
        traj_conf = torch.zeros(total_steps, gen_length, dtype=torch.float32)
        traj_state = torch.full((total_steps, gen_length), mask_id, dtype=torch.long)
        step_counter = 0

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
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

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            CONF_REMASKINGS = ('low_confidence', 'd3im', 'd3im_margin', 'd3im_expand', 'remdm_conf', 'remdm_cap')
            if remasking in CONF_REMASKINGS:
                logits_for_conf = logits
                # Suppress EOS/EoT in early steps to prevent EOS bias
                if eos_suppress_ratio > 0 and i < int(steps * eos_suppress_ratio):
                    logits_for_conf = logits.clone()
                    logits_for_conf[:, :, 126081] = -torch.inf  # EOS
                    logits_for_conf[:, :, 126348] = -torch.inf  # EoT
                    x0 = torch.argmax(add_gumbel_noise(logits_for_conf, temperature=temperature), dim=-1)
                p = F.softmax(logits_for_conf, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                # Margin confidence: P(top1) - P(top2) — better separates "confident" from "lucky"
                if remasking == 'd3im_margin':
                    top2, _ = p.topk(2, dim=-1)  # (B, L, 2)
                    x0_p = top2[:, :, 0] - top2[:, :, 1]  # margin
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # Record trajectory: top-1 prediction + confidence BEFORE remasking
            if save_trajectory:
                traj_tokens[step_counter] = x0[0, prompt_len:prompt_len + gen_length].cpu()
                traj_conf[step_counter] = x0_p[0, prompt_len:prompt_len + gen_length].float().cpu()

            # For D3IM: keep x0 as model's fresh prediction for ALL positions,
            # enabling direct token→token corrections (not just demote→refill).
            # For other strategies: committed positions keep their old token.
            if remasking not in ('d3im', 'd3im_margin', 'd3im_expand'):
                x0 = torch.where(mask_index, x0, x)

            if remasking in ('d3im', 'd3im_margin'):
                # D3IM: confidence over entire block (not just masked positions)
                confidence = x0_p.clone()
                block_start = prompt.shape[1] + num_block * block_length
                block_end = prompt.shape[1] + (num_block + 1) * block_length
                d3im_cumulative = torch.cumsum(num_transfer_tokens, dim=1)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    block_conf = confidence[j, block_start:block_end]
                    quota = min(int(d3im_cumulative[j, i].item()), block_conf.numel())
                    if i < steps - 1:
                        if quota <= 0:
                            continue
                        _, idx = torch.topk(block_conf, k=quota)
                        idx = idx + block_start
                        transfer_index[j, idx] = True
                    else:
                        # Last step: accept model's prediction for ALL block positions
                        transfer_index[j, block_start:block_end] = True

                if i < steps - 1:
                    masks = torch.full_like(x, mask_id)
                    masks[:, :prompt.shape[1]] = x[:, :prompt.shape[1]]
                    if num_block > 0:
                        masks[:, prompt.shape[1]:block_start] = x[:, prompt.shape[1]:block_start]
                    x = torch.where(transfer_index, x0, masks)
                else:
                    x = torch.where(transfer_index, x0, x)

            elif remasking == 'd3im_expand':
                # Variant B: ranking pool expands as blocks open.
                # At block k step i, pool = positions in blocks [0..k].
                # Quota = k*block_length + d3im_cumulative[i] (cumulative across blocks).
                # Previous-block positions CAN be re-masked or revised.
                confidence = x0_p.clone()
                pool_start = prompt.shape[1]
                pool_end = prompt.shape[1] + (num_block + 1) * block_length
                d3im_cumulative = torch.cumsum(num_transfer_tokens, dim=1)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    pool_conf = confidence[j, pool_start:pool_end]
                    quota = num_block * block_length + int(d3im_cumulative[j, i].item())
                    quota = min(quota, pool_conf.numel())
                    if i < steps - 1:
                        if quota <= 0:
                            continue
                        _, idx = torch.topk(pool_conf, k=quota)
                        idx = idx + pool_start
                        transfer_index[j, idx] = True
                    else:
                        # Last step of this block: all pool positions committed
                        transfer_index[j, pool_start:pool_end] = True

                if i < steps - 1:
                    masks = torch.full_like(x, mask_id)
                    masks[:, :prompt.shape[1]] = x[:, :prompt.shape[1]]
                    # NOTE: do NOT preserve previous blocks — let them be subject to remask
                    x = torch.where(transfer_index, x0, masks)
                else:
                    x = torch.where(transfer_index, x0, x)

            elif remasking == 'remdm_cap':
                # ReMDM-cap: simple capped remask probability (η_cap=0.02 from paper)
                eta_cap = 0.02
                block_start = prompt.shape[1] + num_block * block_length
                block_end = prompt.shape[1] + (num_block + 1) * block_length

                # Remask: each unmasked token in block has η_cap probability of being remasked
                if i < steps - 1:  # Don't remask on final step
                    block_unmasked = (~mask_index[:, block_start:block_end])
                    if block_unmasked.any():
                        remask_roll = torch.rand_like(x0_p[:, block_start:block_end])
                        remask_flags = (remask_roll < eta_cap) & block_unmasked
                        x[:, block_start:block_end][remask_flags] = mask_id

                # Reveal: recompute mask, then top-K reveal
                mask_index = (x == mask_id)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                n_reveal = int(num_transfer_tokens[0, i].item())
                if n_reveal > 0:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        n = min(n_reveal, int(mask_index[j].sum().item()))
                        if n > 0:
                            _, select_index = torch.topk(confidence[j], k=n)
                            transfer_index[j, select_index] = True
                    x[transfer_index] = x0[transfer_index]

            elif remasking == 'remdm_conf':
                # Confidence-based remasking (adapted from DSL2 sample_remdm_official_uncertainty)
                # Phase schedule: t goes from 1→0 over steps
                # Remask window: only remask during t ∈ (t_off, t_on)
                t_on, t_off, alpha_loop = 0.55, 0.05, 0.9
                t = 1.0 - i / steps
                dt = 1.0 / steps

                block_start = prompt.shape[1] + num_block * block_length
                block_end = prompt.shape[1] + (num_block + 1) * block_length

                # Determine sigma_max (remask intensity) based on schedule phase
                if t > t_on or t <= t_off:
                    # Outside remask window: pure reveal (like standard MDLM)
                    sigma_max_val = 0.0
                    if t > t_on:
                        move_chance_t = 1.0 - (1.0 - t) * alpha_loop / (1.0 - t_on)
                        move_chance_s = 1.0 - (1.0 - t + dt) * alpha_loop / (1.0 - t_on)
                    else:
                        move_chance_t = t * (1.0 - alpha_loop) / t_off
                        move_chance_s = max(0, t - dt) * (1.0 - alpha_loop) / t_off
                    alpha_t = 1.0 - move_chance_t
                    alpha_s = 1.0 - move_chance_s
                else:
                    # Inside remask window: confidence-based remasking active
                    alpha_t = alpha_loop
                    alpha_s = alpha_loop
                    sigma_max_val = min(1.0, (1.0 - alpha_s) / max(alpha_t, 1e-8))

                # --- Remask phase: probabilistically remask low-confidence unmasked tokens ---
                if sigma_max_val > 0.0:
                    # Uncertainty = 1 - confidence (higher = less confident)
                    uncertainty = 1.0 - x0_p.clamp(0, 1)
                    # Only consider block positions that are currently unmasked
                    block_unmasked = (~mask_index[:, block_start:block_end])
                    if block_unmasked.any():
                        # softmax over uncertainty → per-token remask probability
                        unc_block = uncertainty[:, block_start:block_end].clone()
                        unc_block[~block_unmasked] = -float('inf')
                        eta = torch.softmax(unc_block, dim=-1) * sigma_max_val
                        # Probabilistic remask
                        remask_roll = torch.rand_like(eta)
                        remask_flags = (remask_roll < eta) & block_unmasked
                        # Apply remask
                        x[:, block_start:block_end][remask_flags] = mask_id

                # --- Reveal phase: unmask tokens based on schedule ---
                # Recompute mask after potential remasking
                mask_index = (x == mask_id)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                # Calculate how many to reveal this step
                n_reveal = int(num_transfer_tokens[0, i].item())
                if n_reveal > 0:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        n = min(n_reveal, int(mask_index[j].sum().item()))
                        if n > 0:
                            _, select_index = torch.topk(confidence[j], k=n)
                            transfer_index[j, select_index] = True
                    x[transfer_index] = x0[transfer_index]

            else:
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

            # Record x_state after remasking
            if save_trajectory:
                traj_state[step_counter] = x[0, prompt_len:prompt_len + gen_length].cpu()
                step_counter += 1

    if save_trajectory:
        trajectory = {
            'tokens': traj_tokens,       # (total_steps, gen_length) top-1 predictions
            'confidence': traj_conf,      # (total_steps, gen_length) confidence scores
            'x_state': traj_state,        # (total_steps, gen_length) actual state after remasking
        }
        return x, trajectory

    return x


@torch.no_grad()
def generate_dmax_soft(model, prompt, attention_mask=None, steps=64, gen_length=256, block_length=32,
                       mask_id=126336, tau_acc=0.9, confidence_eos_eot_inf=True):
    '''DMax-style soft embedding interpolation sampler.

    Maintains soft embedding state h for uncommitted positions:
        h_j = renormalize(pi * e(y_j) + (1-pi) * e_mask)
    Commits position hard (fills into x) once pi >= tau_acc OR quota-forced.
    Quota schedule matches d3im's num_transfer_tokens to keep NFE budget identical.
    '''
    device = model.device
    wte = model.get_input_embeddings()
    vocab_dim = wte.embedding_dim
    B = prompt.shape[0]
    prompt_len = prompt.shape[1]
    full_len = prompt_len + gen_length

    # Token state (hard commits): mask everywhere in gen, prompt fixed.
    x = torch.full((B, full_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone()
    prompt_index = (x != mask_id)
    committed = prompt_index.clone()  # which positions have been hard-committed

    # Soft embedding state h (B, full_len, D)
    with torch.no_grad():
        e_mask = wte(torch.tensor([mask_id], device=device)).squeeze(0)  # (D,)
        h = wte(x).clone()  # init: prompt -> e(token), gen -> e_mask
    norm_mask = e_mask.norm()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_uncommitted = block_length  # mask count per sample assumed equal across batch
        mask_per_sample = (~committed[:, block_start:block_end]).sum(dim=1, keepdim=True)  # (B, 1)
        base = mask_per_sample // steps_per_block
        rem = mask_per_sample % steps_per_block
        num_transfer = torch.zeros(B, steps_per_block, device=device, dtype=torch.long) + base
        for b in range(B):
            num_transfer[b, :rem[b]] += 1
        cum_transfer = torch.cumsum(num_transfer, dim=1)  # (B, steps_per_block)

        for i in range(steps_per_block):
            logits = model(inputs_embeds=h, attention_mask=attention_mask).logits
            if confidence_eos_eot_inf:
                logits[:, :, 126081] = -torch.inf
                logits[:, :, 126348] = -torch.inf
            p = F.softmax(logits, dim=-1)
            y = p.argmax(dim=-1)  # (B, L)
            pi = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, L)

            # Restrict attention to current block's uncommitted positions for commit decision
            pi_block = pi[:, block_start:block_end].clone()
            uncommitted_block = ~committed[:, block_start:block_end]
            pi_block[~uncommitted_block] = -float('inf')

            # Commit: by tau_acc threshold OR quota (whichever larger) or final step (all)
            quota_i = cum_transfer[:, i]  # (B,)
            new_commit = torch.zeros_like(committed)
            for b in range(B):
                if i == steps_per_block - 1:
                    # Last step of block: commit all remaining uncommitted in block
                    idxs = torch.nonzero(uncommitted_block[b], as_tuple=False).squeeze(-1)
                else:
                    # Union of (tau_acc threshold hits) and (top-quota by confidence)
                    k = int(min(quota_i[b].item(), uncommitted_block[b].sum().item()))
                    if k <= 0 and not (pi_block[b] >= tau_acc).any():
                        continue
                    # Top-k by conf
                    _, topk_idx = torch.topk(pi_block[b], k=max(k, 1))
                    thr_idx = torch.nonzero(pi_block[b] >= tau_acc, as_tuple=False).squeeze(-1)
                    idxs = torch.unique(torch.cat([topk_idx[:k], thr_idx]))
                if idxs.numel() == 0:
                    continue
                global_idx = idxs + block_start
                new_commit[b, global_idx] = True
                x[b, global_idx] = y[b, global_idx]

            # Update h: committed positions -> e(y), remaining uncommitted gen -> soft interp
            # Fully committed (either just-now or earlier): h = wte(x)
            committed = committed | new_commit
            # Soft interp for still-uncommitted gen positions
            still_soft = (~committed) & (~prompt_index)  # (B, L)

            e_y = wte(y)  # (B, L, D)
            # h_soft = renormalize(pi * e_y + (1-pi) * e_mask) * (pi*||e_y|| + (1-pi)*||e_mask||)
            pi_exp = pi.unsqueeze(-1)  # (B, L, 1)
            h_raw = pi_exp * e_y + (1.0 - pi_exp) * e_mask  # broadcast
            h_raw_norm = h_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            norm_y = e_y.norm(dim=-1, keepdim=True)
            target_norm = pi_exp * norm_y + (1.0 - pi_exp) * norm_mask
            h_soft = h_raw / h_raw_norm * target_norm

            # Apply: committed -> e(x), still_soft -> h_soft
            # Rebuild h efficiently
            h_committed = wte(x)
            mask3 = still_soft.unsqueeze(-1)
            h = torch.where(mask3, h_soft, h_committed)

    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
