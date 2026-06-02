"""DSL ODE correction compatible with LLaDA.

Two approaches:
1. Pure ODE: z_noisy → iterative denoise in noise_embed space → final logits
2. Hybrid: DSL ODE first pass → confidence scores → LLaDA remasking on uncertain positions

Key parameterization:
  Training:    z_noisy = snr * embed + sqrt(snr) * eps  (embed is unit norm)
  EDM sigma:   sigma = 1/sqrt(snr),  z = y/sigma^2
  So z_noisy in our training IS z in EDM.

Usage:
    python dsl_llada/test_ode_correction2.py [gpu_id]
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from transformers import AutoTokenizer, AutoModel
from dsl_llada.dsl_modules import attach_dsl_modules, noisy_embedding
import safetensors.torch

PCA_CKPT = './checkpoints/dsl_pca128_20260312_171035/checkpoint-1000'
ORIGINAL = 'GSAI-ML/LLaDA-8B-Instruct'
MASK_ID = 126336

# --- Load model ---
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)
print('Loading model...')
model = AutoModel.from_pretrained(PCA_CKPT, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
attach_dsl_modules(model, noise_dim=128, freeze_ff_out=True, noise_init='pca')

dsl_keys = {'converter.backbone_embedding.weight', 'converter.backbone_embedding.bias',
            'converter.beta', 'converter.embed.weight', 'converter.logit_bias', 'noise_embed.weight'}
for cf in os.listdir(PCA_CKPT):
    if cf.endswith('.safetensors'):
        state = safetensors.torch.load_file(os.path.join(PCA_CKPT, cf))
        for k, v in state.items():
            if k in dsl_keys:
                parts = k.split('.')
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                param = getattr(obj, parts[-1])
                if isinstance(param, torch.nn.Parameter):
                    param.data.copy_(v.to(param.device))
                else:
                    setattr(obj, parts[-1], v.to(param.device))
model.converter.cuda()
model.noise_embed.cuda()

R = '\033[91m'; G = '\033[92m'; B = '\033[1m'; D = '\033[2m'; Y = '\033[93m'; X = '\033[0m'


@torch.no_grad()
def dsl_forward(model, z, input_ids):
    """z (noise_embed space) → converter → backbone → logits.

    input_ids used for position embeddings / attention structure.
    """
    h = model.converter(z)
    wte_dtype = model.model.transformer.wte.weight.dtype
    logits = model(input_ids=input_ids, inputs_embeds=h.to(dtype=wte_dtype)).logits
    return logits


@torch.no_grad()
def x_hat_from_logits(model, logits):
    """logits → softmax → weighted sum of noise_embed → denoised x_hat in embed space."""
    p = logits.float().softmax(dim=-1)  # (B, L, V_full)
    W = model.noise_embed.weight.float()  # (V_embed, d_noise)
    V_embed = W.shape[0]
    p_trunc = p[:, :, :V_embed]
    # Renormalize after truncation (padding tokens have no noise_embed)
    p_trunc = p_trunc / (p_trunc.sum(dim=-1, keepdim=True) + 1e-8)
    x_hat = torch.matmul(p_trunc, W.to(p_trunc.device))  # (B, L, d_noise)
    return x_hat


@torch.no_grad()
def ode_correction(model, input_ids, snr_start, snr_end, num_steps=10):
    """SDEdit-style correction using DSL ODE.

    1. Encode input tokens to noise_embed space
    2. Add noise at snr_start level
    3. ODE denoise from snr_start → snr_end
    4. Return final logits

    Parameterization mapping:
      sigma = 1/sqrt(snr)
      y = z_noisy / snr = z_noisy * sigma^2
      ODE: d = (y - x_hat) / sigma;  y += (sigma_next - sigma) * d;  z = y / sigma_next^2
    """
    B, L = input_ids.shape

    # Encode to noise_embed space and add noise
    z_clean = model.noise_embed(input_ids).float()  # unit norm

    # z_noisy = snr * z_clean + sqrt(snr) * eps  (matches training)
    eps = torch.randn_like(z_clean)
    z_noisy = snr_start * z_clean + (snr_start ** 0.5) * eps

    # Convert to sigma schedule
    sigma_start = 1.0 / (snr_start ** 0.5)
    sigma_end = 1.0 / (snr_end ** 0.5)

    # Karras-like sigma schedule (sigma decreasing = SNR increasing)
    rho = 7.0
    s_min = sigma_end ** (1.0 / rho)
    s_max = sigma_start ** (1.0 / rho)
    ramp = torch.linspace(0, 1, num_steps + 1, device=input_ids.device, dtype=torch.float64)
    sigmas = (s_max + ramp * (s_min - s_max)) ** rho

    # Convert to y-space: y = sigma^2 * z
    y = (sigmas[0] ** 2) * z_noisy.double()

    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Current z estimate
        z = y / (sigma ** 2)

        # Model prediction: z → converter → backbone → logits → x_hat
        logits = dsl_forward(model, z.float(), input_ids)
        x_hat = x_hat_from_logits(model, logits).double()

        # ODE step (Euler)
        d = (y - x_hat) / sigma
        y = y + (sigma_next - sigma) * d

    # Final prediction
    z_final = y / (sigmas[-1] ** 2)
    logits_final = dsl_forward(model, z_final.float(), input_ids)
    return logits_final


@torch.no_grad()
def hybrid_correction(model, input_ids, snr_start=5.0, snr_end=30.0,
                      ode_steps=5, remask_steps=3):
    """Hybrid DSL+LLaDA correction.

    Phase 1 (DSL): ODE denoise → get per-token confidence
    Phase 2 (LLaDA): Low-confidence positions → MASK → standard LLaDA remasking
    """
    B, L = input_ids.shape

    # Phase 1: DSL ODE → logits + confidence
    logits = ode_correction(model, input_ids, snr_start, snr_end, num_steps=ode_steps)
    probs = F.softmax(logits.float(), dim=-1)
    x0 = logits.argmax(dim=-1)  # (B, L) predicted tokens
    x0_conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L) confidence

    # Phase 2: Mask low-confidence positions, run LLaDA remasking
    # Threshold: if DSL disagrees with input AND confidence > threshold → change
    changed_mask = (x0 != input_ids)  # positions where DSL wants to change
    confident_change = changed_mask & (x0_conf > 0.3)  # only apply confident changes

    # Build hybrid tokens: keep originals, replace where DSL is confident
    x_hybrid = input_ids.clone()

    # For uncertain positions: mask them and let LLaDA fill
    uncertain = changed_mask & (~confident_change)
    x_hybrid[confident_change] = x0[confident_change]
    x_hybrid[uncertain] = MASK_ID

    # Run LLaDA remasking on masked positions
    if uncertain.any():
        for step in range(remask_steps):
            mask_idx = (x_hybrid == MASK_ID)
            if not mask_idx.any():
                break
            logits_r = model(x_hybrid).logits
            x_pred = logits_r.argmax(dim=-1)
            # Unmask most confident
            p_r = F.softmax(logits_r.float(), dim=-1)
            conf_r = p_r.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)
            conf_r[~mask_idx] = -1.0
            n_unmask = max(1, mask_idx.sum().item() // (remask_steps - step))
            _, top_idx = conf_r.view(-1).topk(min(n_unmask, mask_idx.sum().item()))
            batch_idx = top_idx // L
            pos_idx = top_idx % L
            x_hybrid[batch_idx, pos_idx] = x_pred[batch_idx, pos_idx]

    return x_hybrid, x0, x0_conf


texts = [
    ('Wrong', 'One plus one equals three.'),
    ('Right', 'One plus one equals two.'),
    ('Wrong', 'The capital of France is Germany.'),
    ('Right', 'The capital of France is Paris.'),
    ('Wrong', '2 + 2 = 5'),
    ('Right', '2 + 2 = 4'),
    ('Wrong', '9999 + 1 = 99999'),
    ('Right', '9999 + 1 = 10000'),
    ('Wrong', 'Water boils at 50 degrees Celsius.'),
    ('Right', 'Water boils at 100 degrees Celsius.'),
    ('Wrong', 'The Earth orbits around Mars.'),
    ('Right', 'The Earth orbits around the Sun.'),
]

print(f'\n{"="*70}')
print('Method 1: Pure ODE correction (SDEdit-style)')
print(f'{"="*70}\n')

torch.manual_seed(42)

for tag, text in texts:
    ids = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
    color = R if tag == 'Wrong' else G
    print(f'{color}[{tag}]{X} {B}"{text}"{X}')

    # Try different starting noise levels (snr_start)
    # Lower snr_start = more noise = more changes allowed
    for snr_start in [3, 5, 10]:
        logits = ode_correction(model, ids, snr_start=snr_start, snr_end=40.0, num_steps=10)
        top1 = logits.argmax(dim=-1)[0]
        changes = []
        for i in range(ids.shape[1]):
            tid = ids[0, i].item()
            pid = top1[i].item()
            if pid != tid:
                ot = tokenizer.decode([tid])
                pt = tokenizer.decode([pid])
                prob = F.softmax(logits.float(), dim=-1)[0, i, pid].item()
                changes.append(f'{ot}→{R}{pt}{X}({prob:.2f})')
        label = f'ODE snr={snr_start}→40'
        if changes:
            print(f'  {label:>20}: {", ".join(changes)}')
        else:
            print(f'  {label:>20}: {G}agrees{X}')

print(f'\n{"="*70}')
print('Method 2: Single-pass converter (noisy_embedding, for comparison)')
print(f'{"="*70}\n')

torch.manual_seed(42)

for tag, text in texts:
    ids = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
    color = R if tag == 'Wrong' else G
    print(f'{color}[{tag}]{X} {B}"{text}"{X}')

    for snr_val in [5, 10, 20, 30]:
        snr = torch.full((1, ids.shape[1]), float(snr_val), device='cuda', dtype=torch.float32)
        z_noisy = noisy_embedding(model.noise_embed, ids, snr)
        logits = dsl_forward(model, z_noisy, ids)
        top1 = logits.argmax(dim=-1)[0]
        changes = []
        for i in range(ids.shape[1]):
            tid = ids[0, i].item()
            pid = top1[i].item()
            if pid != tid:
                ot = tokenizer.decode([tid])
                pt = tokenizer.decode([pid])
                prob = F.softmax(logits.float(), dim=-1)[0, i, pid].item()
                changes.append(f'{ot}→{R}{pt}{X}({prob:.2f})')
        label = f'SNR={snr_val}'
        if changes:
            print(f'  {label:>20}: {", ".join(changes)}')
        else:
            print(f'  {label:>20}: {G}agrees{X}')

print(f'\n{"="*70}')
print('Method 3: Hybrid DSL→LLaDA correction')
print(f'{"="*70}\n')

torch.manual_seed(42)

for tag, text in texts:
    ids = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
    color = R if tag == 'Wrong' else G
    print(f'{color}[{tag}]{X} {B}"{text}"{X}')

    x_final, x0_dsl, x0_conf = hybrid_correction(model, ids, snr_start=5, snr_end=30,
                                                   ode_steps=5, remask_steps=3)
    changes_dsl = []
    changes_final = []
    for i in range(ids.shape[1]):
        tid = ids[0, i].item()
        did = x0_dsl[0, i].item()
        fid = x_final[0, i].item()
        conf = x0_conf[0, i].item()
        if did != tid:
            changes_dsl.append(f'{tokenizer.decode([tid])}→{Y}{tokenizer.decode([did])}{X}({conf:.2f})')
        if fid != tid:
            changes_final.append(f'{tokenizer.decode([tid])}→{R}{tokenizer.decode([fid])}{X}')

    if changes_dsl:
        print(f'  {"DSL proposal":>16}: {", ".join(changes_dsl)}')
    else:
        print(f'  {"DSL proposal":>16}: {G}agrees{X}')
    if changes_final:
        print(f'  {"Final output":>16}: {", ".join(changes_final)}')
    else:
        print(f'  {"Final output":>16}: {G}agrees{X}')
