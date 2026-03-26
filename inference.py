"""DSL-LLaDA inference: 3 modes — standard remasking, SDE generation, error correction.

Usage:
  python inference.py --model Beta1 --mode sde --prompt "Write a story about a robot."
  python inference.py --model Beta2 --mode standard --prompt "Question: What is 2+3?\nAnswer: Let's think step by step."
  python inference.py --model Beta2 --mode correct --input "The cat sat on teh mat"
"""
import torch
import torch.nn.functional as F
import math, os, argparse
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
SDE_TOP_K = 512

HF_MODELS = {
    'Beta1': 'liddlefish/DSL-LLaDA-Beta1',
    'Beta2': 'liddlefish/DSL-LLaDA-Beta2',
    'Highpass': 'liddlefish/DSL-LLaDA-Highpass',
}
LOCAL_MODEL_ROOT = os.environ.get('DSL_LLADA_MODEL_ROOT', '/data2/ylong030/models/dsl-llada')
LOCAL_TOKENIZER_PATH = os.path.join(LOCAL_MODEL_ROOT, 'LLaDA-8B-Instruct-tokenizer')
LOCAL_MODELS = {
    'Beta1': os.path.join(LOCAL_MODEL_ROOT, 'DSL-LLaDA-Beta1'),
    'Beta2': os.path.join(LOCAL_MODEL_ROOT, 'DSL-LLaDA-Beta2'),
    'Highpass': os.path.join(LOCAL_MODEL_ROOT, 'DSL-LLaDA-Highpass'),
}


def resolve_model_path(model_name):
    local_path = LOCAL_MODELS[model_name]
    return local_path if os.path.isdir(local_path) else HF_MODELS[model_name]


def resolve_tokenizer_path():
    return LOCAL_TOKENIZER_PATH if os.path.isdir(LOCAL_TOKENIZER_PATH) else 'GSAI-ML/LLaDA-8B-Instruct'


def load_model(model_name, device='cuda:0'):
    """Load model + DSL weights from a local mirror when available, else HuggingFace."""
    ckpt = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_path(), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()

    # Load DSL weights (noise_embed, converter)
    ne, lb, bw, bb, bv, rw, rb = None, None, None, None, 2.5, None, None
    import json
    if os.path.isdir(ckpt):
        index_path = os.path.join(ckpt, 'model.safetensors.index.json')
        with open(index_path) as f:
            index = json.load(f)
        shard = index['weight_map'].get('noise_embed.weight')
        if shard is not None:
            st = load_file(os.path.join(ckpt, shard), device=device)
            ne = st.get('noise_embed.weight', None)
            if ne is not None:
                ne = ne.float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].item()
                rw = st.get('converter.residual_proj.weight')
                rb = st.get('converter.residual_proj.bias')
                if rw is not None: rw = rw.float()
                if rb is not None: rb = rb.float()
    else:
        from huggingface_hub import hf_hub_download
        index = json.load(open(hf_hub_download(ckpt, 'model.safetensors.index.json')))
        # Find which shard has noise_embed
        for key, shard in index['weight_map'].items():
            if key == 'noise_embed.weight':
                path = hf_hub_download(ckpt, shard)
                st = load_file(path, device=device)
                ne = st.get('noise_embed.weight', None)
                if ne is not None:
                    ne = ne.float()
                    lb = st['converter.logit_bias'].float()
                    bw = st['converter.backbone_embedding.weight'].float()
                    bb = st['converter.backbone_embedding.bias'].float()
                    bv = st['converter.beta'].item()
                    rw = st.get('converter.residual_proj.weight')
                    rb = st.get('converter.residual_proj.bias')
                    if rw is not None: rw = rw.float()
                    if rb is not None: rb = rb.float()
                break

    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0) if ne is not None else None
    return model, tokenizer, dict(ne=ne, lb=lb, bw=bw, bb=bb, bv=bv, rw=rw, rb=rb, K=K)


@lru_cache(maxsize=None)
def _is_digit_token(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()
    return text.isdigit()


# ═══════════════════════════════════════
# Mode 1: Standard Remasking
# ═══════════════════════════════════════
def standard_remasking(model, tokenizer, prompt, gen_length=256, steps=64, device='cuda:0', seed=42,
                       digit_delay=False, sampling=False, temperature=1.0):
    """Standard confidence-based remasking. Best for reasoning tasks."""
    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]

    x = torch.full((1, pl + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[0, :pl] = iids[0]
    base = gen_length // steps; rem = gen_length % steps
    nt = [base] * steps
    for i in range(rem): nt[i] += 1

    torch.manual_seed(seed)
    delay_until = max(1, int(steps * 0.75))
    for s in range(steps):
        mi = (x == MASK_ID)
        logits = model(x).logits
        scaled_logits = logits / max(float(temperature), 1e-5)
        p = F.softmax(scaled_logits, dim=-1)
        if sampling:
            x0 = torch.multinomial(p.view(-1, p.shape[-1]), 1).view(p.shape[:-1])
        else:
            x0 = scaled_logits.argmax(dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p[~mi] = -float('inf')
        if digit_delay and s < delay_until:
            digit_mask = torch.zeros_like(x0_p, dtype=torch.bool)
            masked_positions = torch.nonzero(mi, as_tuple=False)
            for batch_idx, pos_idx in masked_positions.tolist():
                if _is_digit_token(tokenizer, int(x0[batch_idx, pos_idx])):
                    digit_mask[batch_idx, pos_idx] = True
            x0_p[digit_mask] *= 0.5
        _, idx = x0_p.sort(dim=-1, descending=True)
        x[0, idx[0, :nt[s]]] = x0[0, idx[0, :nt[s]]]

    return tokenizer.decode(x[0, pl:pl + gen_length], skip_special_tokens=True)


# ═══════════════════════════════════════
# Mode 2: SDE Generation (Heun)
# ═══════════════════════════════════════
def sde_generate(model, tokenizer, dsl, prompt, gen_length=128, steps=16,
                 noise_scale=0.3, snr_max=50.0, device='cuda:0', seed=42,
                 norm_init=True, sensitive=True):
    """SDE Heun generation in continuous embedding space. Best for creative writing."""
    ne, lb, bw, bb, bv, rw, rb, K = dsl['ne'], dsl['lb'], dsl['bw'], dsl['bb'], dsl['bv'], dsl['rw'], dsl['rb'], dsl['K']
    use_res = rw is not None

    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    dummy = torch.zeros(1, pl + gen_length, dtype=torch.long, device=device)

    snr_min = 0.01 if not norm_init else 3.0
    if sensitive:
        n1 = max(1, int(steps * 0.05)); n2 = max(1, int(steps * 0.90)); n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1)).to(device)

    torch.manual_seed(seed)
    y = F.normalize(torch.randn(1, gen_length, nd, device=device), dim=-1) if norm_init else torch.randn(1, gen_length, nd, device=device)

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h_low = F.linear(probs, bw, bb)
        if use_res:
            embed_w = ne.to(z.dtype)
            z_low = probs[:, :, :embed_w.shape[0]].to(z.dtype) @ embed_w
            z_ln = z_low / (z_low.norm(dim=-1, keepdim=True) + 1e-8)
            ps = (z * z_ln).sum(dim=-1, keepdim=True)
            z_high = z - ps * z_ln
            h_high = F.linear(z_high.to(rw.dtype), rw, rb)
            h = (h_low + h_high).to(torch.bfloat16)
        else:
            h = h_low.to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16) if pe.dim() == 3 else pe.unsqueeze(0).to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        probs_bb = F.softmax(logits, dim=-1)
        top_vals, top_idx = probs_bb.topk(min(SDE_TOP_K, probs_bb.shape[-1]), dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        xh = (top_vals.unsqueeze(-1) * ne[top_idx.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)
        return xh

    ns = noise_scale
    actual_steps = len(snrs) - 1
    for i in range(actual_steps):
        s, s_next = snrs[i], snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s
        g = ns / s
        y_e = y + f * ds + g * dW
        xh_e = get_xhat(y_e, s_next)
        f_e = (xh_e - y_e) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    # Final decode via backbone
    z_final = snrs[-1] * y
    cl_final = bv * (z_final.float() @ K.T) + lb
    h_final = F.linear(F.softmax(cl_final.float(), dim=-1), bw, bb)
    if use_res:
        embed_w = ne.to(z_final.dtype)
        z_low_f = F.softmax(cl_final.float(), dim=-1)[:, :, :embed_w.shape[0]].to(z_final.dtype) @ embed_w
        z_ln_f = z_low_f / (z_low_f.norm(dim=-1, keepdim=True) + 1e-8)
        ps_f = (z_final * z_ln_f).sum(dim=-1, keepdim=True)
        z_high_f = z_final - ps_f * z_ln_f
        h_high_f = F.linear(z_high_f.to(rw.dtype), rw, rb)
        h_final = (h_final + h_high_f).to(torch.bfloat16)
    else:
        h_final = h_final.to(torch.bfloat16)
    embeds_final = torch.cat([pe.to(torch.bfloat16) if pe.dim() == 3 else pe.unsqueeze(0).to(torch.bfloat16), h_final], dim=1)
    with torch.no_grad():
        lo_final = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
    final_toks = lo_final.argmax(dim=-1)[0]
    return tokenizer.decode(final_toks, skip_special_tokens=True)


# ═══════════════════════════════════════
# Mode 3: Error Correction
# ═══════════════════════════════════════
def error_correction(model, tokenizer, text, device='cuda:0'):
    """Single forward pass: model predicts what each token SHOULD be.
    Positions where prediction != input are potential corrections."""
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    preds = logits.argmax(dim=-1)[0]
    original = ids[0]

    corrections = []
    for i in range(len(original)):
        if preds[i] != original[i]:
            orig_tok = tokenizer.decode([original[i]])
            pred_tok = tokenizer.decode([preds[i]])
            corrections.append((i, orig_tok, pred_tok))

    corrected_text = tokenizer.decode(preds, skip_special_tokens=True)
    return corrected_text, corrections


def main():
    parser = argparse.ArgumentParser(description='DSL-LLaDA Inference')
    parser.add_argument('--model', default='Beta1', choices=list(HF_MODELS.keys()))
    parser.add_argument('--mode', default='sde', choices=['standard', 'sde', 'correct'])
    parser.add_argument('--prompt', default='Write a short story about a robot discovering music.')
    parser.add_argument('--input', default='The cat sat on teh mat and lookd at the brid.')
    parser.add_argument('--gen_length', type=int, default=128)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--digit_delay', action='store_true',
                        help='Delay digit token unmasking during the first 75% of standard remasking steps.')
    parser.add_argument('--sampling', action='store_true',
                        help='Use sampling instead of greedy argmax in standard remasking.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling/confidence temperature for standard remasking.')
    args = parser.parse_args()

    print(f'Loading {args.model}...')
    model, tokenizer, dsl = load_model(args.model, args.device)
    print(f'Loaded. Mode: {args.mode}')

    if args.mode == 'standard':
        text = standard_remasking(
            model, tokenizer, args.prompt, args.gen_length, args.steps, args.device, args.seed,
            digit_delay=args.digit_delay, sampling=args.sampling, temperature=args.temperature,
        )
        print(f'\n{text}')

    elif args.mode == 'sde':
        text = sde_generate(model, tokenizer, dsl, args.prompt, args.gen_length, args.steps,
                            device=args.device, seed=args.seed)
        print(f'\n{text}')

    elif args.mode == 'correct':
        corrected, corrections = error_correction(model, tokenizer, args.input, args.device)
        print(f'\nOriginal:  {args.input}')
        print(f'Corrected: {corrected}')
        if corrections:
            print(f'\nChanges ({len(corrections)}):')
            for pos, orig, pred in corrections[:20]:
                print(f'  pos {pos}: "{orig}" → "{pred}"')


if __name__ == '__main__':
    main()
