"""SDE parameter search for 4 DSL-LLaDA models.
Tests 40 configs per model on 10 prompts, measures GenPPL (GPT-2 Large).
"""
import os, sys, math, json, time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # line-buffered
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2TokenizerFast

MASK_ID = 126336
SDE_TOP_K = 512
device = 'cuda:0'

MODELS_CFG = {
    'sm_b2': {
        'ckpt': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
        'beta_train': 2.0,
        'embed': 'random',
    },
    'b1': {
        'ckpt': 'checkpoints/pertoken_b1_d100_1k/checkpoint-1000',
        'beta_train': 1.0,
        'embed': 'random',
    },
    'sem_b05_3k': {
        'ckpt': 'checkpoints/semantic_b05_d100_10k/checkpoint-3000',
        'beta_train': 0.5,
        'embed': 'semantic',
    },
    'sem_b2': {
        'ckpt': 'checkpoints/semantic_b2_d100_1k/checkpoint-1000',
        'beta_train': 2.0,
        'embed': 'semantic',
    },
}

PROMPTS = [
    "Write a short paragraph about photosynthesis.",
    "Explain machine learning in simple terms.",
    "What is the water cycle?",
    "Describe the solar system.",
    "Summarize the key aspects of democracy.",
    "Question: What is 27 + 58? Show the reasoning briefly.",
    "A store had 45 apples. It sold 18, then got 27 more. How many?",
    "Write a short story about a robot discovering music.",
    "Plan a 3-day Tokyo trip. Keep each day to 3 items.",
    "Explain quantum mechanics in simple terms.",
]

# Parameter grid
BETA_INFERS = [1.0, 2.0, 3.0, 4.0, 5.0]
STEPS_LIST = [32, 64]
SCHEDULES = ["uniform", "sensitive"]
NOISE_SCALES = [0.01, 0.05]
SNR_MAX = 100
GEN_LENGTH = 256
SEED = 42


def build_snr_schedule(schedule, steps, snr_max=100):
    snr_min = 0.01
    if schedule == "sensitive":
        n1 = max(1, int(steps * 0.05))
        n2 = max(1, int(steps * 0.90))
        n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
        ])
    else:  # uniform (loglinear)
        snrs = torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1))
    return snrs


def load_dsl_weights(ckpt, dev):
    ne, lb, bw, bb, bv = None, None, None, None, None
    for fname in sorted(os.listdir(ckpt)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt, fname), device=dev)
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].float().item()
                break
    assert ne is not None, f"DSL weights not found in {ckpt}"
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=dev)], dim=0)
    return ne, lb, bw, bb, bv, K


def compute_repetition_rate(text):
    """Fraction of repeated bigrams."""
    words = text.split()
    if len(words) < 2:
        return 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


@torch.no_grad()
def sde_heun_generate(prompt_text, model, tokenizer, ne, lb, bw, bb, bv_infer, K,
                      snrs, noise_scale, gen_length, seed):
    """Generate text using SDE Heun integrator."""
    nd = ne.shape[1]
    msgs = [{'role': 'user', 'content': prompt_text}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    GEN = gen_length
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)
    snrs_dev = snrs.to(device).float()

    torch.manual_seed(seed)
    y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)
    ns = noise_scale

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv_infer * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        backbone_probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = backbone_probs.topk(min(SDE_TOP_K, backbone_probs.shape[-1]), dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        xh = (top_vals.unsqueeze(-1) * ne[top_idx.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)
        return xh

    actual_steps = len(snrs_dev) - 1
    for i in range(actual_steps):
        s = snrs_dev[i]
        s_next = snrs_dev[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    # Final decode
    z_final = snrs_dev[-1] * y
    cl_final = bv_infer * (z_final.float() @ K.T) + lb
    fp = F.softmax(cl_final.float(), dim=-1)
    h_final = F.linear(fp, bw, bb).to(torch.bfloat16)
    embeds_final = torch.cat([pe.to(torch.bfloat16), h_final], dim=1)
    lo = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
    toks = lo.argmax(dim=-1)[0]
    text = tokenizer.decode(toks, skip_special_tokens=True)
    return text


def compute_gen_ppl(texts, gpt2_model, gpt2_tok, dev):
    """Compute generative perplexity using GPT-2 Large, truncated to 128 words."""
    nlls = []
    for text in texts:
        words = text.split()[:128]
        text_trunc = ' '.join(words)
        if not text_trunc.strip():
            continue
        enc = gpt2_tok(text_trunc, return_tensors='pt', truncation=True, max_length=512)
        ids = enc['input_ids'].to(dev)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = gpt2_model(ids, labels=ids)
            nlls.append(out.loss.item())
    return float(np.exp(np.mean(nlls))) if nlls else float('inf')


def main():
    print("=" * 80)
    print("SDE Parameter Search v2")
    print(f"Models: {list(MODELS_CFG.keys())}")
    print(f"Grid: {len(BETA_INFERS)}x{len(STEPS_LIST)}x{len(SCHEDULES)}x{len(NOISE_SCALES)} = "
          f"{len(BETA_INFERS)*len(STEPS_LIST)*len(SCHEDULES)*len(NOISE_SCALES)} configs per model")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # Load GPT-2 Large for GenPPL
    print("Loading GPT-2 Large for GenPPL scoring...")
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
    gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2-large')

    all_results = {}

    for model_key, model_cfg in MODELS_CFG.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_key} | ckpt: {model_cfg['ckpt']} | beta_train: {model_cfg['beta_train']}")
        print(f"{'='*80}")

        # Load model
        print(f"  Loading model...")
        t_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg['ckpt'], trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        ne, lb, bw, bb, bv_trained, K = load_dsl_weights(model_cfg['ckpt'], device)
        nd = ne.shape[1]
        print(f"  Loaded in {time.time()-t_load:.1f}s | noise_dim={nd} | beta_trained={bv_trained:.4f}")

        configs_results = []
        n_configs = len(BETA_INFERS) * len(STEPS_LIST) * len(SCHEDULES) * len(NOISE_SCALES)
        config_idx = 0
        t_model_start = time.time()

        for beta_infer in BETA_INFERS:
            for steps in STEPS_LIST:
                for schedule in SCHEDULES:
                    for noise_scale in NOISE_SCALES:
                        config_idx += 1
                        params = {
                            'beta_infer': beta_infer,
                            'steps': steps,
                            'schedule': schedule,
                            'noise_scale': noise_scale,
                        }

                        snrs = build_snr_schedule(schedule, steps, SNR_MAX)
                        texts = []
                        t0 = time.time()

                        for prompt in PROMPTS:
                            try:
                                text = sde_heun_generate(
                                    prompt, model, tokenizer, ne, lb, bw, bb,
                                    beta_infer, K, snrs, noise_scale, GEN_LENGTH, SEED
                                )
                                texts.append(text)
                            except Exception as e:
                                texts.append(f"ERROR: {e}")

                        gen_time = time.time() - t0

                        # Compute metrics
                        valid_texts = [t for t in texts if not t.startswith("ERROR")]
                        gen_ppl = compute_gen_ppl(valid_texts, gpt2, gpt2_tok, device)
                        avg_len = np.mean([len(t.split()) for t in valid_texts]) if valid_texts else 0
                        rep_rate = np.mean([compute_repetition_rate(t) for t in valid_texts]) if valid_texts else 1.0

                        result = {
                            'params': params,
                            'gen_ppl': round(gen_ppl, 2),
                            'avg_len': round(float(avg_len), 1),
                            'rep_rate': round(float(rep_rate), 4),
                            'gen_time': round(gen_time, 1),
                            'n_valid': len(valid_texts),
                        }
                        configs_results.append(result)

                        elapsed = time.time() - t_model_start
                        eta = elapsed / config_idx * (n_configs - config_idx)
                        print(f"  [{config_idx}/{n_configs}] bi={beta_infer} s={steps} "
                              f"sch={schedule} ns={noise_scale} | "
                              f"PPL={gen_ppl:.1f} len={avg_len:.0f} rep={rep_rate:.3f} | "
                              f"{gen_time:.0f}s | ETA {eta:.0f}s")

        # Sort by GenPPL (lower is better)
        configs_results.sort(key=lambda x: x['gen_ppl'])
        best = configs_results[0]

        all_results[model_key] = {
            'beta_trained': bv_trained,
            'embed_type': model_cfg['embed'],
            'best': {
                'beta_infer': best['params']['beta_infer'],
                'steps': best['params']['steps'],
                'schedule': best['params']['schedule'],
                'noise_scale': best['params']['noise_scale'],
                'gen_ppl': best['gen_ppl'],
            },
            'all_results': configs_results,
        }

        # Print top-3
        print(f"\n  === Top 3 for {model_key} ===")
        for i, r in enumerate(configs_results[:3]):
            p = r['params']
            print(f"  #{i+1} PPL={r['gen_ppl']:.2f} | bi={p['beta_infer']} s={p['steps']} "
                  f"sch={p['schedule']} ns={p['noise_scale']} | len={r['avg_len']:.0f} rep={r['rep_rate']:.3f}")

        # Cleanup
        del model, ne, lb, bw, bb, K
        torch.cuda.empty_cache()
        print(f"  Model {model_key} done, GPU cache cleared.")

    # Save results
    os.makedirs('eval_results', exist_ok=True)
    out_path = 'eval_results/sde_param_search.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for model_key, data in all_results.items():
        print(f"\n{model_key} (beta_trained={data['beta_trained']:.4f}, embed={data['embed_type']}):")
        top3 = data['all_results'][:3]
        for i, r in enumerate(top3):
            p = r['params']
            print(f"  #{i+1} GenPPL={r['gen_ppl']:.2f} | beta_infer={p['beta_infer']} steps={p['steps']} "
                  f"schedule={p['schedule']} noise_scale={p['noise_scale']} | "
                  f"avg_len={r['avg_len']:.0f} rep_rate={r['rep_rate']:.3f}")


if __name__ == '__main__':
    main()
