"""SDE vs Remask: full NFE-matched comparison for paper.

Runs both SDE (β=1 model) and Standard Remask (Original + β=1)
across NFE = 8, 16, 32, 64, 128 on 100 prompts.
Computes: GenPPL, D2, D3, Rep, AvgLen.

Usage:
    # SDE configs on GPU 6
    python dsl_llada/eval_sde_vs_remask_full.py --mode sde --gpu 6

    # Remask configs on GPU 7
    python dsl_llada/eval_sde_vs_remask_full.py --mode remask --gpu 7

    # Just compute metrics on existing generated texts
    python dsl_llada/eval_sde_vs_remask_full.py --mode metrics --gpu 6
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_root, "LLaDA"))
from generate import generate

MASK_ID = 126336
EVAL_DATA_DIR = os.path.join(_root, "eval_data")
OUTPUT_DIR = os.path.join(_root, "eval_results", "sde_vs_remask_full")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_prompts():
    with open(os.path.join(EVAL_DATA_DIR, "sde_prompts_100.json")) as f:
        return json.load(f)


def compute_gen_ppl(texts, device):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
    nlls = []
    for text in texts:
        words = text.split()[:128]
        truncated = " ".join(words)
        if not truncated.strip():
            continue
        enc = gpt2_tok(truncated, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = gpt2_model(input_ids, labels=input_ids)
            nlls.append(out.loss.item())
    del gpt2_model, gpt2_tok
    torch.cuda.empty_cache()
    return float(np.exp(np.mean(nlls))) if nlls else float("nan")


def compute_distinct_n(texts, n):
    total_ngrams = []
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            total_ngrams.append(tuple(words[i:i+n]))
    return len(set(total_ngrams)) / len(total_ngrams) if total_ngrams else 0.0


def compute_rep_rate(texts):
    total = repeated = 0
    for text in texts:
        words = text.split()
        for i in range(1, len(words)):
            total += 1
            if words[i] == words[i-1]:
                repeated += 1
    return repeated / total if total > 0 else 0.0


def attach_dsl(model, checkpoint_dir, device, dsl_config):
    """Attach DSL converter/noise_embed from checkpoint safetensors."""
    import glob as globmod
    import safetensors.torch

    if hasattr(model, "noise_embed"):
        return

    if dsl_config:
        os.environ["DSL_NOISE_DIM"] = str(dsl_config.get("noise_dim", 48))
        os.environ["DSL_BETA_INIT"] = str(dsl_config.get("beta_init", 5.0))
        os.environ["DSL_NOISE_INIT"] = str(dsl_config.get("noise_init", "random"))

    sys.path.insert(0, _script_dir)
    from dsl_modules import attach_dsl_modules
    attach_dsl_modules(model, freeze_ff_out=True)

    shard_files = sorted(globmod.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
    for sf in shard_files:
        sd = safetensors.torch.load_file(sf, device=str(device))
        for k, v in sd.items():
            if k.startswith("converter.") or k.startswith("noise_embed."):
                parts = k.split(".")
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
    for name, param in model.named_parameters():
        if param.device != torch.device(device):
            param.data = param.data.to(device)


def load_model(model_key, device):
    """Load model. Returns (model, tokenizer, entry)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    registry = json.load(open(os.path.join(_root, "eval_results", "registry.json")))
    entry = registry[model_key]
    ckpt = entry["path"]

    # Resolve local paths
    if entry.get("type") == "local":
        ckpt = os.path.join(_root, ckpt)

    print(f"Loading {model_key} from {ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    # Attach DSL modules AND load trained weights
    if entry.get("dsl"):
        dsl_cfg = entry.get("dsl_config", {})
        attach_dsl(model, ckpt, device, dsl_cfg)
        print(f"  DSL modules attached (beta={model.converter.beta.item():.4f})")
        # Verify weights were loaded (beta should not be default)
        if abs(model.converter.beta.item() - dsl_cfg.get("beta_init", 5.0)) < 0.01:
            print("  WARNING: beta equals init value, weights may not have loaded!")

    return model, tokenizer, entry


def generate_remask_text(model, tokenizer, prompt_text, device, steps=64,
                         block_length=256, suppress_eos=False):
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        out = generate(
            model, input_ids, torch.ones_like(input_ids),
            steps=steps, gen_length=256, block_length=block_length,
            temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
            eos_suppress_ratio=1.0 if suppress_eos else 0.0,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def generate_sde_text(model, tokenizer, prompt_text, device,
                      steps=32, beta_infer=2.0, noise_scale=0.05,
                      schedule="sensitive", solver="heun", gen_length=256):
    """SDE generation using the correct drift/diffusion formulation from sde_sweep_single.py.

    Key differences from the broken eval_unified.py version:
    - drift = (x_hat - y) / s  (not * dt)
    - noise = ns / s  (not fixed scale)
    - y initialized as unit-norm (not raw randn)
    - Proper Heun corrector: 0.5*(f+f_e)*ds + 0.5*(g+g_e)*dW
    """
    TOP_K = 50

    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    P = prompt_ids.shape[1]

    # Extract converter components for direct computation
    ne = model.noise_embed.weight.float()  # (V, d)
    nd = ne.shape[1]
    lb = model.converter.logit_bias.float()  # (V+1,)
    bw = model.converter.backbone_embedding.weight.float()  # (d_model, V+1)
    bb = model.converter.backbone_embedding.bias.float()  # (d_model,)
    # K needs V+1 rows: V token embeds + 1 mask slot (zero vector)
    K = torch.cat([ne, torch.zeros(1, nd, device=device)], dim=0)  # (V+1, d)
    bv = beta_infer

    # Prompt embedding from backbone wte
    pe = model.model.transformer.wte(prompt_ids).float()
    dummy = torch.zeros(1, P + gen_length, dtype=torch.long, device=device)

    # SNR schedule
    snr_min = 0.01
    if schedule == "sensitive":
        n1, n2 = max(1, int(steps * 0.05)), max(1, int(steps * 0.90))
        n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(100), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(
            torch.linspace(math.log(snr_min), math.log(100), steps + 1)
        ).to(device)

    # Init y as unit-norm random vectors
    torch.manual_seed(42)
    y = F.normalize(torch.randn(1, gen_length, nd, device=device), dim=-1)

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h = F.linear(probs, bw, bb).to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, P:, :].float()
        bp = F.softmax(logits, dim=-1)
        tv, ti = bp.topk(min(TOP_K, bp.shape[-1]), dim=-1)
        tv = tv / tv.sum(dim=-1, keepdim=True)
        return (tv.unsqueeze(-1) * ne[ti.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)

    ns = noise_scale
    for i in range(len(snrs) - 1):
        s, s_next = snrs[i], snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)
        xh = get_xhat(y, s)
        f = (xh - y) / s
        g = ns / s
        # Heun predictor
        y_euler = y + f * ds + g * dW
        # Heun corrector
        xh_e = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

    # Final decode
    z_f = snrs[-1] * y
    cl_f = bv * (z_f.float() @ K.T) + lb
    pf = F.softmax(cl_f.float(), dim=-1)
    hf = F.linear(pf, bw, bb).to(torch.bfloat16)
    ef = torch.cat([pe.to(torch.bfloat16), hf], dim=1)
    with torch.no_grad():
        lo = model(input_ids=dummy, inputs_embeds=ef).logits[:, P:, :].float()
    text = tokenizer.decode(lo.argmax(dim=-1)[0], skip_special_tokens=True)

    nfe = steps * 2  # Heun always
    return text, nfe


def run_and_save(config_name, generate_fn, prompts, output_path):
    """Generate texts and save. Skip if output already exists."""
    if os.path.exists(output_path):
        print(f"  {config_name}: already exists, loading...")
        with open(output_path) as f:
            return json.load(f)

    print(f"\n  Generating: {config_name}")
    texts = []
    t0 = time.time()
    for item in tqdm(prompts, desc=config_name):
        try:
            result = generate_fn(item["prompt"])
            if isinstance(result, tuple):
                result = result[0]
            texts.append(result)
        except Exception as e:
            print(f"    Error: {e}")
            texts.append("")
    elapsed = time.time() - t0

    data = {"config": config_name, "texts": texts, "n": len(texts), "time_s": round(elapsed, 1)}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Done: {len(texts)} texts in {elapsed:.0f}s")
    return data


def compute_all_metrics(texts, device):
    valid = [t for t in texts if t.strip()]
    if not valid:
        return {"gen_ppl": None, "d2": 0, "d3": 0, "rep_rate": 0, "avg_len": 0, "n_valid": 0}

    ppl = compute_gen_ppl(valid, device)
    return {
        "gen_ppl": round(ppl, 2) if not math.isnan(ppl) else None,
        "d2": round(compute_distinct_n(valid, 2), 4),
        "d3": round(compute_distinct_n(valid, 3), 4),
        "rep_rate": round(compute_rep_rate(valid), 4),
        "avg_len": round(float(np.mean([len(t.split()) for t in valid])), 1),
        "n_valid": len(valid),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sde", "remask", "metrics", "all"], default="all")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--nfe", type=str, default="8,16,32,64,128",
                        help="Comma-separated NFE values")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    nfe_list = [int(x) for x in args.nfe.split(",")]
    prompts = load_prompts()

    # ====== SDE generation (β=1 model) ======
    if args.mode in ("sde", "all"):
        model, tokenizer, entry = load_model("b1", device)

        for nfe in nfe_list:
            steps = nfe // 2  # Heun solver: NFE = steps * 2
            if steps < 1:
                continue

            # Adaptive noise scale (from sde_param_rules)
            if nfe <= 8:
                ns = 0.1
            elif nfe <= 16:
                ns = 0.05
            elif nfe <= 32:
                ns = 0.01
            else:
                ns = 0.005

            config_name = f"b1_sde_nfe{nfe}"
            path = os.path.join(OUTPUT_DIR, f"{config_name}.json")

            def gen_fn(prompt, _steps=steps, _ns=ns):
                return generate_sde_text(
                    model, tokenizer, prompt, device,
                    steps=_steps, beta_infer=2.0, noise_scale=_ns,
                    schedule="sensitive", solver="heun"
                )

            run_and_save(config_name, gen_fn, prompts, path)

        del model
        torch.cuda.empty_cache()

    # ====== Remask generation ======
    if args.mode in ("remask", "all"):
        # Original model
        model, tokenizer, entry = load_model("original", device)

        for nfe in nfe_list:
            # remask_free: NFE = steps (1 forward per step)
            steps = nfe
            config_name = f"original_remask_nfe{nfe}"
            path = os.path.join(OUTPUT_DIR, f"{config_name}.json")

            def gen_fn(prompt, _steps=steps):
                return generate_remask_text(model, tokenizer, prompt, device, steps=_steps)

            run_and_save(config_name, gen_fn, prompts, path)

        # Also block32+noEOS at key NFEs
        for nfe in [8, 16, 32, 64]:
            steps = nfe
            config_name = f"original_b32noEOS_nfe{nfe}"
            path = os.path.join(OUTPUT_DIR, f"{config_name}.json")

            def gen_fn(prompt, _steps=steps):
                return generate_remask_text(model, tokenizer, prompt, device,
                                           steps=_steps, block_length=32, suppress_eos=True)

            run_and_save(config_name, gen_fn, prompts, path)

        del model
        torch.cuda.empty_cache()

        # β=1 model remask
        model, tokenizer, entry = load_model("b1", device)

        for nfe in nfe_list:
            steps = nfe
            config_name = f"b1_remask_nfe{nfe}"
            path = os.path.join(OUTPUT_DIR, f"{config_name}.json")

            def gen_fn(prompt, _steps=steps):
                return generate_remask_text(model, tokenizer, prompt, device, steps=_steps)

            run_and_save(config_name, gen_fn, prompts, path)

        del model
        torch.cuda.empty_cache()

    # ====== Compute metrics ======
    if args.mode in ("metrics", "all"):
        print("\n" + "=" * 60)
        print("Computing metrics for all generated texts")
        print("=" * 60)

        results = {}
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            if not fname.endswith(".json") or fname == "summary.json":
                continue
            path = os.path.join(OUTPUT_DIR, fname)
            data = json.load(open(path))
            texts = data.get("texts", [])
            config = data.get("config", fname.replace(".json", ""))

            print(f"\n  {config}: {len(texts)} texts")
            metrics = compute_all_metrics(texts, device)
            results[config] = metrics
            print(f"    PPL={metrics['gen_ppl']} D2={metrics['d2']} D3={metrics['d3']} "
                  f"Rep={metrics['rep_rate']} Len={metrics['avg_len']}")

        # Save summary
        summary_path = os.path.join(OUTPUT_DIR, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

        # Print LaTeX-ready table
        print("\n" + "=" * 60)
        print("LaTeX table:")
        print("=" * 60)
        print("NFE & Model & Method & GenPPL & D2 & Rep & Len \\\\")
        print("\\midrule")
        for config, m in sorted(results.items()):
            # Parse NFE from config name
            nfe = config.split("nfe")[-1] if "nfe" in config else "?"
            print(f"{nfe:>4} & {config:30s} & {m['gen_ppl']:>7} & {m['d2']:.3f} & "
                  f"{m['rep_rate']:.3f} & {m['avg_len']:>5.0f} \\\\")


if __name__ == "__main__":
    main()
