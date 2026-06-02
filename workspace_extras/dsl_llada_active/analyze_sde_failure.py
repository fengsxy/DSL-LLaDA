#!/usr/bin/env python3
"""Analyze SDE generation failure modes: step-by-step trace of what happens
during SDE vs remasking on GSM8K problems.

Compares b1 (random embed) vs sem_b05 (semantic embed) dynamics.
"""

import sys, os, math, json
import torch
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda:0"  # Use with CUDA_VISIBLE_DEVICES=7
MASK_ID = 126336

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)


# ── Model loading ──────────────────────────────────────────────────────────

def load_model_with_dsl(checkpoint_dir, dsl_config, device=DEVICE):
    """Load LLaDA + attach DSL modules from checkpoint."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        checkpoint_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Set env vars BEFORE importing dsl_modules (module-level constants)
    noise_dim = dsl_config.get("noise_dim", 100)
    os.environ["DSL_NOISE_DIM"] = str(noise_dim)
    os.environ["DSL_BETA_INIT"] = str(dsl_config.get("beta_init", 1.0))
    os.environ["DSL_NOISE_INIT"] = str(dsl_config.get("noise_init", "random"))
    if dsl_config.get("noise_init") == "ae_contrastive":
        os.environ["DSL_AE_EMBED_PATH"] = os.environ.get(
            "DSL_AE_EMBED_PATH", "results/wte_ae_contrastive_embedding.pt"
        )

    # Force reimport with correct env vars
    import importlib
    import dsl_modules as _dm
    importlib.reload(_dm)
    _dm.attach_dsl_modules(model, noise_dim=noise_dim, freeze_ff_out=True)

    # Load trained weights
    import safetensors.torch, glob
    shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
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
    print(f"  Loaded: {checkpoint_dir}")
    print(f"  beta={model.converter.beta.item():.3f}, noise_dim={model.noise_embed.weight.shape[1]}")
    return model


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )


# ── SDE with step-by-step tracing ─────────────────────────────────────────

@torch.no_grad()
def sde_trace(model, tokenizer, prompt_text, device=DEVICE, sde_config=None):
    """Run SDE generation with detailed per-step diagnostics."""
    if sde_config is None:
        sde_config = {}

    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    B, P = prompt_ids.shape
    gen_length = 256
    seq_len = P + gen_length

    steps = sde_config.get("steps", 32)
    schedule = sde_config.get("schedule", [3, 100])
    noise_scale = sde_config.get("noise_scale", 0.05)
    beta_infer = sde_config.get("beta_infer", 2.0)
    solver = sde_config.get("solver", "heun")

    noise_dim = model.noise_embed.weight.shape[1]
    vocab_size = model.noise_embed.weight.shape[0]

    # Override beta
    orig_beta = model.converter.beta.data.clone()
    model.converter.beta.data.fill_(beta_infer)

    trace = {
        "steps": steps, "solver": solver, "beta_infer": beta_infer,
        "noise_scale": noise_scale, "schedule": schedule,
        "step_data": [],
    }

    try:
        snr_lo, snr_hi = schedule
        snr_schedule = torch.exp(
            torch.linspace(math.log(snr_lo), math.log(snr_hi), steps + 1, device=device)
        )

        # Init
        prompt_snr = torch.full((1, P), snr_hi, device=device)
        from dsl_modules import noisy_embedding
        z_prompt = noisy_embedding(model.noise_embed, prompt_ids, prompt_snr).float()
        z_gen = torch.randn(1, gen_length, noise_dim, dtype=torch.float32, device=device)

        prev_tokens = None

        for step_idx in range(steps):
            snr_t = snr_schedule[step_idx]
            snr_next = snr_schedule[step_idx + 1]

            z_full = torch.cat([z_prompt, z_gen], dim=1)

            # ── Converter analysis ──
            converter_probs = model.converter.get_token_probs(z_full[:, P:])  # (1, gen_len, V+1)
            conv_max_prob, conv_tokens = converter_probs[:, :, :vocab_size].max(dim=-1)  # exclude mask slot
            conv_mask_prob = converter_probs[:, :, -1]  # mask slot probability

            # ── Backbone forward ──
            h = model.converter(z_full).to(dtype=torch.bfloat16)
            out = model(
                input_ids=torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device),
                inputs_embeds=h,
            )
            logits = out.logits.float()
            backbone_probs = F.softmax(logits[:, P:, :vocab_size], dim=-1)
            backbone_max_prob, backbone_tokens = backbone_probs.max(dim=-1)

            # ── Predicted clean embedding ──
            probs_for_drift = F.softmax(logits[:, P:, :vocab_size], dim=-1)
            embed_w = model.noise_embed.weight.float()
            x_hat = torch.matmul(probs_for_drift, embed_w)

            # ── Token changes ──
            if prev_tokens is not None:
                n_changed = (backbone_tokens[0] != prev_tokens).sum().item()
            else:
                n_changed = gen_length
            prev_tokens = backbone_tokens[0].clone()

            # ── z_gen magnitude and drift analysis ──
            z_gen_norm = z_gen.norm(dim=-1).mean().item()
            x_hat_norm = x_hat.norm(dim=-1).mean().item()
            drift_vec = x_hat - z_gen
            drift_norm = drift_vec.norm(dim=-1).mean().item()

            # ── Cosine similarity between z_gen and x_hat (are we converging?) ──
            cos_sim = F.cosine_similarity(z_gen[0], x_hat[0], dim=-1).mean().item()

            # ── Token agreement: converter vs backbone ──
            agree = (conv_tokens[0] == backbone_tokens[0]).float().mean().item()

            # ── Confidence distribution ──
            conf_quartiles = torch.quantile(
                backbone_max_prob[0].float(),
                torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
            ).cpu().tolist()

            # ── Entropy of backbone predictions ──
            entropy = -(backbone_probs * torch.log(backbone_probs + 1e-10)).sum(dim=-1).mean().item()

            step_info = {
                "step": step_idx,
                "snr_t": snr_t.item(),
                "snr_next": snr_next.item(),
                # Converter
                "conv_max_prob_mean": conv_max_prob.mean().item(),
                "conv_mask_prob_mean": conv_mask_prob.mean().item(),
                "conv_top_token_unique": conv_tokens[0].unique().numel(),
                # Backbone
                "backbone_conf_mean": backbone_max_prob.mean().item(),
                "backbone_conf_quartiles": conf_quartiles,
                "backbone_entropy": entropy,
                "backbone_unique_tokens": backbone_tokens[0].unique().numel(),
                # Dynamics
                "n_changed": n_changed,
                "pct_changed": n_changed / gen_length * 100,
                "z_gen_norm": z_gen_norm,
                "x_hat_norm": x_hat_norm,
                "drift_norm": drift_norm,
                "cos_sim_z_xhat": cos_sim,
                "converter_backbone_agree": agree,
            }
            trace["step_data"].append(step_info)

            # ── SDE step (Euler-Maruyama, no Heun for trace simplicity) ──
            dt = (snr_next - snr_t) / snr_hi
            drift = (x_hat - z_gen) * dt.abs()

            if solver == "heun" and step_idx < steps - 1:
                z_gen_pred = z_gen + drift + noise_scale * math.sqrt(abs(dt.item())) * torch.randn_like(z_gen)
                z_full_pred = torch.cat([z_prompt, z_gen_pred], dim=1)
                h_pred = model.converter(z_full_pred).to(dtype=torch.bfloat16)
                out_pred = model(
                    input_ids=torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device),
                    inputs_embeds=h_pred,
                )
                logits_pred = out_pred.logits.float()
                probs_pred = F.softmax(logits_pred[:, P:, :vocab_size], dim=-1)
                x_hat_pred = torch.matmul(probs_pred, embed_w)
                drift_pred = (x_hat_pred - z_gen_pred) * dt.abs()
                z_gen = z_gen + 0.5 * (drift + drift_pred) + noise_scale * math.sqrt(abs(dt.item())) * torch.randn_like(z_gen)
            else:
                z_gen = z_gen + drift + noise_scale * math.sqrt(abs(dt.item())) * torch.randn_like(z_gen)

        # ── Final decode ──
        z_full = torch.cat([z_prompt, z_gen], dim=1)
        h = model.converter(z_full).to(dtype=torch.bfloat16)
        out = model(
            input_ids=torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device),
            inputs_embeds=h,
        )
        final_logits = out.logits.float()
        final_tokens = final_logits[0, P:].argmax(dim=-1)
        final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
        trace["final_text"] = final_text

        # Final confidence
        final_probs = F.softmax(final_logits[:, P:, :vocab_size], dim=-1)
        final_conf = final_probs.max(dim=-1).values
        trace["final_conf_mean"] = final_conf.mean().item()

        # Final converter analysis
        final_conv_probs = model.converter.get_token_probs(z_full[:, P:])
        final_conv_max = final_conv_probs[:, :, :vocab_size].max(dim=-1)
        trace["final_conv_conf_mean"] = final_conv_max.values.mean().item()
        trace["final_conv_backbone_agree"] = (final_conv_max.indices[0] == final_tokens).float().mean().item()

    finally:
        model.converter.beta.data.copy_(orig_beta)

    return trace


# ── Remasking generation with trace ───────────────────────────────────────

@torch.no_grad()
def remask_trace(model, tokenizer, prompt_text, device=DEVICE, steps=64):
    """Run remasking generation with per-step diagnostics."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    B, P = prompt_ids.shape
    gen_length = 256

    # Init: all mask tokens
    gen_ids = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, gen_ids], dim=1)

    trace = {"steps": steps, "step_data": []}
    prev_tokens = None

    for step_idx in range(steps):
        # Determine how many to unmask this step
        mask_positions = (input_ids[0, P:] == MASK_ID)
        n_masked = mask_positions.sum().item()
        if n_masked == 0:
            break

        # Forward pass
        out = model(input_ids=input_ids)
        logits = out.logits.float()

        # Get predictions for masked positions
        gen_logits = logits[:, P:, :]
        probs = F.softmax(gen_logits, dim=-1)
        max_prob, pred_tokens = probs.max(dim=-1)

        # How many to unmask this step (linear schedule)
        n_unmask = max(1, n_masked // max(1, steps - step_idx))

        # Select by confidence (only from masked positions)
        conf = max_prob[0].clone()
        conf[~mask_positions] = -1  # don't re-select unmasked
        _, top_idx = conf.topk(n_unmask)

        # Current tokens (for change tracking)
        current_tokens = pred_tokens[0].clone()
        current_tokens[~mask_positions] = input_ids[0, P:][~mask_positions]

        if prev_tokens is not None:
            n_changed = (current_tokens != prev_tokens).sum().item()
        else:
            n_changed = gen_length
        prev_tokens = current_tokens.clone()

        # Confidence stats
        masked_conf = max_prob[0][mask_positions]
        conf_quartiles = torch.quantile(
            masked_conf.float(),
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
        ).cpu().tolist() if masked_conf.numel() > 0 else [0]*5

        step_info = {
            "step": step_idx,
            "n_masked": n_masked,
            "n_unmask": n_unmask,
            "backbone_conf_mean": masked_conf.mean().item() if masked_conf.numel() > 0 else 0,
            "backbone_conf_quartiles": conf_quartiles,
            "backbone_unique_tokens": pred_tokens[0][mask_positions].unique().numel() if mask_positions.any() else 0,
            "n_changed": n_changed,
            "pct_changed": n_changed / gen_length * 100,
        }
        trace["step_data"].append(step_info)

        # Apply unmasking
        for idx in top_idx:
            input_ids[0, P + idx] = pred_tokens[0, idx]

    # Final output
    final_text = tokenizer.decode(input_ids[0, P:], skip_special_tokens=True)
    trace["final_text"] = final_text

    # Final confidence (full forward)
    out = model(input_ids=input_ids)
    final_probs = F.softmax(out.logits[:, P:].float(), dim=-1)
    trace["final_conf_mean"] = final_probs.max(dim=-1).values.mean().item()

    return trace


# ── Pretty printing ───────────────────────────────────────────────────────

def print_sde_trace(trace, model_name, problem_id):
    print(f"\n{'='*80}")
    print(f"SDE TRACE: {model_name} | Problem {problem_id}")
    print(f"  steps={trace['steps']}, solver={trace['solver']}, "
          f"beta_infer={trace['beta_infer']}, noise_scale={trace['noise_scale']}")
    print(f"  schedule={trace['schedule']}")
    print(f"{'='*80}")

    print(f"\n{'Step':>4} | {'SNR':>7} | {'ConvConf':>8} {'MaskP':>6} | "
          f"{'BkConf':>7} {'Entropy':>8} | {'Changed':>7} | "
          f"{'||z||':>6} {'||x̂||':>6} {'cos(z,x̂)':>8} | {'Agree':>5} {'UniTok':>6}")
    print("-" * 110)

    for s in trace["step_data"]:
        print(f"{s['step']:4d} | {s['snr_t']:7.1f} | "
              f"{s['conv_max_prob_mean']:8.4f} {s['conv_mask_prob_mean']:6.3f} | "
              f"{s['backbone_conf_mean']:7.4f} {s['backbone_entropy']:8.2f} | "
              f"{s['pct_changed']:6.1f}% | "
              f"{s['z_gen_norm']:6.1f} {s['x_hat_norm']:6.1f} {s['cos_sim_z_xhat']:8.4f} | "
              f"{s['converter_backbone_agree']:5.2f} {s['backbone_unique_tokens']:6d}")

    print(f"\nFinal: conv_conf={trace.get('final_conv_conf_mean',0):.4f}, "
          f"backbone_conf={trace.get('final_conf_mean',0):.4f}, "
          f"conv-backbone agree={trace.get('final_conv_backbone_agree',0):.2f}")
    print(f"\nGenerated text (first 300 chars):")
    print(f"  {trace['final_text'][:300]}")


def print_remask_trace(trace, model_name, problem_id):
    print(f"\n{'='*80}")
    print(f"REMASK TRACE: {model_name} | Problem {problem_id}")
    print(f"  steps={trace['steps']}")
    print(f"{'='*80}")

    print(f"\n{'Step':>4} | {'Masked':>6} {'Unmask':>6} | {'BkConf':>7} | {'Changed':>7} | {'UniTok':>6}")
    print("-" * 60)

    for s in trace["step_data"][:20]:  # first 20 steps
        print(f"{s['step']:4d} | {s['n_masked']:6d} {s['n_unmask']:6d} | "
              f"{s['backbone_conf_mean']:7.4f} | "
              f"{s['pct_changed']:6.1f}% | {s['backbone_unique_tokens']:6d}")
    if len(trace["step_data"]) > 20:
        print(f"  ... ({len(trace['step_data']) - 20} more steps)")

    print(f"\nFinal confidence: {trace.get('final_conf_mean',0):.4f}")
    print(f"\nGenerated text (first 300 chars):")
    print(f"  {trace['final_text'][:300]}")


# ── Summary comparison ────────────────────────────────────────────────────

def compare_models(traces_b1, traces_sem, remask_traces):
    """Print comparative summary across models and methods."""
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY")
    print("=" * 80)

    for pid in traces_b1:
        print(f"\n--- Problem {pid} ---")

        # SDE traces
        for name, traces in [("b1", traces_b1), ("sem_b05", traces_sem)]:
            if pid not in traces:
                continue
            t = traces[pid]
            sd = t["step_data"]

            # Early vs late dynamics
            early = sd[:5]   # first 5 steps
            late = sd[-5:]   # last 5 steps

            early_change = np.mean([s["pct_changed"] for s in early])
            late_change = np.mean([s["pct_changed"] for s in late])
            early_conf = np.mean([s["backbone_conf_mean"] for s in early])
            late_conf = np.mean([s["backbone_conf_mean"] for s in late])
            early_agree = np.mean([s["converter_backbone_agree"] for s in early])
            late_agree = np.mean([s["converter_backbone_agree"] for s in late])
            early_cos = np.mean([s["cos_sim_z_xhat"] for s in early])
            late_cos = np.mean([s["cos_sim_z_xhat"] for s in late])
            early_conv_mask = np.mean([s["conv_mask_prob_mean"] for s in early])
            late_conv_mask = np.mean([s["conv_mask_prob_mean"] for s in late])

            print(f"  {name} SDE:")
            print(f"    Early(0-4): chg={early_change:.1f}% conf={early_conf:.3f} "
                  f"agree={early_agree:.2f} cos={early_cos:.3f} mask_p={early_conv_mask:.3f}")
            print(f"    Late(27-31): chg={late_change:.1f}% conf={late_conf:.3f} "
                  f"agree={late_agree:.2f} cos={late_cos:.3f} mask_p={late_conv_mask:.3f}")
            print(f"    Final: conf={t.get('final_conf_mean',0):.3f}")

        # Remasking
        if pid in remask_traces:
            t = remask_traces[pid]
            print(f"  b1 Remasking:")
            print(f"    Final: conf={t.get('final_conf_mean',0):.3f}")

        # Compare outputs
        print(f"\n  Outputs:")
        if pid in traces_b1:
            print(f"    b1 SDE:     {traces_b1[pid]['final_text'][:120]}...")
        if pid in traces_sem:
            print(f"    sem_b05 SDE: {traces_sem[pid]['final_text'][:120]}...")
        if pid in remask_traces:
            print(f"    b1 Remask:  {remask_traces[pid]['final_text'][:120]}...")


# ── Convergence analysis ──────────────────────────────────────────────────

def analyze_convergence(trace, name):
    """Deeper analysis of why SDE may fail to converge."""
    print(f"\n{'='*80}")
    print(f"CONVERGENCE ANALYSIS: {name}")
    print(f"{'='*80}")

    sd = trace["step_data"]

    # 1. Token stability: do positions settle or keep flipping?
    changes = [s["pct_changed"] for s in sd]
    print(f"\n1. Token stability (% positions changing per step):")
    print(f"   Steps 0-7:   {np.mean(changes[:8]):.1f}%")
    print(f"   Steps 8-15:  {np.mean(changes[8:16]):.1f}%")
    print(f"   Steps 16-23: {np.mean(changes[16:24]):.1f}%")
    print(f"   Steps 24-31: {np.mean(changes[24:32]):.1f}%")
    if np.mean(changes[-5:]) > 10:
        print(f"   !! PROBLEM: Still changing {np.mean(changes[-5:]):.1f}% in last 5 steps → NOT converging")

    # 2. Converter mask probability: is converter mostly outputting mask?
    mask_probs = [s["conv_mask_prob_mean"] for s in sd]
    print(f"\n2. Converter mask probability (should decrease as SNR increases):")
    print(f"   Steps 0-7:   {np.mean(mask_probs[:8]):.4f}")
    print(f"   Steps 8-15:  {np.mean(mask_probs[8:16]):.4f}")
    print(f"   Steps 16-23: {np.mean(mask_probs[16:24]):.4f}")
    print(f"   Steps 24-31: {np.mean(mask_probs[24:32]):.4f}")
    if np.mean(mask_probs[-5:]) > 0.3:
        print(f"   !! PROBLEM: Converter still {np.mean(mask_probs[-5:])*100:.1f}% mask at end → bottleneck")

    # 3. Converter-backbone agreement
    agrees = [s["converter_backbone_agree"] for s in sd]
    print(f"\n3. Converter-backbone token agreement:")
    print(f"   Steps 0-7:   {np.mean(agrees[:8]):.3f}")
    print(f"   Steps 24-31: {np.mean(agrees[24:32]):.3f}")
    if np.mean(agrees[-5:]) < 0.5:
        print(f"   !! PROBLEM: Converter and backbone disagree on {(1-np.mean(agrees[-5:]))*100:.0f}% of tokens")

    # 4. z convergence (cosine similarity z_gen vs x_hat)
    cos_sims = [s["cos_sim_z_xhat"] for s in sd]
    print(f"\n4. z-space convergence (cos(z_gen, x_hat)):")
    print(f"   Steps 0-7:   {np.mean(cos_sims[:8]):.4f}")
    print(f"   Steps 24-31: {np.mean(cos_sims[24:32]):.4f}")
    if np.mean(cos_sims[-5:]) < 0.5:
        print(f"   !! PROBLEM: z not converging to predicted clean embedding")

    # 5. Backbone entropy
    entropies = [s["backbone_entropy"] for s in sd]
    print(f"\n5. Backbone prediction entropy (lower = more confident):")
    print(f"   Steps 0-7:   {np.mean(entropies[:8]):.2f}")
    print(f"   Steps 24-31: {np.mean(entropies[24:32]):.2f}")

    # 6. Unique tokens (diversity vs repetition)
    uniques = [s["backbone_unique_tokens"] for s in sd]
    print(f"\n6. Unique predicted tokens (diversity):")
    print(f"   Steps 0-7:   {np.mean(uniques[:8]):.0f}")
    print(f"   Steps 24-31: {np.mean(uniques[24:32]):.0f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    tokenizer = load_tokenizer()

    # Load GSM8K problems
    with open("/home/ubuntu/efs/RMDM/eval_data/gsm8k_100.json") as f:
        gsm8k = json.load(f)

    # Pick 3 problems: easy, medium, hard
    problems = [
        gsm8k[6],   # "If Ann is 9 years old..." (simple arithmetic)
        gsm8k[0],   # "The girls are trying to raise money..." (multi-step)
        gsm8k[7],   # "Twenty dozen cups..." (word problem)
    ]
    problem_ids = [p["id"] for p in problems]

    # ── Model configs ──
    b1_config = {
        "path": "checkpoints/pertoken_b1_d100_1k/checkpoint-1000",
        "dsl_config": {"beta_init": 1.0, "noise_dim": 100, "noise_init": "random"},
        "sde_config": {"beta_infer": 2.0, "noise_scale": 0.05, "schedule": [3, 100],
                       "steps": 32, "solver": "heun"},
    }
    sem_config = {
        "path": "checkpoints/semantic_b05_d100_10k/checkpoint-3000",
        "dsl_config": {"beta_init": 0.5, "noise_dim": 100, "noise_init": "ae_contrastive"},
        "sde_config": {"beta_infer": 1.0, "noise_scale": 0.05, "schedule": [10, 100],
                       "steps": 32, "solver": "heun"},
    }

    # ══════════════════════════════════════════════════════════════════════
    # TASK 1: b1 model analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 80)
    print("# TASK 1: b1 model (random embed, beta=1)")
    print("#" * 80)

    os.chdir("/home/ubuntu/efs/RMDM")
    model_b1 = load_model_with_dsl(b1_config["path"], b1_config["dsl_config"])

    traces_b1_sde = {}
    traces_b1_remask = {}

    for prob in problems:
        pid = prob["id"]
        q = prob["question"]
        gold = prob["gold_answer"]
        print(f"\n\nProblem {pid}: {q[:80]}... (gold={gold})")

        # SDE trace
        trace_sde = sde_trace(model_b1, tokenizer, q, sde_config=b1_config["sde_config"])
        traces_b1_sde[pid] = trace_sde
        print_sde_trace(trace_sde, "b1", pid)
        analyze_convergence(trace_sde, f"b1 problem {pid}")

        # Remasking trace
        trace_rm = remask_trace(model_b1, tokenizer, q, steps=64)
        traces_b1_remask[pid] = trace_rm
        print_remask_trace(trace_rm, "b1", pid)

    # Free GPU memory
    del model_b1
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # TASK 2: sem_b05 model analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 80)
    print("# TASK 2: sem_b05 model (semantic embed, beta=0.5)")
    print("#" * 80)

    model_sem = load_model_with_dsl(sem_config["path"], sem_config["dsl_config"])

    traces_sem_sde = {}

    for prob in problems:
        pid = prob["id"]
        q = prob["question"]
        gold = prob["gold_answer"]
        print(f"\n\nProblem {pid}: {q[:80]}... (gold={gold})")

        trace_sde = sde_trace(model_sem, tokenizer, q, sde_config=sem_config["sde_config"])
        traces_sem_sde[pid] = trace_sde
        print_sde_trace(trace_sde, "sem_b05", pid)
        analyze_convergence(trace_sde, f"sem_b05 problem {pid}")

    del model_sem
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # COMPARATIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    compare_models(traces_b1_sde, traces_sem_sde, traces_b1_remask)

    # ══════════════════════════════════════════════════════════════════════
    # KEY DIAGNOSTIC QUESTIONS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    # Aggregate across problems for b1
    all_b1_steps = []
    for pid in traces_b1_sde:
        all_b1_steps.extend(traces_b1_sde[pid]["step_data"])

    all_sem_steps = []
    for pid in traces_sem_sde:
        all_sem_steps.extend(traces_sem_sde[pid]["step_data"])

    print("\n1. CONVERTER BOTTLENECK?")
    for name, steps in [("b1", all_b1_steps), ("sem_b05", all_sem_steps)]:
        late = [s for s in steps if s["step"] >= 24]
        avg_mask = np.mean([s["conv_mask_prob_mean"] for s in late])
        avg_conf = np.mean([s["conv_max_prob_mean"] for s in late])
        print(f"   {name}: late-step converter mask_prob={avg_mask:.4f}, max_token_prob={avg_conf:.4f}")

    print("\n2. TOKEN FLIPPING (non-convergence)?")
    for name, steps in [("b1", all_b1_steps), ("sem_b05", all_sem_steps)]:
        late = [s for s in steps if s["step"] >= 24]
        avg_chg = np.mean([s["pct_changed"] for s in late])
        print(f"   {name}: late-step change rate={avg_chg:.1f}%")

    print("\n3. BACKBONE CONFIDENCE?")
    for name, steps in [("b1", all_b1_steps), ("sem_b05", all_sem_steps)]:
        late = [s for s in steps if s["step"] >= 24]
        avg_conf = np.mean([s["backbone_conf_mean"] for s in late])
        avg_entropy = np.mean([s["backbone_entropy"] for s in late])
        print(f"   {name}: late-step backbone conf={avg_conf:.3f}, entropy={avg_entropy:.1f}")

    print("\n4. SEMANTIC EMBED SMOOTHER TRAJECTORIES?")
    for name, steps in [("b1", all_b1_steps), ("sem_b05", all_sem_steps)]:
        early = [s for s in steps if s["step"] < 8]
        late = [s for s in steps if s["step"] >= 24]
        early_cos = np.mean([s["cos_sim_z_xhat"] for s in early])
        late_cos = np.mean([s["cos_sim_z_xhat"] for s in late])
        print(f"   {name}: cos(z,x_hat) early={early_cos:.4f} → late={late_cos:.4f}")

    print("\n5. CONVERTER-BACKBONE ALIGNMENT?")
    for name, steps in [("b1", all_b1_steps), ("sem_b05", all_sem_steps)]:
        late = [s for s in steps if s["step"] >= 24]
        avg_agree = np.mean([s["converter_backbone_agree"] for s in late])
        print(f"   {name}: late-step agreement={avg_agree:.3f}")


if __name__ == "__main__":
    main()
