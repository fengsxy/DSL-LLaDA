import os
from typing import Optional
import numpy as np

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

# Match train.py speed knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Project imports
from dsl.dsl import build_dsl_from_cfg
from dataloader import get_tokenizer, get_dataloaders
from dsl.utils import load_checkpoint, pick_device, prepare_batch
from dsl.metrics import compute_metrics, nll, generative_perplexity
from dsl.snrs import build_snr_path, build_discrete_path_from_kw
import dsl.samplers as samplers


def resolve_ckpt_path(cfg: DictConfig) -> str:
    """Resolve a checkpoint path from eval.ckpt """
    if getattr(cfg, "eval", None) and getattr(cfg.eval, "ckpt", None):
        p = os.path.expanduser(os.path.expandvars(cfg.eval.ckpt))
        if os.path.isfile(p):
            return p
    print("[eval] cfg.eval.ckpt not specified or found")

def evaluate_nll(model, val_loader, snr_path, cfg, pad_id, device, logger):
    val_ce_sum = 0.0
    cum_tok = 0
    nll_sum = nll_diff_sum = nll_recon_sum = 0.0
    roar_sum = 0.0
    nll_max_snr_sum = 0.0

    with torch.no_grad():
        max_batches = getattr(cfg.eval, "max_batches", None)
        for i, batch in enumerate(val_loader):
            input_ids, valid_tokens = prepare_batch(batch, pad_id, device)

            m = compute_metrics(
                model,
                input_ids,
                valid_tokens,
                snr_path,
                nll_n_rep=cfg.eval.nll_n_rep,
                roar_n_rep=cfg.eval.roar_n_rep,
                nll_int_steps=cfg.eval.nll_int_steps,
            )

            # Token-weighted accumulation for all metrics
            ntok = int(m["ntok"])  # non-pad tokens in this batch
            val_ce_sum += m["ce"] * ntok
            nll_sum += m["nll"] * ntok
            nll_diff_sum += m["nll_diff"] * ntok
            nll_recon_sum += m["nll_recon"] * ntok
            roar_sum += m["nll_roar"] * ntok
            nll_max_snr_sum += m["nll_max_snr"] * ntok
            cum_tok += ntok

            total_batches = max_batches if max_batches is not None else len(val_loader)
            print(f"[eval] Processed {i + 1} / {total_batches} batches", end="\r")
            if max_batches is not None and (i + 1) >= int(max_batches):
                break

    val_ce = val_ce_sum / max(1, cum_tok)
    mean_val_nll = nll_sum / max(1, cum_tok)
    mean_val_nll_diff = nll_diff_sum / max(1, cum_tok)
    mean_val_nll_recon = nll_recon_sum / max(1, cum_tok)
    mean_val_roar = roar_sum / max(1, cum_tok)
    mean_val_nll_max_snr = nll_max_snr_sum / max(1, cum_tok)

    print(
        f"[eval] loss={val_ce:.6f}  nll={mean_val_nll:.6f}  "
        f"nll_diff={mean_val_nll_diff:.6f}  nll_recon={mean_val_nll_recon:.6f}  "
        f"roar={mean_val_roar:.6f}  nll_max_snr={mean_val_nll_max_snr:.6f}  tokens={cum_tok}"
    )

    if logger is not None:
        wandb.log(
            {
                "eval/ce": val_ce,
                "eval/nll_best": mean_val_nll,
                "eval/nll_max_snr": mean_val_nll_max_snr,
                "eval/nll_roar": mean_val_roar,
                "eval/nll_diff": mean_val_nll_diff,
                "eval/nll_recon": mean_val_nll_recon,
                "eval/tokens": cum_tok,
            }
        )

        # Viz curves on one batch.
        _, _, _, curves = nll(
            model, input_ids, valid_tokens,
            n_rep=5, int_steps=50,
            return_curves=True
        )

        snr_all = curves["snr_all"].detach().cpu().numpy()
        integ = curves["integrand_bpt"].detach().cpu().numpy()
        reconst = curves["reconst_bpt"].detach().cpu().numpy()
        snr_pos = curves["snr_pos"].detach().cpu().numpy()
        bound = curves["bound_curve_bpt"].detach().cpu().numpy()

        # Raw arrays (useful for downstream analysis/panels)
        wandb.log({
            "nll/snr_all": snr_all,
            "nll/integrand_bpt": integ,
            "nll/reconst_bpt": reconst,
            "nll/snr_pos": snr_pos,
            "nll/bound_curve_bpt": bound,
        })

        # Pretty plot: one table with multiple strokes
        tbl = wandb.Table(columns=["snr", "value", "curve"])
        for s, v in zip(snr_all, integ):
            tbl.add_data(float(s), float(v), "integrand")
        for s, v in zip(snr_all, reconst):
            tbl.add_data(float(s), float(v), "reconst")
        for s, v in zip(snr_pos, bound):
            tbl.add_data(float(s), float(v), "bound")

        wandb.log({
            "nll/curves": wandb.plot.line(
                tbl, x="snr", y="value", title="NLL Curves vs SNR", stroke="curve"
            )
        })

        # --- Separate table and plot for reconst curve only ---
        tbl_reconst = wandb.Table(columns=["snr", "reconst"])
        for s, v in zip(snr_all, reconst):
            tbl_reconst.add_data(float(s), float(v))

        wandb.log({
            "nll/reconst_table": tbl_reconst,
            "nll/reconst_plot": wandb.plot.line(
                tbl_reconst, x="snr", y="reconst", title="Reconst Curve"
            ),
        })

        tbl = wandb.Table(columns=["snr", "value"])
        for s, v in zip(snr_all, integ):
            tbl.add_data(float(s), float(v))
        wandb.log({"nll/integrand_plot": wandb.plot.line(tbl, "snr", "value", title="1/2 MSE vs SNR")})


def evaluate_sampling_methods(model, tokenizer, cfg, samplers, device, logger=None):
    """Evaluate both path-based and procedural samplers with explicit parameters."""

    # Total samples to generate (from config) vs GPU batch size (fixed small value)
    total_samples = cfg.train.eval_batch_size
    gen_batch_size = 16  # Fixed small batch for GPU memory
    
    print(f"[eval] Generating {total_samples} samples in batches of {gen_batch_size}")

    # -----------------------------
    # 1️⃣ Path-based samplers
    # -----------------------------
    path_methods = [
        {"name": "loglinear", "snr_min": 0.01, "snr_max": 5.0},  # w/ few steps, better to focus on low SNR
        {"name": "lognormal", "mu": 2.69, "sigma": 1.41, "snr_min": 0.0, "snr_max": 100.0},
        {"name": "linear", "snr_min": 0.0, "snr_max": 100.0},
    ]

    for params in path_methods:
        name = params["name"]
        print(f"\n=== Evaluating path-based sampler: {name} ===")

        # Build SNR path with explicit parameters
        snr_path = build_snr_path(params, device=device)
        snrs = snr_path.discrete_path(100)

        # Generate samples in small batches to avoid OOM
        all_samples = []
        num_batches = (total_samples + gen_batch_size - 1) // gen_batch_size
        for batch_idx in range(num_batches):
            current_batch_size = min(gen_batch_size, total_samples - len(all_samples))
            print(f"  Generating batch {batch_idx+1}/{num_batches} (size={current_batch_size})...", end="\r")
            
            samples = samplers.sample_path(
                model,
                batch_size=current_batch_size,
                seq_length=cfg.data.block_size,
                snrs=snrs,
                history=False,
            )
            all_samples.append(samples)
        
        all_samples = torch.cat(all_samples, dim=0)
        print(f"  Generated {len(all_samples)} samples total" + " " * 30)

        texts = [tokenizer.decode(s, skip_special_tokens=True) for s in all_samples]
        result = generative_perplexity(texts)
        ppl, sent_ent = result.get("ppl"), result.get("sentence_entropy")

        print(f"{name:12s} | ppl={ppl:10.3f} | sentence_entropy={sent_ent:7.3f}")

        if logger is not None:
            wandb.log({
                f"eval/{name}_ppl": ppl,
                f"eval/{name}_sentence_entropy": sent_ent,
            })

            wandb.log({
                f"eval/{name}_samples": wandb.Table(
                    columns=["method", "sample"],
                    data=[[name, t] for t in texts[:3]]
                )
            })

    # -----------------------------
    # 2️⃣ Procedural / custom samplers (ROAR variants)
    # -----------------------------
    procedural_variants = [
        {"name": "roar_high_precision", "causal": False, "low_precision_sampling": False},
        {"name": "roar_low_precision",  "causal": False, "low_precision_sampling": True},
        {"name": "ar_high_precision",   "causal": True,  "low_precision_sampling": False},
    ]

    for params in procedural_variants:
        name = params["name"]
        print(f"\n=== Evaluating procedural sampler: {name} ===")

        # Generate samples in small batches to avoid OOM
        all_samples = []
        num_batches = (total_samples + gen_batch_size - 1) // gen_batch_size
        for batch_idx in range(num_batches):
            current_batch_size = min(gen_batch_size, total_samples - len(all_samples))
            print(f"  Generating batch {batch_idx+1}/{num_batches} (size={current_batch_size})...", end="\r")
            
            samples = samplers.sample_roar(
                model,
                batch_size=current_batch_size,
                seq_length=cfg.data.block_size,
                causal=params["causal"],
                low_precision_sampling=params["low_precision_sampling"],
            )
            all_samples.append(samples)
        
        all_samples = torch.cat(all_samples, dim=0)
        print(f"  Generated {len(all_samples)} samples total" + " " * 30)

        texts = [tokenizer.decode(s, skip_special_tokens=True) for s in all_samples]
        result = generative_perplexity(texts)
        ppl, sent_ent = result.get("ppl"), result.get("sentence_entropy")

        print(f"{name:18s} | ppl={ppl:10.3f} | sentence_entropy={sent_ent:7.3f}")

        if logger is not None:
            wandb.log({
                f"eval/{name}_ppl": ppl,
                f"eval/{name}_sentence_entropy": sent_ent,
            })

            wandb.log({
                f"eval/{name}_samples": wandb.Table(
                    columns=["method", "causal", "low_precision_sampling", "sample"],
                    data=[[name, params["causal"], params["low_precision_sampling"], t] for t in texts[:3]]
                )
            })


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = pick_device(cfg)

    # --- Data
    tokenizer = get_tokenizer(cfg)
    V = len(tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    print(f"[eval] vocab_size={V}  pad_id={pad_id}")
    
    # Only load data if NLL metrics are needed
    if cfg.eval.nll_metrics:
        train_loader, val_loader = get_dataloaders(cfg, tokenizer)
        print(f"[eval] val_loader={len(val_loader)}")
        assert isinstance(val_loader, DataLoader)
    else:
        train_loader, val_loader = None, None
        print("[eval] Skipping data loading (nll_metrics disabled)")

    # --- Model and snr path
    model = build_dsl_from_cfg(cfg, vocab_size=V).to(device)
    model.eval()
    print(f"[eval] model params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")
    snr_path = build_snr_path(cfg.snrpath, device=device)

    # --- Load weights
    ckpt_path = resolve_ckpt_path(cfg)
    print(f"[eval] Loading checkpoint: {ckpt_path}")
    _ = load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None, map_location=device)
    if getattr(cfg.eval, "use_pretrained", False):
        from dsl.utils import load_weights_from_pretrained
        print(f"[eval] Overwriting with pretrained weights: {cfg.train.pretrained_model}")
        model = load_weights_from_pretrained(model, model_name=cfg.train.pretrained_model)

    # BUG FIX HACK: when loading the model, the normalized weight embedding is not correctly loaded
    # until some forward call registers it. We manually set it here, but should refactor away from buggy weight_norm
    with torch.no_grad():
        v = model.embed.weight_v
        normalized = (model.embed.weight_g / v.norm(dim=-1, keepdim=True)) * v
        model.embed.weight.copy_(normalized)

    # --- Optional wandb
    logger = None
    if getattr(cfg, "logging", None) and getattr(cfg.logging, "enabled", False):
        logger = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True, throw_on_missing=False),
        )

    if cfg.eval.interactive:
        import IPython
        IPython.embed()

    ##################### NLL METRICS #####################
    if cfg.eval.nll_metrics:
        print('[eval] NLL metrics')
        evaluate_nll(model, val_loader, snr_path, cfg, pad_id, device, logger)

    ##################### SAMPLING METRICS #####################
    if cfg.eval.sampling_metrics:
        print("[eval] Sampling metrics")
        evaluate_sampling_methods(model, tokenizer, cfg, samplers, device, logger)

    if logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()