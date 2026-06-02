#!/usr/bin/env python3
"""Offline calibration for DSL converter beta/SNR sharpness.

This script does not load the backbone model or train anything. It loads the
DSL noise embedding and converter logit bias, samples clean token ids, applies
the same noising formula used in training,

    z = SNR * e_x + sqrt(SNR) * eps,

and measures how sharp the converter posterior is for a grid of beta/SNR pairs.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


DEFAULT_BETAS = [0.1, 0.18, 0.3, 0.5, 1.0, 2.0]
DEFAULT_SNRS = [1, 3, 5, 10, 20, 50, 100, 150]


def parse_float_list(text):
    return [float(x) for x in text.replace(",", " ").split()]


def find_safetensors(ckpt):
    ckpt = Path(ckpt)
    if ckpt.is_file() and ckpt.suffix == ".safetensors":
        return [ckpt]
    return sorted(ckpt.glob("*.safetensors"))


def load_dsl_components(ckpt, device):
    for path in find_safetensors(ckpt):
        state = load_file(str(path), device="cpu")
        if "noise_embed.weight" not in state:
            continue
        noise = state["noise_embed.weight"].float().to(device)
        bias = state.get("converter.logit_bias")
        if bias is None:
            bias = torch.zeros(noise.shape[0] + 1, dtype=torch.float32)
            bias[-1] = math.log(noise.shape[0])
        bias = bias.float().to(device)
        beta_train = state.get("converter.beta")
        beta_train = float(beta_train.float().item()) if beta_train is not None else None
        return noise, bias, beta_train, str(path)
    raise FileNotFoundError(f"Could not find noise_embed.weight under {ckpt}")


@torch.no_grad()
def calibrate_pair(noise_embed, logit_bias, token_ids, beta, snr, chunk_size):
    V, dim = noise_embed.shape
    log_norm = math.log(V)
    totals = {
        "entropy": 0.0,
        "entropy_over_logv": 0.0,
        "n_eff": 0.0,
        "p_gold": 0.0,
        "p_mask": 0.0,
        "p_top1": 0.0,
        "top1_is_gold": 0.0,
        "logit_margin_top1_top2": 0.0,
    }
    n_seen = 0

    K = torch.cat(
        [noise_embed.float(), torch.zeros(1, dim, device=noise_embed.device)],
        dim=0,
    )
    bias = logit_bias.float()

    for start in range(0, token_ids.numel(), chunk_size):
        ids = token_ids[start : start + chunk_size]
        e = noise_embed[ids].float()
        eps = torch.randn_like(e)
        z = snr * e + math.sqrt(max(snr, 0.0)) * eps
        logits = beta * (z @ K.T) + bias
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        entropy = -(probs * log_probs).sum(dim=-1)
        top2 = torch.topk(logits, k=2, dim=-1).values
        p_top1 = probs.max(dim=-1).values
        top1_idx = probs.argmax(dim=-1)
        p_gold = probs.gather(1, ids[:, None]).squeeze(1)
        p_mask = probs[:, V]
        margin = top2[:, 0] - top2[:, 1]

        bsz = ids.numel()
        totals["entropy"] += entropy.sum().item()
        totals["entropy_over_logv"] += (entropy / log_norm).sum().item()
        totals["n_eff"] += entropy.exp().sum().item()
        totals["p_gold"] += p_gold.sum().item()
        totals["p_mask"] += p_mask.sum().item()
        totals["p_top1"] += p_top1.sum().item()
        totals["top1_is_gold"] += (top1_idx == ids).float().sum().item()
        totals["logit_margin_top1_top2"] += margin.sum().item()
        n_seen += bsz

    return {k: v / n_seen for k, v in totals.items()}


def write_csv(rows, path):
    fields = [
        "beta",
        "snr",
        "entropy",
        "entropy_over_logv",
        "n_eff",
        "p_gold",
        "p_mask",
        "p_top1",
        "top1_is_gold",
        "logit_margin_top1_top2",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def write_heatmaps(rows, betas, snrs, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("entropy_over_logv", "H / log|V|", "viridis"),
        ("p_gold", "p_gold", "magma"),
        ("p_mask", "p_mask", "magma"),
        ("p_top1", "p_top1", "magma"),
        ("top1_is_gold", "top1 is gold", "magma"),
        ("logit_margin_top1_top2", "top1-top2 logit margin", "viridis"),
    ]
    row_map = {(r["beta"], r["snr"]): r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title, cmap) in zip(axes.flat, metrics):
        mat = torch.tensor(
            [[row_map[(b, s)][key] for s in snrs] for b in betas],
            dtype=torch.float32,
        ).numpy()
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(snrs)), [str(int(s)) if s.is_integer() else str(s) for s in snrs])
        ax.set_yticks(range(len(betas)), [str(b) for b in betas])
        ax.set_xlabel("SNR")
        ax.set_ylabel("beta")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="checkpoints/unit_uniform_b01_mu169_sig09_trainable_10k_8gpu_20260511_2337/checkpoint-10000",
        help="checkpoint directory containing noise_embed.weight and converter.logit_bias",
    )
    p.add_argument("--out_dir", default="eval_results/calibration")
    p.add_argument("--tag", default=None)
    p.add_argument("--betas", default=" ".join(str(x) for x in DEFAULT_BETAS))
    p.add_argument("--snrs", default=" ".join(str(x) for x in DEFAULT_SNRS))
    p.add_argument("--num_tokens", type=int, default=512)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.checkpoint).name

    device = torch.device(args.device)
    noise_embed, logit_bias, beta_train, tensor_file = load_dsl_components(args.checkpoint, device)
    V, dim = noise_embed.shape
    token_ids = torch.randint(0, V, (args.num_tokens,), device=device)
    betas = parse_float_list(args.betas)
    snrs = parse_float_list(args.snrs)

    print(f"[calib] checkpoint={args.checkpoint}")
    print(f"[calib] tensors={tensor_file}")
    print(f"[calib] V={V} dim={dim} beta_train={beta_train}")
    print(f"[calib] tokens={args.num_tokens} betas={betas} snrs={snrs}")

    rows = []
    for beta in betas:
        for snr in snrs:
            row = {"beta": beta, "snr": snr}
            row.update(calibrate_pair(noise_embed, logit_bias, token_ids, beta, snr, args.chunk_size))
            rows.append(row)
            print(
                f"beta={beta:g} snr={snr:g} "
                f"H/logV={row['entropy_over_logv']:.3f} "
                f"Neff={row['n_eff']:.1f} "
                f"p_gold={row['p_gold']:.4f} "
                f"p_mask={row['p_mask']:.4f} "
                f"p_top1={row['p_top1']:.4f} "
                f"top1_gold={row['top1_is_gold']:.3f} "
                f"margin={row['logit_margin_top1_top2']:.3f}",
                flush=True,
            )

    out = {
        "checkpoint": args.checkpoint,
        "tensor_file": tensor_file,
        "vocab_size": V,
        "noise_dim": dim,
        "beta_train": beta_train,
        "num_tokens": args.num_tokens,
        "seed": args.seed,
        "betas": betas,
        "snrs": snrs,
        "rows": rows,
    }
    json_path = out_dir / f"beta_snr_calibration_{tag}.json"
    csv_path = out_dir / f"beta_snr_calibration_{tag}.csv"
    png_path = out_dir / f"beta_snr_calibration_{tag}.png"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    write_csv(rows, csv_path)
    write_heatmaps(rows, betas, snrs, png_path)
    print(f"[calib] wrote {json_path}")
    print(f"[calib] wrote {csv_path}")
    print(f"[calib] wrote {png_path}")


if __name__ == "__main__":
    main()
