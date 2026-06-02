"""Analyze DSL noise embedding structure for a checkpoint.

The primary question is whether the DSL codebook has developed token-semantic
neighborhoods aligned with the backbone token embedding table.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


def load_key(ckpt: Path, key: str) -> torch.Tensor:
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    shard = ckpt / index[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def summarize_embedding(name: str, embed: torch.Tensor, sample_size: int = 2000) -> dict:
    embed = embed.float()
    gen = torch.Generator().manual_seed(0)
    sample = torch.randperm(embed.shape[0], generator=gen)[:sample_size]
    unit = F.normalize(embed[sample], dim=1)
    sim = unit @ unit.T
    off_diag = sim[~torch.eye(len(sample), dtype=torch.bool)]

    centered = embed[sample] - embed[sample].mean(0, keepdim=True)
    sv = torch.linalg.svdvals(centered)
    probs = sv / sv.sum()
    effective_rank = torch.exp(-(probs * (probs + 1e-12).log()).sum()).item()

    norms = embed.norm(dim=1)
    return {
        f"{name}/norm_mean": norms.mean().item(),
        f"{name}/norm_std": norms.std().item(),
        f"{name}/avg_pair_cosine": off_diag.mean().item(),
        f"{name}/effective_rank_sample": effective_rank,
    }


def valid_token_ids(tokenizer, vocab_size: int, limit: int = 2500) -> torch.Tensor:
    ids = []
    for token_id in range(min(tokenizer.vocab_size, vocab_size)):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if text and not text.isspace() and len(text.strip()) >= 2:
            if any(ch.isalpha() for ch in text):
                ids.append(token_id)
    random.Random(123).shuffle(ids)
    return torch.tensor(ids[:limit], dtype=torch.long)


def upper_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    mask = torch.triu(torch.ones(a.shape[0], a.shape[1], dtype=torch.bool), 1)
    x = a[mask]
    y = b[mask]
    x = x - x.mean()
    y = y - y.mean()
    return ((x * y).mean() / (x.std() * y.std() + 1e-12)).item()


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    a = a.clone()
    b = b.clone()
    a.fill_diagonal_(-9)
    b.fill_diagonal_(-9)
    top_a = a.topk(k, dim=1).indices
    top_b = b.topk(k, dim=1).indices
    overlaps = []
    for lhs, rhs in zip(top_a, top_b):
        overlaps.append(len(set(lhs.tolist()) & set(rhs.tolist())) / k)
    return sum(overlaps) / len(overlaps)


def nearest_examples(tokenizer, dsl_embed: torch.Tensor, wte: torch.Tensor) -> dict:
    anchors = [
        " the",
        " China",
        " computer",
        " science",
        " money",
        " woman",
        " happy",
        " football",
        " protein",
        " COVID",
    ]
    dsl_unit = F.normalize(dsl_embed.float(), dim=1)
    wte_unit = F.normalize(wte.float(), dim=1)
    examples = {}
    for anchor in anchors:
        ids = tokenizer.encode(anchor, add_special_tokens=False)
        if not ids:
            continue
        token_id = ids[-1]
        if token_id >= dsl_unit.shape[0]:
            continue
        dsl_sim = dsl_unit @ dsl_unit[token_id]
        wte_sim = wte_unit @ wte_unit[token_id]
        dsl_sim[token_id] = -9
        wte_sim[token_id] = -9
        examples[anchor] = {
            "token_id": token_id,
            "token": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            "dsl_top8": [
                tokenizer.decode([idx], clean_up_tokenization_spaces=False)
                for idx in dsl_sim.topk(8).indices.tolist()
            ],
            "wte_top8": [
                tokenizer.decode([idx], clean_up_tokenization_spaces=False)
                for idx in wte_sim.topk(8).indices.tolist()
            ],
        }
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset-size", type=int, default=2500)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

    dsl_embed = load_key(ckpt, "converter.embed.weight").float()
    noise_embed = load_key(ckpt, "noise_embed.weight").float()
    wte = load_key(ckpt, "model.transformer.wte.weight")

    metrics = {
        "checkpoint": str(ckpt),
        "converter_vs_noise_maxdiff": (dsl_embed - noise_embed).abs().max().item(),
    }
    metrics.update(summarize_embedding("dsl", dsl_embed))

    if args.baseline_checkpoint:
        baseline = load_key(Path(args.baseline_checkpoint), "converter.embed.weight").float()
        delta = dsl_embed - baseline
        row_cos = F.cosine_similarity(dsl_embed, baseline, dim=1)
        metrics.update(
            {
                "baseline_checkpoint": args.baseline_checkpoint,
                "delta/max_abs": delta.abs().max().item(),
                "delta/mean_abs": delta.abs().mean().item(),
                "delta/rms": delta.pow(2).mean().sqrt().item(),
                "delta/relative_rms": (
                    delta.pow(2).sum().sqrt() / baseline.pow(2).sum().sqrt()
                ).item(),
                "row_cos_vs_baseline/mean": row_cos.mean().item(),
                "row_cos_vs_baseline/min": row_cos.min().item(),
                "row_cos_vs_baseline/p01": torch.quantile(row_cos, 0.01).item(),
            }
        )

    ids = valid_token_ids(tokenizer, dsl_embed.shape[0], limit=args.subset_size)
    dsl_subset = F.normalize(dsl_embed[ids], dim=1)
    wte_subset = F.normalize(wte[ids].float(), dim=1)
    dsl_cos = dsl_subset @ dsl_subset.T
    wte_cos = wte_subset @ wte_subset.T
    metrics.update(
        {
            "semantic_subset_size": int(ids.numel()),
            "pair_cos_corr_vs_wte": upper_corr(dsl_cos, wte_cos),
            "top10_overlap_vs_wte": topk_overlap(dsl_cos, wte_cos, k=10),
        }
    )

    result = {
        "metrics": metrics,
        "nearest_examples": nearest_examples(tokenizer, dsl_embed, wte),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
