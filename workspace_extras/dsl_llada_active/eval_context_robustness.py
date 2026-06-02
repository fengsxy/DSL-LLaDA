"""Masked-token accuracy under corrupted visible context.

This matches the context-robustness protocol used in paper.tex: mask a fixed
fraction of tokens, corrupt a fraction of the remaining visible context with
random vocabulary tokens, and report accuracy only on masked positions.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)

from eval_unified import MASK_ID, load_model, load_registry, load_tokenizer, resolve_checkpoint


def parse_csv_numbers(value, cast):
    return [cast(x) for x in value.split(",") if x.strip()]


def resolve_data_file(path):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(_root, path)


def evaluate(model, tokenizer, texts_data, device, mask_rates, corruption_rates, max_length, seed):
    results = {}
    torch.manual_seed(seed)
    np.random.seed(seed)

    for mask_rate in mask_rates:
        for corrupt_pct in corruption_rates:
            total_correct = 0
            total_masked = 0

            for item in tqdm(texts_data, desc=f"mask={int(mask_rate * 100)} corrupt={corrupt_pct}"):
                text = item["text"] if isinstance(item, dict) else item
                ids = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                ).input_ids.to(device)
                length = ids.shape[1]
                if length < 20:
                    continue

                rng = np.random.RandomState(seed + length + int(mask_rate * 100) + corrupt_pct)
                n_mask = max(1, int(length * mask_rate))
                mask_pos = np.sort(rng.choice(length, n_mask, replace=False))
                mask_set = set(mask_pos.tolist())

                clean_pos = np.array([i for i in range(length) if i not in mask_set])
                corrupted_ids = ids.clone()
                if corrupt_pct > 0 and len(clean_pos) > 0:
                    n_corrupt = int(len(clean_pos) * corrupt_pct / 100)
                    if n_corrupt > 0:
                        corr_pos = clean_pos[rng.choice(len(clean_pos), n_corrupt, replace=False)]
                        rand_tokens = torch.randint(100, 126000, (n_corrupt,), device=device)
                        corrupted_ids[0, corr_pos] = rand_tokens

                input_ids = corrupted_ids.clone()
                input_ids[0, mask_pos] = MASK_ID

                with torch.no_grad():
                    logits = model(input_ids).logits

                preds = logits[0, mask_pos].argmax(dim=-1)
                gold = ids[0, mask_pos]
                total_correct += (preds == gold).sum().item()
                total_masked += len(mask_pos)

            key = f"mask{int(mask_rate * 100)}_corrupt{corrupt_pct}"
            acc = total_correct / total_masked if total_masked else 0.0
            results[key] = round(float(acc), 4)
            print(f"{key}: acc={acc:.4f} n={total_masked}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mask_rates", default="0.3")
    parser.add_argument("--corruption_rates", default="0,10,20,30,50")
    parser.add_argument("--data_file", default="eval_data/texts_100.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    registry = load_registry()
    if args.model_key not in registry:
        raise SystemExit(f"Unknown model_key {args.model_key}; available={list(registry)}")

    device = torch.device(f"cuda:{args.gpu}")
    checkpoint = resolve_checkpoint(registry[args.model_key])
    print(f"Loading {args.model_key} from {checkpoint}", flush=True)
    model = load_model(checkpoint, device)
    tokenizer = load_tokenizer()

    data_path = resolve_data_file(args.data_file)
    with open(data_path) as f:
        texts_data = json.load(f)

    mask_rates = parse_csv_numbers(args.mask_rates, float)
    corruption_rates = parse_csv_numbers(args.corruption_rates, int)
    results = evaluate(
        model, tokenizer, texts_data, device,
        mask_rates, corruption_rates, args.max_length, args.seed,
    )

    out = args.out or os.path.join(
        _root, "eval_results", "sde_gen_formal",
        f"context_robustness_{args.model_key}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "model": args.model_key,
        "timestamp": datetime.datetime.now().isoformat(),
        "mask_rates": mask_rates,
        "corruption_rates": corruption_rates,
        "data_file": args.data_file,
        "seed": args.seed,
        "max_length": args.max_length,
        "results": {args.model_key: results},
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
