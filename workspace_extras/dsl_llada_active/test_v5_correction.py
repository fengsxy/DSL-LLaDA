"""Quick corruption probe for v5 d3im-aligned LoRA vs Original.
Tests: random corruption, semantic corruption, and factual correction (1+1=3).
"""
import os, sys, torch, json
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336

TEXTS = [
    "The quick brown fox jumps over the lazy dog in the sunny park.",
    "Machine learning is a subset of artificial intelligence that focuses on data.",
    "The capital of France is Paris and it is known for the Eiffel Tower.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Great Wall of China stretches over thirteen thousand miles.",
    "Shakespeare wrote thirty seven plays during his lifetime.",
    "The speed of light is approximately three hundred thousand kilometers per second.",
    "DNA carries genetic instructions for the development of living organisms.",
    "Photosynthesis converts sunlight and carbon dioxide into glucose and oxygen.",
    "The Amazon River is the largest river in the world by volume.",
]

SEMANTIC_CASES = [
    ("The Earth orbits the Sun once every three hundred and sixty five days.", "Sun", "Moon"),
    ("Water boils at one hundred degrees Celsius.", "hundred", "thousand"),
    ("The capital of Japan is Tokyo.", "Tokyo", "Beijing"),
    ("Light travels faster than sound.", "faster", "slower"),
    ("The Pacific Ocean is the largest ocean on Earth.", "largest", "smallest"),
    ("Humans have two eyes and one nose.", "two", "three"),
    ("Ice melts at zero degrees Celsius.", "zero", "fifty"),
    ("The Nile is the longest river in Africa.", "longest", "shortest"),
    ("Diamond is the hardest natural material.", "hardest", "softest"),
    ("Oxygen is essential for human breathing.", "Oxygen", "Nitrogen"),
]

FACTUAL_CASES = [
    "One plus one equals three.",
    "The Sun revolves around the Earth.",
    "Water freezes at fifty degrees Celsius.",
    "Humans have three arms and four legs.",
    "Paris is the capital of Germany.",
]


def load_model(model_path, lora_path, device):
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if lora_path and os.path.isdir(lora_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model = model.to(device).eval()
    return model, tokenizer


def test_random_corruption(model, tokenizer, device, rates=[0.1, 0.3, 0.5]):
    print("\n=== Random Corruption ===")
    for rate in rates:
        fix_rates = []
        clean_rates = []
        for text in TEXTS:
            ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
            L = ids.shape[1]
            mask = torch.rand(1, L, device=device) < rate
            random_tok = torch.randint(100, 126000, (1, L), device=device)
            corrupted = ids.clone()
            corrupted[mask] = random_tok[mask]

            with torch.no_grad():
                pred = model(corrupted).logits.argmax(dim=-1)

            if mask.any():
                fix_rates.append((pred[mask] == ids[mask]).float().mean().item())
            if (~mask).any():
                clean_rates.append((pred[~mask] == ids[~mask]).float().mean().item())

        fix = np.mean(fix_rates) * 100
        clean = np.mean(clean_rates) * 100
        print(f"  @{rate:.0%}: fix={fix:.1f}%  clean={clean:.1f}%")


def test_semantic_corruption(model, tokenizer, device):
    print("\n=== Semantic Corruption ===")
    fixed = 0
    preserved = 0
    total = len(SEMANTIC_CASES)

    for text, gold_word, wrong_word in SEMANTIC_CASES:
        corrupted_text = text.replace(gold_word, wrong_word)
        ids_gold = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        ids_corr = tokenizer(corrupted_text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)

        if ids_gold.shape != ids_corr.shape:
            continue

        with torch.no_grad():
            pred = model(ids_corr).logits.argmax(dim=-1)

        # Find positions that differ
        diff = (ids_gold != ids_corr)
        if diff.any():
            if (pred[diff] == ids_gold[diff]).all():
                fixed += 1
            # Check clean preservation
            clean = ~diff
            if clean.any() and (pred[clean] == ids_gold[clean]).all():
                preserved += 1

    print(f"  Semantic fix: {fixed}/{total} ({fixed/total*100:.0f}%)")
    print(f"  Clean preserved: {preserved}/{total} ({preserved/total*100:.0f}%)")


def test_factual(model, tokenizer, device):
    print("\n=== Factual Correction (single pass) ===")
    for text in FACTUAL_CASES:
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        with torch.no_grad():
            pred = model(ids).logits.argmax(dim=-1)
        pred_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        changed = (pred != ids).sum().item()
        print(f"  IN:  {text}")
        print(f"  OUT: {pred_text}")
        print(f"  Changed {changed}/{ids.shape[1]} tokens")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    name = args.name or (args.lora.split("/")[-2] if args.lora else "Original")
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    model, tokenizer = load_model(args.model, args.lora, device)
    test_random_corruption(model, tokenizer, device)
    test_semantic_corruption(model, tokenizer, device)
    test_factual(model, tokenizer, device)
