"""Adaptive d3im sampling strategies.

Drop-in replacement for standard d3im remasking in LLaDA's generate().
Tests 4 strategies for deciding how many tokens to keep at each step.
"""
import os, sys, torch, json, re
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336


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


def extract_number(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else ""


def generate_adaptive(model, tokenizer, prompt_text, device,
                       steps=64, gen_length=256, strategy="standard",
                       threshold=0.5, percentile=10):
    """Generate with various adaptive d3im strategies.

    Strategies:
        standard: fixed linear schedule (baseline d3im)
        threshold: keep all positions with confidence > threshold, + min guarantee
        percentile: keep all except bottom p-percentile of committed confidence
        gap: find largest confidence gap, keep everything above it
    """
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    P = input_ids.shape[1]
    L = gen_length

    x = torch.cat([
        input_ids,
        torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    ], dim=1)

    # Standard schedule (used as min guarantee for adaptive methods)
    linear_schedule = [int((i + 1) * L / steps) for i in range(steps)]
    linear_schedule[-1] = L

    with torch.no_grad():
        for step in range(steps):
            logits = model(x).logits[0, P:P+L]
            probs = F.softmax(logits.float(), dim=-1)
            x0 = logits.argmax(dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            if strategy == "standard":
                # Standard d3im: fixed linear schedule
                n_keep = linear_schedule[step]
                _, topk_idx = conf.topk(min(n_keep, L))
                new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
                new_gen[topk_idx] = x0[topk_idx]

            elif strategy == "threshold":
                # Keep all positions above threshold, with min guarantee
                min_keep = max(4, linear_schedule[step] // 2)  # at least half of linear
                above_threshold = (conf > threshold)
                n_above = above_threshold.sum().item()
                n_keep = max(n_above, min_keep)
                n_keep = min(n_keep, L)
                _, topk_idx = conf.topk(n_keep)
                new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
                new_gen[topk_idx] = x0[topk_idx]

            elif strategy == "percentile":
                # Keep all except bottom p-percentile
                # First, establish minimum from linear schedule
                min_keep = max(4, linear_schedule[step] // 2)
                # Compute percentile threshold from current confidence distribution
                conf_sorted = conf.sort().values
                n_total = L
                p_idx = max(0, int(n_total * percentile / 100) - 1)
                p_threshold = conf_sorted[p_idx].item()
                above = (conf > p_threshold)
                n_keep = max(above.sum().item(), min_keep)
                n_keep = min(n_keep, L)
                _, topk_idx = conf.topk(n_keep)
                new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
                new_gen[topk_idx] = x0[topk_idx]

            elif strategy == "gap":
                # Find largest gap in sorted confidence, split there
                min_keep = max(4, linear_schedule[step] // 2)
                conf_sorted, conf_order = conf.sort()
                gaps = conf_sorted[1:] - conf_sorted[:-1]
                # Only consider gaps in the middle 20-80% range
                start = L // 5
                end = L * 4 // 5
                if end > start:
                    gap_region = gaps[start:end]
                    best_gap_idx = start + gap_region.argmax().item()
                else:
                    best_gap_idx = L // 2
                # Keep everything above the gap
                n_keep = L - best_gap_idx - 1
                n_keep = max(n_keep, min_keep)
                n_keep = min(n_keep, L)
                _, topk_idx = conf.topk(n_keep)
                new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
                new_gen[topk_idx] = x0[topk_idx]

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            x[0, P:P+L] = new_gen

    final_text = tokenizer.decode(x[0, P:], skip_special_tokens=True)
    return final_text


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--strategies", default="standard,threshold,percentile,gap")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--percentile", type=float, default=10)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model, tokenizer = load_model(args.model, args.lora, device)

    data = json.load(open("eval_data/gsm8k_100.json"))[:args.n]
    strategies = args.strategies.split(",")

    for strategy in strategies:
        correct = 0
        for item in tqdm(data, desc=f"{strategy} s={args.steps}"):
            prompt = (f"Solve step by step. Put final answer after ####.\n"
                      f"Question: {item['question']}\nAnswer:")
            text = generate_adaptive(
                model, tokenizer, prompt, device,
                steps=args.steps, strategy=strategy,
                threshold=args.threshold, percentile=args.percentile,
            )
            pred = extract_number(text)
            gold = str(item['gold_answer'])
            if pred.strip() == gold.strip():
                correct += 1

        print(f"[{strategy:>12}] s={args.steps}: {correct}/{len(data)} = {correct/len(data)*100:.1f}%")

    print("\nDone!")
