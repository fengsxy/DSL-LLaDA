"""
Test Soft Remasking on GSM8K: compare standard vs d3im vs soft_remask.

Usage:
    CUDA_VISIBLE_DEVICES=6 python dsl_llada/test_soft_remasking.py [--checkpoint PATH] [--n 20] [--steps 64]
"""

import os
import sys
import json
import re
import time
import argparse

import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from generate import generate

from soft_remasking import generate_soft_remask


def extract_number(text):
    """Extract final number from GSM8K-style output."""
    # Try #### format
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # "the answer is X"
    match = re.search(r'(?:the answer is|answer is|= )\s*\$?([+-]?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    # Last number in text
    numbers = re.findall(r'[+-]?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def load_gsm8k(path, n=20):
    with open(path) as f:
        data = json.load(f)
    return data[:n]


def format_prompt(question, tokenizer):
    prompt_text = (
        f"Solve the following math problem step by step. "
        f"Show your work and put your final answer after ####.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return formatted


@torch.no_grad()
def run_method(model, tokenizer, problems, method, steps, gen_length, device, **kwargs):
    """Run a generation method on all problems. Returns list of (pred, gold, correct)."""
    results = []
    for idx, prob in enumerate(problems):
        question = prob['question']
        gold = str(prob['gold_answer']).strip()

        formatted = format_prompt(question, tokenizer)
        encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = torch.ones_like(input_ids)

        if method == 'soft_remask':
            out = generate_soft_remask(
                model, input_ids, attention_mask,
                steps=steps, gen_length=gen_length,
                block_length=gen_length,
                temperature=0., cfg_scale=0.,
                **kwargs
            )
        else:
            remasking = method if method in ('d3im',) else 'low_confidence'
            out = generate(
                model, input_ids, attention_mask,
                steps=steps, gen_length=gen_length,
                block_length=gen_length,
                temperature=0., cfg_scale=0.,
                remasking=remasking,
            )

        generated = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_number(generated)
        correct = (pred is not None and pred == gold)
        results.append({
            'idx': idx,
            'pred': pred,
            'gold': gold,
            'correct': correct,
            'text': generated[:200],
        })
        status = "OK" if correct else "X"
        print(f"  [{idx+1}/{len(problems)}] {status} pred={pred} gold={gold}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (default: LLaDA-8B-Instruct)')
    parser.add_argument('--n', type=int, default=20, help='Number of GSM8K problems')
    parser.add_argument('--steps', type=int, default=64, help='Generation steps')
    parser.add_argument('--gen_length', type=int, default=512, help='Generation length')
    parser.add_argument('--data', type=str,
                        default='/home/ubuntu/efs/RMDM/eval_data/gsm8k_100.json')
    args = parser.parse_args()

    device = 'cuda'

    # Load model
    model_name = args.checkpoint or 'GSAI-ML/LLaDA-8B-Instruct'
    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    tokenizer.padding_side = 'left'
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"Model loaded on {device}", flush=True)

    # Load data
    problems = load_gsm8k(args.data, args.n)
    print(f"Loaded {len(problems)} GSM8K problems", flush=True)
    print(f"Steps={args.steps}, gen_length={args.gen_length}", flush=True)
    print("=" * 70, flush=True)

    all_results = {}

    # --- Method 1: Standard (low_confidence) ---
    print("\n[1/5] Standard remasking (low_confidence)")
    t0 = time.time()
    res = run_method(model, tokenizer, problems, 'low_confidence', args.steps, args.gen_length, device)
    acc = sum(r['correct'] for r in res) / len(res) * 100
    elapsed = time.time() - t0
    all_results['standard'] = {'acc': acc, 'results': res, 'time': elapsed}
    print(f"  => Accuracy: {acc:.1f}% ({elapsed:.1f}s)", flush=True)

    # --- Method 2: d3im ---
    print("\n[2/5] d3im remasking")
    t0 = time.time()
    res = run_method(model, tokenizer, problems, 'd3im', args.steps, args.gen_length, device)
    acc = sum(r['correct'] for r in res) / len(res) * 100
    elapsed = time.time() - t0
    all_results['d3im'] = {'acc': acc, 'results': res, 'time': elapsed}
    print(f"  => Accuracy: {acc:.1f}% ({elapsed:.1f}s)", flush=True)

    # --- Method 3: Soft Remask (exponential) ---
    print("\n[3/5] Soft Remasking (exponential decay)")
    t0 = time.time()
    res = run_method(model, tokenizer, problems, 'soft_remask', args.steps, args.gen_length, device,
                     remask_schedule='exponential', remask_initial=0.3, remask_decay=0.92,
                     protect_steps=4, freeze_steps=8)
    acc = sum(r['correct'] for r in res) / len(res) * 100
    elapsed = time.time() - t0
    all_results['soft_exp'] = {'acc': acc, 'results': res, 'time': elapsed}
    print(f"  => Accuracy: {acc:.1f}% ({elapsed:.1f}s)", flush=True)

    # --- Method 4: Soft Remask (linear) ---
    print("\n[4/5] Soft Remasking (linear decay)")
    t0 = time.time()
    res = run_method(model, tokenizer, problems, 'soft_remask', args.steps, args.gen_length, device,
                     remask_schedule='linear', remask_initial=0.25, remask_decay=0.9,
                     protect_steps=4, freeze_steps=8)
    acc = sum(r['correct'] for r in res) / len(res) * 100
    elapsed = time.time() - t0
    all_results['soft_linear'] = {'acc': acc, 'results': res, 'time': elapsed}
    print(f"  => Accuracy: {acc:.1f}% ({elapsed:.1f}s)", flush=True)

    # --- Method 5: Soft Remask (cosine) ---
    print("\n[5/5] Soft Remasking (cosine decay)")
    t0 = time.time()
    res = run_method(model, tokenizer, problems, 'soft_remask', args.steps, args.gen_length, device,
                     remask_schedule='cosine', remask_initial=0.3, remask_decay=0.9,
                     protect_steps=4, freeze_steps=8)
    acc = sum(r['correct'] for r in res) / len(res) * 100
    elapsed = time.time() - t0
    all_results['soft_cosine'] = {'acc': acc, 'results': res, 'time': elapsed}
    print(f"  => Accuracy: {acc:.1f}% ({elapsed:.1f}s)", flush=True)

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"SUMMARY — {len(problems)} GSM8K problems, {args.steps} steps, gen_length={args.gen_length}")
    print(f"Model: {model_name}")
    print("-" * 70)
    print(f"{'Method':<25} {'Accuracy':>10} {'Time':>10}")
    print("-" * 70)
    for name, data in all_results.items():
        print(f"{name:<25} {data['acc']:>9.1f}% {data['time']:>9.1f}s")
    print("=" * 70)

    # Per-problem comparison
    print("\nPer-problem breakdown:")
    print(f"{'#':<4} {'Gold':<8}", end='')
    for name in all_results:
        print(f" {name:<12}", end='')
    print()
    for i in range(len(problems)):
        gold = problems[i]['gold_answer']
        print(f"{i:<4} {str(gold):<8}", end='')
        for name, data in all_results.items():
            r = data['results'][i]
            marker = "OK" if r['correct'] else f"X({r['pred']})"
            print(f" {marker:<12}", end='')
        print()


if __name__ == '__main__':
    main()
