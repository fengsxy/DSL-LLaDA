"""Analyze GSM8K d3im generation trajectories: Original vs v5.

For each problem, record per-step:
- Number of committed tokens
- Number of demotions (token→MASK)
- Number of corrections (token→different token)
- Whether the answer is correct at each step
- When the correct answer first appears / disappears

Produces summary statistics and per-problem analysis.
"""
import os, sys, torch, json, re
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from transformers import AutoTokenizer, AutoModel
from generate import generate

MASK_ID = 126336
EOS_ID = 126081
EOT_ID = 126348


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
    return ""


def trace_d3im_with_answer(model, tokenizer, prompt_text, gold, device,
                            steps=64, gen_length=512):
    """Run d3im step-by-step, check answer at each step."""
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

    schedule = [int((i + 1) * L / steps) for i in range(steps)]
    schedule[-1] = L

    step_data = []
    first_correct = -1
    last_correct = -1

    with torch.no_grad():
        for step in range(steps):
            prev = x[0, P:P+L].clone()

            logits = model(x).logits[0, P:P+L]
            # Suppress EOS/EOT
            logits[:, EOS_ID] = -float('inf')
            logits[:, EOT_ID] = -float('inf')

            probs = F.softmax(logits.float(), dim=-1)
            x0 = logits.argmax(dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            n_keep = schedule[step]
            _, topk = conf.topk(min(n_keep, L))
            new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
            new_gen[topk] = x0[topk]
            x[0, P:P+L] = new_gen

            curr = x[0, P:P+L]

            # Count events
            prev_committed = (prev != MASK_ID)
            curr_committed = (curr != MASK_ID)

            n_new = ((~prev_committed) & curr_committed).sum().item()
            n_demoted = (prev_committed & (~curr_committed)).sum().item()
            n_changed = (prev_committed & curr_committed & (prev != curr)).sum().item()
            n_committed = curr_committed.sum().item()
            avg_conf = conf.mean().item()

            # Check answer
            text = tokenizer.decode(curr, skip_special_tokens=True)
            pred = extract_number(text)
            is_correct = (pred == str(gold).strip())

            if is_correct and first_correct < 0:
                first_correct = step
            if is_correct:
                last_correct = step

            step_data.append({
                "step": step,
                "n_committed": n_committed,
                "n_new": n_new,
                "n_demoted": n_demoted,
                "n_changed": n_changed,
                "avg_conf": avg_conf,
                "is_correct": is_correct,
                "pred": pred,
            })

    final_text = tokenizer.decode(x[0, P:], skip_special_tokens=True)
    final_pred = extract_number(final_text)
    final_correct = (final_pred == str(gold).strip())

    total_dem = sum(s["n_demoted"] for s in step_data)
    total_chg = sum(s["n_changed"] for s in step_data)

    return {
        "step_data": step_data,
        "final_pred": final_pred,
        "final_correct": final_correct,
        "first_correct": first_correct,
        "last_correct": last_correct,
        "total_demotions": total_dem,
        "total_corrections": total_chg,
    }


def run_analysis(model, tokenizer, data, device, model_name, steps=64):
    """Run trace on all problems, collect statistics."""
    results = []
    for item in tqdm(data, desc=model_name):
        prompt = (f"Solve step by step. Put final answer after ####.\n"
                  f"Question: {item['question']}\nAnswer:")
        r = trace_d3im_with_answer(
            model, tokenizer, prompt, item["gold_answer"], device, steps=steps
        )
        r["id"] = item["id"]
        r["gold"] = item["gold_answer"]
        results.append(r)
    return results


def print_summary(results_orig, results_v5):
    """Print comparative summary."""
    print("\n" + "=" * 80)
    print("GSM8K D3IM TRAJECTORY ANALYSIS")
    print("=" * 80)

    # Overall accuracy
    orig_correct = sum(1 for r in results_orig if r["final_correct"])
    v5_correct = sum(1 for r in results_v5 if r["final_correct"])
    n = len(results_orig)

    print(f"\n=== Overall Accuracy ===")
    print(f"  Original: {orig_correct}/{n} ({orig_correct/n*100:.1f}%)")
    print(f"  v5_r64:   {v5_correct}/{n} ({v5_correct/n*100:.1f}%)")

    # Demotions and corrections
    orig_dem = [r["total_demotions"] for r in results_orig]
    v5_dem = [r["total_demotions"] for r in results_v5]
    orig_chg = [r["total_corrections"] for r in results_orig]
    v5_chg = [r["total_corrections"] for r in results_v5]

    print(f"\n=== Demotions (token→MASK) ===")
    print(f"  Original: mean={np.mean(orig_dem):.1f}, median={np.median(orig_dem):.0f}")
    print(f"  v5_r64:   mean={np.mean(v5_dem):.1f}, median={np.median(v5_dem):.0f}")

    print(f"\n=== Corrections (token→different) ===")
    print(f"  Original: mean={np.mean(orig_chg):.1f}, median={np.median(orig_chg):.0f}")
    print(f"  v5_r64:   mean={np.mean(v5_chg):.1f}, median={np.median(v5_chg):.0f}")

    # Answer stability
    orig_ever_correct = sum(1 for r in results_orig if r["first_correct"] >= 0)
    v5_ever_correct = sum(1 for r in results_v5 if r["first_correct"] >= 0)
    orig_lost = sum(1 for r in results_orig if r["first_correct"] >= 0 and not r["final_correct"])
    v5_lost = sum(1 for r in results_v5 if r["first_correct"] >= 0 and not r["final_correct"])

    print(f"\n=== Answer Evolution ===")
    print(f"  Original: ever-correct={orig_ever_correct}/{n}, lost={orig_lost} ({orig_lost/max(1,orig_ever_correct)*100:.0f}%)")
    print(f"  v5_r64:   ever-correct={v5_ever_correct}/{n}, lost={v5_lost} ({v5_lost/max(1,v5_ever_correct)*100:.0f}%)")

    # First correct step distribution
    orig_first = [r["first_correct"] for r in results_orig if r["first_correct"] >= 0]
    v5_first = [r["first_correct"] for r in results_v5 if r["first_correct"] >= 0]
    if orig_first:
        print(f"  Original first-correct step: mean={np.mean(orig_first):.1f}, median={np.median(orig_first):.0f}")
    if v5_first:
        print(f"  v5_r64 first-correct step:   mean={np.mean(v5_first):.1f}, median={np.median(v5_first):.0f}")

    # Per-problem comparison: v5 correct but Original wrong
    print(f"\n=== v5 correct, Original wrong ===")
    for ro, rv in zip(results_orig, results_v5):
        if rv["final_correct"] and not ro["final_correct"]:
            print(f"  Q{ro['id']} gold={ro['gold']}: "
                  f"orig_pred={ro['final_pred']} dem={ro['total_demotions']} | "
                  f"v5_pred={rv['final_pred']} dem={rv['total_demotions']}")

    print(f"\n=== Original correct, v5 wrong ===")
    for ro, rv in zip(results_orig, results_v5):
        if ro["final_correct"] and not rv["final_correct"]:
            print(f"  Q{ro['id']} gold={ro['gold']}: "
                  f"orig_pred={ro['final_pred']} dem={ro['total_demotions']} | "
                  f"v5_pred={rv['final_pred']} dem={rv['total_demotions']}")

    # Demotion vs correctness correlation
    print(f"\n=== Demotion vs Correctness ===")
    for name, results in [("Original", results_orig), ("v5_r64", results_v5)]:
        correct_dem = [r["total_demotions"] for r in results if r["final_correct"]]
        wrong_dem = [r["total_demotions"] for r in results if not r["final_correct"]]
        print(f"  {name}: correct_avg_dem={np.mean(correct_dem):.1f}, wrong_avg_dem={np.mean(wrong_dem):.1f}" if correct_dem and wrong_dem else f"  {name}: insufficient data")

    # Step-wise accuracy curve
    print(f"\n=== Step-wise Accuracy ===")
    print(f"{'Step':>4} | {'Original':>8} | {'v5_r64':>8}")
    print("-" * 30)
    for step in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
        orig_acc = sum(1 for r in results_orig if r["step_data"][step]["is_correct"]) / n * 100
        v5_acc = sum(1 for r in results_v5 if r["step_data"][step]["is_correct"]) / n * 100
        print(f"{step:>4} | {orig_acc:>7.1f}% | {v5_acc:>7.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", default="eval_results/gsm8k_trace_analysis.json")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    data = json.load(open("eval_data/gsm8k_100.json"))[:args.n]

    # Original
    print("Loading Original...")
    model, tokenizer = load_model("GSAI-ML/LLaDA-8B-Instruct", "", device)
    results_orig = run_analysis(model, tokenizer, data, device, "Original", args.steps)
    del model; torch.cuda.empty_cache()

    # v5 r=64
    print("\nLoading v5_r64...")
    model, tokenizer = load_model(
        "GSAI-ML/LLaDA-8B-Instruct",
        "checkpoints/d3im_lora_v5_r64/checkpoint-500", device
    )
    results_v5 = run_analysis(model, tokenizer, data, device, "v5_r64", args.steps)
    del model; torch.cuda.empty_cache()

    # Print summary
    print_summary(results_orig, results_v5)

    # Save raw data
    output = {
        "original": [{k: v for k, v in r.items() if k != "step_data"} for r in results_orig],
        "v5_r64": [{k: v for k, v in r.items() if k != "step_data"} for r in results_v5],
        "step_curves": {
            "original": [[s["is_correct"] for s in r["step_data"]] for r in results_orig],
            "v5_r64": [[s["is_correct"] for s in r["step_data"]] for r in results_v5],
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRaw data saved to {args.output}")
