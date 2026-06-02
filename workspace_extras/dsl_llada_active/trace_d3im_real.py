"""Trace REAL d3im generation: every step resets non-top-K to MASK.

Records at each step:
- Which tokens were kept (top-K confident)
- Which tokens were CHANGED from previous step (corrections!)
- Which corrections were good (old wrong → new right) vs bad (old right → new wrong)
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


def trace_real_d3im(model, tokenizer, prompt_text, device, steps=64, gen_length=256):
    """Run REAL d3im generation with full tracing."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    P = input_ids.shape[1]
    L = gen_length

    # Initialize: prompt + all MASK
    x = torch.cat([
        input_ids,
        torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    ], dim=1)

    # How many tokens to keep at each step (cumulative schedule)
    tokens_per_step = L / steps
    cumulative = torch.tensor([int((i + 1) * tokens_per_step) for i in range(steps)])
    cumulative[-1] = L  # ensure all unmasked at last step

    traces = []

    with torch.no_grad():
        for step in range(steps):
            prev_tokens = x[0, P:P+L].clone()  # tokens before this step

            # Forward pass
            logits = model(x).logits
            gen_logits = logits[0, P:P+L]

            # Suppress EOS/EOT tokens
            EOS_ID, EOT_ID = 126081, 126348
            gen_logits[:, EOS_ID] = -float('inf')
            gen_logits[:, EOT_ID] = -float('inf')

            probs = F.softmax(gen_logits.float(), dim=-1)
            x0 = gen_logits.argmax(dim=-1)  # predicted tokens for all positions
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # confidence of predictions

            # d3im: keep top-K confident, remask everything else
            n_keep = int(cumulative[step].item())
            _, topk_idx = conf.topk(min(n_keep, L))

            # Build new sequence: MASK everywhere, then fill top-K with predictions
            new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
            new_gen[topk_idx] = x0[topk_idx]

            x[0, P:P+L] = new_gen

            # === Analyze what happened ===
            current = x[0, P:P+L].clone()

            # Positions that are NOT mask in both steps
            prev_committed = (prev_tokens != MASK_ID)
            curr_committed = (current != MASK_ID)
            both_committed = prev_committed & curr_committed

            # Token changes on committed positions
            if both_committed.any():
                changed = both_committed & (current != prev_tokens)
                n_changed = changed.sum().item()
                n_both = both_committed.sum().item()
            else:
                n_changed = 0
                n_both = 0

            # Newly committed (was MASK, now has token)
            newly_committed = (~prev_committed) & curr_committed
            n_new = newly_committed.sum().item()

            # Lost (was committed, now MASK again)
            lost = prev_committed & (~curr_committed)
            n_lost = lost.sum().item()

            # Confidence stats
            committed_conf = conf[curr_committed].mean().item() if curr_committed.any() else 0
            mask_conf = conf[~curr_committed].mean().item() if (~curr_committed).any() else 0

            n_mask = (current == MASK_ID).sum().item()

            traces.append({
                "step": step,
                "n_keep": n_keep,
                "n_mask": n_mask,
                "n_changed": n_changed,  # committed→different token (CORRECTION!)
                "n_both_committed": n_both,
                "n_new": n_new,  # MASK→token
                "n_lost": n_lost,  # token→MASK (demoted!)
                "committed_conf": committed_conf,
                "mask_conf": mask_conf,
            })

    final_tokens = x[0, P:P+L]
    final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
    return traces, final_text


def print_trace(traces, name, final_text, gold_answer=""):
    print(f"\n{'='*70}")
    print(f"Model: {name}")
    print(f"{'='*70}")
    print(f"{'Step':>4} {'Keep':>4} {'Mask':>4} {'Changed':>7} {'New':>4} {'Lost':>4} {'CommConf':>8} {'MaskConf':>8}")
    print("-" * 70)

    total_changed = 0
    total_lost = 0
    for t in traces:
        total_changed += t["n_changed"]
        total_lost += t["n_lost"]
        if t["step"] % 4 == 0 or t["step"] == len(traces) - 1:
            print(f"{t['step']:>4} {t['n_keep']:>4} {t['n_mask']:>4} "
                  f"{t['n_changed']:>7} {t['n_new']:>4} {t['n_lost']:>4} "
                  f"{t['committed_conf']:>8.3f} {t['mask_conf']:>8.3f}")

    print(f"\nTotal corrections (changed committed): {total_changed}")
    print(f"Total demotions (committed→MASK): {total_lost}")
    print(f"Final: {final_text[:300]}")
    if gold_answer:
        pred = extract_number(final_text)
        print(f"Predicted: {pred}, Gold: {gold_answer}, {'✅' if pred == gold_answer else '❌'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", default="")
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    name = args.name or "Model"
    model, tokenizer = load_model(args.model, args.lora, device)

    problems = [
        ("Solve step by step. Put final answer after ####.\nQuestion: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?\nAnswer:", "18"),
        ("Solve step by step. Put final answer after ####.\nQuestion: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nAnswer:", "3"),
        ("Solve step by step. Put final answer after ####.\nQuestion: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\nAnswer:", "70000"),
    ]

    for prompt, gold in problems:
        traces, final_text = trace_real_d3im(
            model, tokenizer, prompt, device, steps=args.steps
        )
        print_trace(traces, name, final_text, gold)
