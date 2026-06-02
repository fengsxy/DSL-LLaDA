"""Trace d3im generation step-by-step to see if error correction actually happens.

For each step, record:
- Which tokens changed
- Were the old tokens wrong? (compare to final answer)
- Were the new tokens correct? (compare to final answer)
- This reveals: does the model fix its own mistakes during generation?
"""
import os, sys, torch, json
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from transformers import AutoTokenizer, AutoModel
from generate import generate

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


def trace_d3im_generation(model, tokenizer, prompt_text, device, steps=64, gen_length=256):
    """Run d3im generation and record token changes at each step."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    P = input_ids.shape[1]

    # Initialize: prompt + all MASK
    x = torch.cat([
        input_ids,
        torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    ], dim=1)

    traces = []  # list of per-step records

    with torch.no_grad():
        for step in range(steps):
            prev_tokens = x[0, P:].clone()

            logits = model(x).logits
            gen_logits = logits[0, P:]
            probs = F.softmax(gen_logits.float(), dim=-1)
            pred = gen_logits.argmax(dim=-1)
            conf = probs.max(dim=-1).values

            # d3im remasking: unmask top-K confident, remask rest
            mask_pos = (x[0, P:] == MASK_ID)
            n_mask = mask_pos.sum().item()
            if n_mask == 0:
                break

            n_unmask = max(1, n_mask // (steps - step))

            # For all positions: get prediction
            x0 = pred.clone()
            x0_conf = conf.clone()

            # Only unmask from currently masked positions
            masked_conf = x0_conf.clone()
            masked_conf[~mask_pos] = -1  # don't re-select already unmasked

            _, topk = masked_conf.topk(min(n_unmask, n_mask))

            # Remask all non-topk masked positions
            new_x = x[0, P:].clone()
            new_x[mask_pos] = MASK_ID  # reset all masked
            new_x[topk] = x0[topk]  # unmask top-K

            # Also: d3im can change already-committed tokens
            # Check if model wants to change any committed token
            committed = ~mask_pos
            if committed.any():
                # Model's prediction for committed positions
                committed_pred = x0[committed]
                committed_old = prev_tokens[committed]
                committed_changed = (committed_pred != committed_old)
                # Don't actually change them in standard d3im
                # But record if the model WANTS to change them
                n_want_change = committed_changed.sum().item()
            else:
                n_want_change = 0

            x[0, P:] = new_x
            current_tokens = x[0, P:].clone()

            # Record changes
            changed = (current_tokens != prev_tokens)
            n_changed = changed.sum().item()
            n_committed = committed.sum().item()
            n_still_mask = (current_tokens == MASK_ID).sum().item()

            traces.append({
                "step": step,
                "n_mask_before": n_mask,
                "n_unmask": n_unmask,
                "n_changed": n_changed,
                "n_committed": n_committed,
                "n_still_mask": n_still_mask,
                "n_want_change_committed": n_want_change,
                "avg_conf": conf.mean().item(),
            })

    final_text = tokenizer.decode(x[0, P:], skip_special_tokens=True)
    return traces, final_text


def analyze_traces(traces, name):
    """Print summary of generation traces."""
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    # Key steps
    for t in traces:
        if t["step"] % 8 == 0 or t["step"] == len(traces) - 1:
            print(f"  Step {t['step']:>3}: mask={t['n_mask_before']:>3} → unmask {t['n_unmask']:>3} "
                  f"| committed={t['n_committed']:>3} want_change={t['n_want_change_committed']:>3} "
                  f"| conf={t['avg_conf']:.3f}")

    # Summary: how many times did model want to change committed tokens?
    total_want_change = sum(t["n_want_change_committed"] for t in traces)
    total_committed_steps = sum(t["n_committed"] for t in traces)
    print(f"\n  Total want-to-change committed: {total_want_change} / {total_committed_steps} "
          f"({total_want_change/max(1,total_committed_steps)*100:.1f}%)")


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
        "Solve step by step. Put final answer after ####.\nQuestion: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?\nAnswer:",
        "Solve step by step. Put final answer after ####.\nQuestion: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nAnswer:",
        "Solve step by step. Put final answer after ####.\nQuestion: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\nAnswer:",
    ]

    for prompt in problems:
        traces, final_text = trace_d3im_generation(
            model, tokenizer, prompt, device, steps=args.steps
        )
        analyze_traces(traces, name)
        print(f"\n  Final: {final_text[:200]}")
        print()
