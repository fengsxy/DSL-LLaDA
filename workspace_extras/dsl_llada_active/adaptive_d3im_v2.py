"""Adaptive d3im v2: monotonic committed count, accelerate on easy problems.

Key constraint: committed set only grows, never shrinks.
- Easy problem: confidence high → commit many per step → finish early
- Hard problem: confidence low → commit minimum per step → same as standard d3im

This means adaptive can ONLY be faster, never slower than standard d3im.
"""
import os, sys, torch, json, re
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

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


def generate_monotonic_adaptive(model, tokenizer, prompt_text, device,
                                 steps=64, gen_length=256,
                                 threshold=0.5, mode="adaptive"):
    """Monotonic adaptive d3im: committed set only grows.

    Modes:
        standard: fixed K per step (baseline)
        adaptive: commit all confident masked positions, with min guarantee
        adaptive_decay: threshold decays over steps (0.8 → 0.2)
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

    # Track committed positions (monotonic: once committed, stays committed)
    committed = torch.zeros(L, dtype=torch.bool, device=device)
    committed_tokens = torch.zeros(L, dtype=torch.long, device=device)

    min_per_step = max(1, L // steps)  # minimum new commits per step
    total_steps_used = steps

    with torch.no_grad():
        for step in range(steps):
            # Check if all committed
            n_remaining = (~committed).sum().item()
            if n_remaining == 0:
                total_steps_used = step
                break

            # Forward pass
            logits = model(x).logits[0, P:P+L]

            # Suppress EOS/EOT tokens
            EOS_ID, EOT_ID = 126081, 126348
            logits[:, EOS_ID] = -float('inf')
            logits[:, EOT_ID] = -float('inf')

            probs = F.softmax(logits.float(), dim=-1)
            x0 = logits.argmax(dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            # Only consider MASKED (uncommitted) positions
            masked_positions = ~committed
            masked_conf = conf.clone()
            masked_conf[committed] = -1  # already committed, don't re-select

            if mode == "standard":
                # Fixed: commit exactly min_per_step new positions
                n_new = min(min_per_step, n_remaining)

            elif mode == "adaptive":
                # Commit all masked positions above threshold, with min guarantee
                above = (masked_conf > threshold) & masked_positions
                n_above = above.sum().item()
                n_new = max(n_above, min(min_per_step, n_remaining))

            elif mode == "adaptive_decay":
                # Threshold decays: high early (conservative), low late (aggressive)
                progress = step / steps
                t = 0.8 - 0.6 * progress  # 0.8 → 0.2
                above = (masked_conf > t) & masked_positions
                n_above = above.sum().item()
                n_new = max(n_above, min(min_per_step, n_remaining))

            elif mode == "adaptive_median":
                # Threshold = median of masked confidence (always commit top half)
                if masked_positions.any():
                    mc = masked_conf[masked_positions]
                    t = mc.median().item()
                    above = (masked_conf > t) & masked_positions
                    n_above = above.sum().item()
                    n_new = max(n_above, min(min_per_step, n_remaining))
                else:
                    n_new = 0

            else:
                raise ValueError(f"Unknown mode: {mode}")

            n_new = min(n_new, n_remaining)
            if n_new <= 0:
                continue

            # Select top-n_new confident from MASKED positions only
            _, topk_idx = masked_conf.topk(n_new)

            # Commit these positions (monotonic: add to committed set)
            committed[topk_idx] = True
            committed_tokens[topk_idx] = x0[topk_idx]

            # Update x: committed positions get their tokens, rest stay MASK
            gen_part = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
            gen_part[committed] = committed_tokens[committed]
            x[0, P:P+L] = gen_part

    final_text = tokenizer.decode(x[0, P:], skip_special_tokens=True)
    return final_text, total_steps_used


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--modes", default="standard,adaptive,adaptive_decay,adaptive_median")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model, tokenizer = load_model(args.model, args.lora, device)

    data = json.load(open("eval_data/gsm8k_100.json"))[:args.n]
    modes = args.modes.split(",")

    for mode in modes:
        correct = 0
        total_steps_used = []
        for item in tqdm(data, desc=f"{mode} s={args.steps}"):
            prompt = (f"Solve step by step. Put final answer after ####.\n"
                      f"Question: {item['question']}\nAnswer:")
            text, steps_used = generate_monotonic_adaptive(
                model, tokenizer, prompt, device,
                steps=args.steps, mode=mode,
                threshold=args.threshold,
            )
            total_steps_used.append(steps_used)
            pred = extract_number(text)
            gold = str(item['gold_answer'])
            if pred.strip() == gold.strip():
                correct += 1

        avg_steps = sum(total_steps_used) / len(total_steps_used)
        print(f"[{mode:>16}] s={args.steps}: {correct}/{len(data)} = {correct/len(data)*100:.1f}%  "
              f"avg_steps={avg_steps:.1f}")

    print("\nDone!")
