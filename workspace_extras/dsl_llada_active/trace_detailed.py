"""Detailed d3im trace: compare Original vs v5 token-by-token on same problems.

For each step, show:
- Current text (decoded)
- Which positions changed from previous step
- Which were demotions (token→MASK) vs corrections (token→different token) vs new (MASK→token)
- Focus on the ANSWER region to see how reasoning evolves
"""
import os, sys, torch, json, re
import torch.nn.functional as F
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


def trace_d3im_detailed(model, tokenizer, prompt_text, device, steps=64, gen_length=256):
    """Full d3im trace with per-step text snapshots."""
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

    tokens_per_step = L / steps
    cumulative = [int((i + 1) * tokens_per_step) for i in range(steps)]
    cumulative[-1] = L

    snapshots = []

    with torch.no_grad():
        for step in range(steps):
            prev = x[0, P:P+L].clone()

            logits = model(x).logits[0, P:P+L]
            probs = F.softmax(logits.float(), dim=-1)
            x0 = logits.argmax(dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            n_keep = cumulative[step]
            _, topk_idx = conf.topk(min(n_keep, L))

            new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
            new_gen[topk_idx] = x0[topk_idx]
            x[0, P:P+L] = new_gen

            curr = x[0, P:P+L].clone()

            # Classify each position's change
            prev_is_mask = (prev == MASK_ID)
            curr_is_mask = (curr == MASK_ID)

            n_new = ((prev_is_mask) & (~curr_is_mask)).sum().item()  # MASK → token
            n_demoted = ((~prev_is_mask) & (curr_is_mask)).sum().item()  # token → MASK
            n_changed = ((~prev_is_mask) & (~curr_is_mask) & (prev != curr)).sum().item()  # token → different token
            n_kept = ((~prev_is_mask) & (~curr_is_mask) & (prev == curr)).sum().item()  # token → same token

            # Decode current text
            text = tokenizer.decode(curr, skip_special_tokens=False)
            # Replace MASK tokens with ███ for readability
            mask_token_str = tokenizer.decode([MASK_ID])
            text_display = text.replace(mask_token_str, '█')

            # Find #### answer if present
            answer_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text_display)
            answer = answer_match.group(1) if answer_match else ""

            avg_conf = conf.mean().item()
            n_unmasked = (~curr_is_mask).sum().item()

            snapshots.append({
                "step": step,
                "n_unmasked": n_unmasked,
                "n_new": n_new,
                "n_demoted": n_demoted,
                "n_changed": n_changed,
                "n_kept": n_kept,
                "avg_conf": avg_conf,
                "answer": answer,
                "text": text_display[:500],
            })

    return snapshots


def compare_traces(snaps_orig, snaps_v5, gold, question_short):
    """Print side-by-side comparison focusing on key moments."""
    print(f"\n{'='*80}")
    print(f"Question: {question_short}")
    print(f"Gold answer: {gold}")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Step':>4} | {'--- Original ---':^35} | {'--- D3IM-A v5 ---':^35}")
    print(f"     | {'unmask new dem chg conf ans':>35} | {'unmask new dem chg conf ans':>35}")
    print("-" * 80)

    key_steps = [0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 60, 63]

    for s in key_steps:
        if s >= len(snaps_orig) or s >= len(snaps_v5):
            continue
        o = snaps_orig[s]
        v = snaps_v5[s]
        o_str = f"{o['n_unmasked']:>4} {o['n_new']:>3} {o['n_demoted']:>3} {o['n_changed']:>3} {o['avg_conf']:.2f} {o['answer'][:6]:>6}"
        v_str = f"{v['n_unmasked']:>4} {v['n_new']:>3} {v['n_demoted']:>3} {v['n_changed']:>3} {v['avg_conf']:.2f} {v['answer'][:6]:>6}"
        print(f"{s:>4} | {o_str:>35} | {v_str:>35}")

    # Final comparison
    o_final = snaps_orig[-1]
    v_final = snaps_v5[-1]
    o_ans = o_final['answer'] or "(none)"
    v_ans = v_final['answer'] or "(none)"
    o_correct = "✅" if o_ans.replace(",","").strip() == str(gold).strip() else "❌"
    v_correct = "✅" if v_ans.replace(",","").strip() == str(gold).strip() else "❌"

    print(f"\nFinal answer: Original={o_ans} {o_correct}  |  v5={v_ans} {v_correct}")

    # Total demotions
    o_dem = sum(s['n_demoted'] for s in snaps_orig)
    v_dem = sum(s['n_demoted'] for s in snaps_v5)
    o_chg = sum(s['n_changed'] for s in snaps_orig)
    v_chg = sum(s['n_changed'] for s in snaps_v5)
    print(f"Total demotions: Original={o_dem}  v5={v_dem}")
    print(f"Total corrections: Original={o_chg}  v5={v_chg}")

    # Show final text snippets (last 200 chars)
    print(f"\nOriginal final text (last 200 chars):")
    print(f"  ...{snaps_orig[-1]['text'][-200:]}")
    print(f"\nv5 final text (last 200 chars):")
    print(f"  ...{snaps_v5[-1]['text'][-200:]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    data = json.load(open("eval_data/gsm8k_100.json"))[:args.n]

    # Load both models
    print("Loading Original...")
    model_orig, tokenizer = load_model("GSAI-ML/LLaDA-8B-Instruct", "", device)

    for i, item in enumerate(data):
        prompt = (f"Solve step by step. Put final answer after ####.\n"
                  f"Question: {item['question']}\nAnswer:")

        print(f"\n[{i+1}/{args.n}] Tracing Original...")
        snaps_orig = trace_d3im_detailed(model_orig, tokenizer, prompt, device)

        # Store for comparison
        data[i]['snaps_orig'] = snaps_orig

    del model_orig
    torch.cuda.empty_cache()

    print("\nLoading v5 r=64...")
    model_v5, _ = load_model("GSAI-ML/LLaDA-8B-Instruct",
                              "checkpoints/d3im_lora_v5_r64/checkpoint-500", device)

    for i, item in enumerate(data):
        prompt = (f"Solve step by step. Put final answer after ####.\n"
                  f"Question: {item['question']}\nAnswer:")

        print(f"\n[{i+1}/{args.n}] Tracing v5...")
        snaps_v5 = trace_d3im_detailed(model_v5, tokenizer, prompt, device)

        compare_traces(item['snaps_orig'], snaps_v5,
                       item['gold_answer'], item['question'][:80])
