"""GSM8K trajectory analysis using LLaDA's OFFICIAL generate() with save_trajectory.

This ensures identical behavior to the actual eval (confidence_eos_eot_inf, d3im, etc.)
"""
import os, sys, torch, json, re
import numpy as np
from tqdm import tqdm
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


def extract_number(text):
    """Extract number after #### only. No fallback."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return ""


def analyze_trajectory(traj_state, tokenizer, gold):
    """Analyze a trajectory tensor (steps, gen_length)."""
    steps, L = traj_state.shape

    total_dem = 0
    total_chg = 0
    total_new = 0
    step_correct = []
    first_correct = -1

    for s in range(steps):
        curr = traj_state[s]
        prev = traj_state[s - 1] if s > 0 else torch.full((L,), MASK_ID)

        prev_committed = (prev != MASK_ID)
        curr_committed = (curr != MASK_ID)

        n_new = ((~prev_committed) & curr_committed).sum().item()
        n_dem = (prev_committed & (~curr_committed)).sum().item()
        n_chg = (prev_committed & curr_committed & (prev != curr)).sum().item()

        total_new += n_new
        total_dem += n_dem
        total_chg += n_chg

        # Check answer at this step
        text = tokenizer.decode(curr[curr != MASK_ID], skip_special_tokens=True)
        pred = extract_number(text)
        is_correct = (pred == str(gold).strip())
        step_correct.append(is_correct)

        if is_correct and first_correct < 0:
            first_correct = s

    final_text = tokenizer.decode(traj_state[-1], skip_special_tokens=True)
    final_pred = extract_number(final_text)
    final_correct = (final_pred == str(gold).strip())

    # Lost answer: was correct at some point but not at end
    ever_correct = any(step_correct)
    lost = ever_correct and not final_correct

    return {
        "total_demotions": total_dem,
        "total_corrections": total_chg,
        "total_new": total_new,
        "final_correct": final_correct,
        "final_pred": final_pred,
        "ever_correct": ever_correct,
        "lost_answer": lost,
        "first_correct": first_correct,
        "step_correct": step_correct,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=512)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    data = json.load(open("eval_data/gsm8k_100.json"))[:args.n]

    all_results = {}

    for name, lora in [("Original", ""), ("v5_r64", "checkpoints/d3im_lora_v5_r64/checkpoint-500")]:
        print(f"\nLoading {name}...")
        model, tokenizer = load_model("GSAI-ML/LLaDA-8B-Instruct", lora, device)

        results = []
        for item in tqdm(data, desc=name):
            prompt = (f"Solve step by step. Put final answer after ####.\n"
                      f"Question: {item['question']}\nAnswer:")
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)

            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                out = generate(
                    model, input_ids, attention_mask,
                    steps=args.steps, gen_length=args.gen_length,
                    block_length=args.gen_length,
                    temperature=0.0, cfg_scale=0.0,
                    remasking='d3im',
                    confidence_eos_eot_inf=True,
                    save_trajectory=True,
                )

            if isinstance(out, tuple):
                x, traj = out
                traj_state = traj['x_state']  # (steps, gen_length)
            else:
                print(f"  WARNING: save_trajectory not supported, skipping")
                continue

            r = analyze_trajectory(traj_state, tokenizer, item["gold_answer"])
            r["id"] = item["id"]
            r["gold"] = item["gold_answer"]
            results.append(r)

        all_results[name] = results
        del model; torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 80)
    print("GSM8K D3IM TRAJECTORY ANALYSIS (OFFICIAL generate())")
    print("=" * 80)

    for name in ["Original", "v5_r64"]:
        results = all_results[name]
        n = len(results)
        correct = sum(1 for r in results if r["final_correct"])
        dem = [r["total_demotions"] for r in results]
        chg = [r["total_corrections"] for r in results]
        ever = sum(1 for r in results if r["ever_correct"])
        lost = sum(1 for r in results if r["lost_answer"])

        print(f"\n--- {name} ---")
        print(f"  Accuracy: {correct}/{n} ({correct/n*100:.1f}%)")
        print(f"  Demotions: mean={np.mean(dem):.1f}, median={np.median(dem):.0f}")
        print(f"  Corrections: mean={np.mean(chg):.1f}, median={np.median(chg):.0f}")
        print(f"  Ever-correct: {ever}/{n}, Lost: {lost} ({lost/max(1,ever)*100:.0f}%)")

        # Demotion for correct vs wrong
        c_dem = [r["total_demotions"] for r in results if r["final_correct"]]
        w_dem = [r["total_demotions"] for r in results if not r["final_correct"]]
        if c_dem and w_dem:
            print(f"  Correct avg_dem={np.mean(c_dem):.0f}, Wrong avg_dem={np.mean(w_dem):.0f}")

    # v5 correct but Original wrong
    print(f"\n=== v5 correct, Original wrong ===")
    for ro, rv in zip(all_results["Original"], all_results["v5_r64"]):
        if rv["final_correct"] and not ro["final_correct"]:
            print(f"  Q{ro['id']} gold={ro['gold']}: orig={ro['final_pred']} dem={ro['total_demotions']} | v5={rv['final_pred']} dem={rv['total_demotions']}")

    print(f"\n=== Original correct, v5 wrong ===")
    for ro, rv in zip(all_results["Original"], all_results["v5_r64"]):
        if ro["final_correct"] and not rv["final_correct"]:
            print(f"  Q{ro['id']} gold={ro['gold']}: orig={ro['final_pred']} dem={ro['total_demotions']} | v5={rv['final_pred']} dem={rv['total_demotions']}")

    # Step-wise accuracy
    print(f"\n=== Step-wise Accuracy ===")
    print(f"{'Step':>4} | {'Original':>8} | {'v5_r64':>8}")
    print("-" * 30)
    n_steps = len(all_results["Original"][0]["step_correct"])
    for s in range(0, n_steps, max(1, n_steps // 10)):
        o_acc = sum(1 for r in all_results["Original"] if r["step_correct"][s]) / n * 100
        v_acc = sum(1 for r in all_results["v5_r64"] if r["step_correct"][s]) / n * 100
        print(f"{s:>4} | {o_acc:>7.1f}% | {v_acc:>7.1f}%")

    # Save
    output_path = "eval_results/gsm8k_trace_official.json"
    save_data = {}
    for name, results in all_results.items():
        save_data[name] = [{k: v for k, v in r.items() if k != "step_correct"} for r in results]
        save_data[f"{name}_step_curves"] = [r["step_correct"] for r in results]
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_path}")
