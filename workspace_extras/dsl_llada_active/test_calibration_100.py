"""Calibration evaluation — 100+ text samples, token-level + task-level.

Token-level: ECE (Expected Calibration Error) at mask=30%/50%/70%
Task-level: Per-problem average confidence vs correctness on GSM8K

Outputs:
  results/calibration_token_100.json  — token-level ECE
  results/calibration_task_gsm.json   — task-level calibration bins
"""
import os, sys, torch, json
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6,7')

from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336
DEVICE_ORIG = 'cuda:0'
DEVICE_CKPT = 'cuda:1'

# Load texts
script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, 'eval_texts_100.json')) as f:
    TEXTS_FINEWEB = json.load(f)
with open(os.path.join(script_dir, 'eval_texts_gsm100.json')) as f:
    TEXTS_GSM = json.load(f)
print(f"Loaded {len(TEXTS_FINEWEB)} fineweb + {len(TEXTS_GSM)} GSM8K texts")


def compute_ece(confidences, accuracies, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # include upper bound in last bin
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            bin_data.append({'lo': lo, 'hi': hi, 'n': 0, 'avg_conf': 0, 'avg_acc': 0})
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        n = mask.sum()
        ece += (n / len(confidences)) * abs(avg_acc - avg_conf)
        bin_data.append({
            'lo': float(lo), 'hi': float(hi),
            'n': int(n), 'avg_conf': float(avg_conf), 'avg_acc': float(avg_acc),
        })

    return float(ece), bin_data


def eval_token_calibration(model, tokenizer, texts, device, mask_rates):
    """Token-level calibration: confidence vs accuracy at masked positions."""
    results = {}
    for rate in mask_rates:
        all_confs = []
        all_accs = []

        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
            input_ids = tokens['input_ids'].to(device)
            L = input_ids.shape[1]
            if L < 10:
                continue

            mask = torch.rand(1, L, device=device) < rate
            masked = input_ids.clone()
            masked[0, mask[0]] = MASK_ID

            with torch.no_grad():
                out = model(masked)
                logits = out.logits[0].float()
                probs = F.softmax(logits, dim=-1)
                pred_conf, pred_ids = probs.max(dim=-1)

            if mask[0].any():
                m = mask[0]
                correct = (pred_ids[m] == input_ids[0, m]).cpu().numpy().astype(float)
                conf = pred_conf[m].cpu().numpy()
                all_confs.extend(conf.tolist())
                all_accs.extend(correct.tolist())

        all_confs = np.array(all_confs)
        all_accs = np.array(all_accs)
        ece, bins = compute_ece(all_confs, all_accs)

        results[rate] = {
            'ece': ece,
            'n_tokens': len(all_confs),
            'avg_conf': float(all_confs.mean()),
            'avg_acc': float(all_accs.mean()),
            'bins': bins,
        }
        print(f"    mask={rate:.0%}: ECE={ece:.4f}, acc={all_accs.mean():.3f}, "
              f"conf={all_confs.mean():.3f}, n={len(all_confs)}", flush=True)

    return results


def eval_task_calibration(model, tokenizer, texts, device, n_steps=64, gen_length=256):
    """Task-level calibration on GSM8K: per-problem confidence vs correctness.

    For each problem:
    1. Mask the answer portion
    2. Run n_steps of standard remasking
    3. Record average confidence across all generated tokens
    4. Check if answer is correct
    """
    from dsl_llada.eval_reasoning import extract_number, normalize_answer

    all_confs = []
    all_correct = []

    for i, text in enumerate(texts):
        # Parse question and gold answer
        parts = text.split('\nAnswer: ', 1)
        if len(parts) != 2:
            continue
        question = parts[0].replace('Question: ', '')
        gold_answer = parts[1]
        # Extract numeric answer
        gold_num = extract_number(gold_answer)

        # Tokenize question
        q_tokens = tokenizer(question, return_tensors='pt', add_special_tokens=False)
        q_ids = q_tokens['input_ids'].to(device)
        q_len = q_ids.shape[1]

        # Create input: question tokens + MASK tokens for generation
        total_len = q_len + gen_length
        input_ids = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
        input_ids[0, :q_len] = q_ids[0]

        # Standard remasking generation
        with torch.no_grad():
            x = input_ids.clone()
            gen_mask = torch.zeros(1, total_len, dtype=torch.bool, device=device)
            gen_mask[0, q_len:] = True

            for step in range(n_steps):
                out = model(x)
                logits = out.logits[0].float()
                probs = F.softmax(logits, dim=-1)
                pred_conf, pred_ids = probs.max(dim=-1)

                # Only update masked positions
                still_masked = (x[0] == MASK_ID) & gen_mask[0]
                if not still_masked.any():
                    break

                if step < n_steps - 1:
                    # Unmask fraction of remaining masked positions
                    n_to_unmask = max(1, int(still_masked.sum().item() * 1.0 / (n_steps - step)))
                    # Use confidence to decide which to unmask
                    conf_at_masked = pred_conf.clone()
                    conf_at_masked[~still_masked] = -1.0
                    _, top_idx = conf_at_masked.topk(n_to_unmask)
                    x[0, top_idx] = pred_ids[top_idx]
                else:
                    # Last step: unmask all remaining
                    x[0, still_masked] = pred_ids[still_masked]

            # Get final confidence for all generated tokens
            out = model(x)
            final_probs = F.softmax(out.logits[0].float(), dim=-1)
            final_conf, _ = final_probs.max(dim=-1)
            gen_conf = final_conf[q_len:].mean().item()

            # Get generated text and extract answer
            gen_text = tokenizer.decode(x[0, q_len:], skip_special_tokens=True)
            pred_num = extract_number(gen_text)
            correct = (normalize_answer(pred_num) == normalize_answer(gold_num)
                       if pred_num and gold_num else False)

        all_confs.append(gen_conf)
        all_correct.append(1.0 if correct else 0.0)

        if (i + 1) % 10 == 0:
            print(f"    Task cal: {i+1}/{len(texts)}, "
                  f"acc_so_far={np.mean(all_correct):.1%}, "
                  f"avg_conf={np.mean(all_confs):.3f}", flush=True)

    all_confs = np.array(all_confs)
    all_correct = np.array(all_correct)
    ece, bins = compute_ece(all_confs, all_correct, n_bins=5)

    return {
        'ece': float(ece),
        'n_problems': len(all_confs),
        'avg_conf': float(all_confs.mean()),
        'avg_acc': float(all_correct.mean()),
        'bins': bins,
        'per_problem': [{'conf': float(c), 'correct': int(a)}
                        for c, a in zip(all_confs, all_correct)],
    }


def main(ckpt_path=None, output_prefix=None):
    if ckpt_path is None:
        ckpt_path = './checkpoints/dsl_1000step/checkpoint-1000'
    if output_prefix is None:
        output_prefix = 'results/calibration'

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print("Loading Original LLaDA...")
    model_orig = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(DEVICE_ORIG).eval()

    print(f"Loading checkpoint: {ckpt_path}")
    model_ckpt = AutoModel.from_pretrained(
        ckpt_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(DEVICE_CKPT).eval()

    os.makedirs('results', exist_ok=True)

    # ---- Part 1: Token-level calibration ----
    print("\n" + "=" * 60)
    print("  TOKEN-LEVEL CALIBRATION (fineweb 100 + GSM8K 100)")
    print("=" * 60)

    mask_rates = [0.3, 0.5, 0.7]
    all_texts = TEXTS_FINEWEB + TEXTS_GSM  # 200 texts total

    print(f"\n  Original LLaDA ({len(all_texts)} texts):")
    torch.manual_seed(42)
    cal_orig = eval_token_calibration(model_orig, tokenizer, all_texts, DEVICE_ORIG, mask_rates)

    print(f"\n  Softmasker ({len(all_texts)} texts):")
    torch.manual_seed(42)
    cal_ckpt = eval_token_calibration(model_ckpt, tokenizer, all_texts, DEVICE_CKPT, mask_rates)

    print(f"\n  Summary — Token-level ECE:")
    print(f"  {'Mask Rate':>10s} | {'Original':>10s} | {'Softmasker':>10s} | {'Δ':>8s}")
    print(f"  {'-'*45}")
    for rate in mask_rates:
        o = cal_orig[rate]['ece']
        c = cal_ckpt[rate]['ece']
        print(f"  {rate:>9.0%}  | {o:>10.4f} | {c:>10.4f} | {c-o:>+8.4f}")

    token_results = {
        'original': {str(k): v for k, v in cal_orig.items()},
        'softmasker': {str(k): v for k, v in cal_ckpt.items()},
        'n_texts': len(all_texts),
    }
    token_out = f'{output_prefix}_token_100.json'
    os.makedirs(os.path.dirname(token_out) or '.', exist_ok=True)
    with open(token_out, 'w') as f:
        json.dump(token_results, f, indent=2)
    print(f"  Saved to {token_out}")

    # ---- Part 2: Task-level calibration ----
    print("\n" + "=" * 60)
    print("  TASK-LEVEL CALIBRATION (GSM8K 100 problems)")
    print("=" * 60)

    print(f"\n  Original LLaDA:")
    task_orig = eval_task_calibration(
        model_orig, tokenizer, TEXTS_GSM, DEVICE_ORIG, n_steps=64, gen_length=256)

    print(f"\n  Softmasker:")
    task_ckpt = eval_task_calibration(
        model_ckpt, tokenizer, TEXTS_GSM, DEVICE_CKPT, n_steps=64, gen_length=256)

    print(f"\n  Summary — Task-level:")
    print(f"  {'':>20s} | {'Original':>10s} | {'Softmasker':>10s}")
    print(f"  {'-'*45}")
    print(f"  {'ECE':>20s} | {task_orig['ece']:>10.4f} | {task_ckpt['ece']:>10.4f}")
    print(f"  {'Avg Confidence':>20s} | {task_orig['avg_conf']:>10.3f} | {task_ckpt['avg_conf']:>10.3f}")
    print(f"  {'Accuracy':>20s} | {task_orig['avg_acc']:>10.1%} | {task_ckpt['avg_acc']:>10.1%}")

    print(f"\n  Confidence bins:")
    print(f"  {'Bin':>15s} | {'Orig Acc':>8s} {'N':>4s} | {'SM Acc':>8s} {'N':>4s}")
    for bo, bc in zip(task_orig['bins'], task_ckpt['bins']):
        if bo['n'] > 0 or bc['n'] > 0:
            o_acc = f"{bo['avg_acc']:.1%}" if bo['n'] > 0 else "—"
            c_acc = f"{bc['avg_acc']:.1%}" if bc['n'] > 0 else "—"
            print(f"  [{bo['lo']:.1f},{bo['hi']:.1f})"
                  f"  | {o_acc:>8s} {bo['n']:>4d} | {c_acc:>8s} {bc['n']:>4d}")

    task_results = {
        'original': task_orig,
        'softmasker': task_ckpt,
    }
    task_out = f'{output_prefix}_task_gsm.json'
    with open(task_out, 'w') as f:
        json.dump(task_results, f, indent=2)
    print(f"  Saved to {task_out}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Checkpoint path (default: ./checkpoints/dsl_1000step/checkpoint-1000)')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Output file prefix (default: results/calibration)')
    args = parser.parse_args()
    main(ckpt_path=args.ckpt_path, output_prefix=args.output_prefix)
