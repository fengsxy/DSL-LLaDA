"""Quick evaluation: remasking accuracy + calibration (ECE) comparison.

Compares original LLaDA-8B-Instruct vs DSL-trained checkpoint on:
1. Infilling accuracy at various mask ratios
2. Calibration (ECE): model confidence vs actual accuracy

Usage:
    python dsl_llada/eval_quick.py [checkpoint_path] [--device cuda:0]
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))

MASK_ID = 126336


def load_model(path, device='cuda:0'):
    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    return model.to(device).eval()


def get_eval_data(tokenizer, n_samples=50, seq_len=512, device='cuda:0'):
    """Get eval data from wikitext."""
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test', streaming=True)
    samples = []
    for item in ds:
        text = item['text'].strip()
        if len(text) < 100:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)[:seq_len]
        if len(ids) < seq_len:
            continue
        samples.append(torch.tensor(ids, device=device))
        if len(samples) >= n_samples:
            break
    return torch.stack(samples)  # (n_samples, seq_len)


@torch.no_grad()
def eval_infilling(model, input_ids, mask_ratios=[0.3, 0.5, 0.7]):
    """Test infilling at various mask ratios. Returns {ratio: accuracy}."""
    results = {}
    B, L = input_ids.shape
    for ratio in mask_ratios:
        n_mask = int(L * ratio)
        total_correct = 0
        total_masked = 0
        for i in range(B):
            perm = torch.randperm(L, device=input_ids.device)[:n_mask]
            masked = input_ids[i:i+1].clone()
            masked[0, perm] = MASK_ID
            logits = model(masked).logits
            preds = logits[0].argmax(dim=-1)
            total_correct += (preds[perm] == input_ids[i, perm]).sum().item()
            total_masked += n_mask
        results[ratio] = total_correct / total_masked
    return results


@torch.no_grad()
def eval_calibration(model, input_ids, n_bins=10):
    """Compute ECE: mask 50% tokens, compare confidence vs accuracy.

    Returns: (bin_confs, bin_accs, ece, avg_confidence, avg_accuracy)
    """
    B, L = input_ids.shape
    all_conf, all_correct = [], []

    for i in range(B):
        n_mask = L // 2
        perm = torch.randperm(L, device=input_ids.device)[:n_mask]
        masked = input_ids[i:i+1].clone()
        masked[0, perm] = MASK_ID

        logits = model(masked).logits.float()
        probs = F.softmax(logits[0], dim=-1)
        conf, pred = probs.max(dim=-1)

        all_conf.append(conf[perm].cpu())
        all_correct.append((pred[perm] == input_ids[i, perm]).cpu())

    all_conf = torch.cat(all_conf).numpy()
    all_correct = torch.cat(all_correct).numpy().astype(float)

    # Bin and compute ECE
    bins = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (all_conf >= lo) & (all_conf < hi)
        n = mask.sum()
        if n > 0:
            bin_conf.append(all_conf[mask].mean())
            bin_acc.append(all_correct[mask].mean())
            bin_count.append(n)

    bin_conf = np.array(bin_conf)
    bin_acc = np.array(bin_acc)
    bin_count = np.array(bin_count)

    ece = (bin_count * np.abs(bin_conf - bin_acc)).sum() / bin_count.sum()
    return bin_conf, bin_acc, ece, all_conf.mean(), all_correct.mean()


def plot_comparison(results, save_path=None):
    """Plot calibration comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: calibration curves
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.5)
    for name, r in results.items():
        ax.plot(r['cal_conf'], r['cal_acc'], 'o-',
                label=f"{name} (ECE={r['ece']:.4f})")
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Calibration (Reliability Diagram)')
    ax.legend()

    # Right: infilling accuracy
    ax = axes[1]
    for name, r in results.items():
        ratios = sorted(r['infill'].keys())
        accs = [r['infill'][ratio] for ratio in ratios]
        ax.plot(ratios, accs, 'o-', label=name)
    ax.set_xlabel('Mask Ratio')
    ax.set_ylabel('Recovery Accuracy')
    ax.set_title('Infilling Accuracy')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


def eval_model(name, model, input_ids):
    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print(f"{'='*50}")

    # Infilling
    infill = eval_infilling(model, input_ids)
    for ratio, acc in sorted(infill.items()):
        print(f"  Infilling (mask={ratio:.0%}): accuracy={acc:.4f}")

    # Calibration
    cal_conf, cal_acc, ece, avg_conf, avg_acc = eval_calibration(model, input_ids)
    print(f"  ECE: {ece:.4f}")
    print(f"  Avg confidence: {avg_conf:.4f}, Avg accuracy: {avg_acc:.4f}")
    print(f"  Calibration bins:")
    for c, a in zip(cal_conf, cal_acc):
        bar = '█' * int(a * 20)
        print(f"    conf={c:.2f}  acc={a:.4f}  {bar}")

    return {
        'infill': infill,
        'cal_conf': cal_conf,
        'cal_acc': cal_acc,
        'ece': ece,
        'avg_conf': avg_conf,
        'avg_acc': avg_acc,
    }


if __name__ == '__main__':
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    device = 'cuda:0'
    for arg in sys.argv:
        if arg.startswith('--device'):
            device = sys.argv[sys.argv.index(arg) + 1]

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print("Loading eval data...")
    input_ids = get_eval_data(tokenizer, n_samples=50, seq_len=512, device=device)
    print(f"Eval data: {input_ids.shape}")

    results = {}

    # Original model
    print("\nLoading original LLaDA-8B-Instruct...")
    model_orig = load_model('GSAI-ML/LLaDA-8B-Instruct', device)
    results['Original'] = eval_model('Original LLaDA-8B-Instruct', model_orig, input_ids)
    del model_orig
    torch.cuda.empty_cache()

    # DSL checkpoint
    if checkpoint:
        print(f"\nLoading DSL checkpoint: {checkpoint}")
        model_dsl = load_model(checkpoint, device)
        results['DSL-50step'] = eval_model(f'DSL-50step ({checkpoint})', model_dsl, input_ids)
        del model_dsl
        torch.cuda.empty_cache()

    # Plot comparison
    if len(results) > 1:
        save_path = 'docs/eval_comparison_50step.png'
        os.makedirs('docs', exist_ok=True)
        plot_comparison(results, save_path)

    print("\n" + "="*50)
    print("Summary:")
    for name, r in results.items():
        print(f"  {name}: ECE={r['ece']:.4f}, "
              f"infill@50%={r['infill'][0.5]:.4f}, "
              f"avg_conf={r['avg_conf']:.4f}, avg_acc={r['avg_acc']:.4f}")
