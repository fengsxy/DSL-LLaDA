"""Quick calibration test: compare original vs ckpt-1000 confidence-accuracy relationship.

For each model, mask 50% of tokens in real text, predict, and measure:
- Per-bin: confidence vs actual accuracy (reliability diagram)
- ECE (Expected Calibration Error)
"""
import os, sys, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336

# 10 diverse text samples for calibration
TEXTS = [
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the park.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "To make a chocolate cake, you need flour, sugar, cocoa powder, eggs, butter, and milk. Preheat the oven to 350 degrees.",
    "The capital of France is Paris. It is known for the Eiffel Tower, the Louvre Museum, and its rich cultural heritage.",
    "In mathematics, the Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure. Ice melts at 0 degrees Celsius.",
    "The human body has 206 bones. The largest bone is the femur, located in the thigh.",
    "Python is a popular programming language known for its simple syntax and versatility. It is widely used in data science and web development.",
    "The Great Wall of China is one of the most famous structures in the world. It was built over many centuries to protect against invasions.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
]


def compute_calibration(confidences, correctness, n_bins=10):
    """Compute ECE and per-bin reliability data."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # include 1.0
            mask = mask | (confidences == hi)
        n = mask.sum()
        if n > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctness[mask].mean()
            bins.append({'lo': lo, 'hi': hi, 'n': int(n),
                        'avg_conf': float(avg_conf), 'avg_acc': float(avg_acc)})
        else:
            bins.append({'lo': lo, 'hi': hi, 'n': 0, 'avg_conf': 0, 'avg_acc': 0})

    # ECE
    total = len(confidences)
    ece = sum(b['n'] / total * abs(b['avg_conf'] - b['avg_acc']) for b in bins if b['n'] > 0)
    return ece, bins


def eval_model_calibration(model, tokenizer, texts, device, mask_ratio=0.5):
    """Mask tokens and measure confidence vs accuracy."""
    all_confs = []
    all_correct = []

    for text in texts:
        tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        input_ids = tokens['input_ids'].to(device)  # (1, L)
        L = input_ids.shape[1]

        # Random mask
        mask = torch.rand(L, device=device) < mask_ratio
        # Keep at least one unmasked
        if mask.all():
            mask[0] = False

        masked_ids = input_ids.clone()
        masked_ids[0, mask] = MASK_ID

        with torch.no_grad():
            out = model(masked_ids)
            logits = out.logits[0].float()  # (L, V)
            probs = F.softmax(logits, dim=-1)

        # Only evaluate on masked positions
        mask_positions = mask.nonzero(as_tuple=True)[0]
        for pos in mask_positions:
            gold = input_ids[0, pos].item()
            pred_prob, pred_id = probs[pos].max(dim=-1)
            confidence = pred_prob.item()
            correct = (pred_id.item() == gold)
            all_confs.append(confidence)
            all_correct.append(float(correct))

    return np.array(all_confs), np.array(all_correct)


def print_reliability_diagram(name, ece, bins):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  ECE = {ece:.4f}")
    print(f"{'='*60}")
    print(f"  {'Bin':>10s} {'Count':>7s} {'Avg Conf':>10s} {'Avg Acc':>10s} {'Gap':>8s}")
    print(f"  {'-'*47}")
    for b in bins:
        if b['n'] > 0:
            gap = b['avg_conf'] - b['avg_acc']
            bar_acc = '#' * int(b['avg_acc'] * 20)
            bar_conf = '.' * int(b['avg_conf'] * 20)
            print(f"  [{b['lo']:.1f},{b['hi']:.1f}) {b['n']:>6d} {b['avg_conf']:>10.4f} {b['avg_acc']:>10.4f} {gap:>+8.4f}  |{bar_acc}|")
    # Summary stats
    total_tokens = sum(b['n'] for b in bins)
    overall_acc = sum(b['n'] * b['avg_acc'] for b in bins if b['n'] > 0) / total_tokens
    overall_conf = sum(b['n'] * b['avg_conf'] for b in bins if b['n'] > 0) / total_tokens
    print(f"  {'-'*47}")
    print(f"  Total: {total_tokens} tokens, overall acc={overall_acc:.4f}, conf={overall_conf:.4f}")


def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Model checkpoint path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--output', default=None, help='Output JSON path')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print(f"Loading model: {args.checkpoint}")
    model = AutoModel.from_pretrained(
        args.checkpoint, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(device).eval()

    all_results = {'checkpoint': args.checkpoint}
    for ratio in [0.3, 0.5, 0.7]:
        torch.manual_seed(42)
        confs, correct = eval_model_calibration(model, tokenizer, TEXTS, device, mask_ratio=ratio)
        ece, bins = compute_calibration(confs, correct)
        print_reliability_diagram(f"{args.checkpoint} (mask={ratio})", ece, bins)
        overall_acc = correct.mean() if len(correct) > 0 else 0
        overall_conf = confs.mean() if len(confs) > 0 else 0
        all_results[f'mask_{ratio}'] = {
            'ece': float(ece),
            'accuracy': float(overall_acc),
            'avg_confidence': float(overall_conf),
            'bins': bins,
        }

    os.makedirs('results/calibration', exist_ok=True)
    ckpt_name = args.checkpoint.replace('/', '_').replace('.', '_')
    output_path = args.output or f'results/calibration/calibration_{ckpt_name}.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
