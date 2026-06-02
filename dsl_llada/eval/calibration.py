# dsl_llada/eval/calibration.py
"""
Calibration curve (reliability diagram) for a trained DSLLaDA checkpoint.

Usage:
    python dsl_llada/eval/calibration.py \
        --checkpoint ./checkpoints/dsl_phase1/checkpoint-1000 \
        --save_path docs/results/calibration.png \
        --n_batches 100
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_calibration(model, dataloader, snr_max=100.0, n_bins=10, device='cuda'):
    """
    Evaluate calibration: for each masked position, compare model confidence
    to whether the prediction was correct.

    Strategy: for each batch, randomly mask half the tokens (SNR=0) and give
    the model the other half (SNR=snr_max). Measure confidence vs accuracy on
    the masked half.

    Returns:
        bin_conf (np.ndarray): mean confidence per bin.
        bin_acc  (np.ndarray): mean accuracy per bin.
        ece      (float):      expected calibration error.
    """
    all_conf, all_correct = [], []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            B, L = input_ids.shape

            snrs = torch.zeros(B, L, device=device)
            for i in range(B):
                k = L // 2
                perm = torch.randperm(L, device=device)[:k]
                snrs[i, perm] = snr_max

            out = model(input_ids=input_ids, snrs=snrs)
            probs = F.softmax(out.logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)

            masked = (snrs == 0)
            all_conf.extend(conf[masked].cpu().numpy())
            all_correct.extend((pred[masked] == input_ids[masked]).cpu().numpy())

    all_conf = np.array(all_conf)
    all_correct = np.array(all_correct)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (all_conf >= lo) & (all_conf < hi)
        if mask.sum() > 0:
            bin_conf.append(all_conf[mask].mean())
            bin_acc.append(all_correct[mask].mean())
            bin_count.append(mask.sum())

    bin_conf = np.array(bin_conf)
    bin_acc = np.array(bin_acc)
    bin_count = np.array(bin_count)

    ece = (bin_count * np.abs(bin_conf - bin_acc)).sum() / bin_count.sum()
    return bin_conf, bin_acc, ece


def plot_calibration(bin_conf, bin_acc, ece, label='DSL-LLaDA', save_path=None):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(bin_conf, bin_acc, 'o-', label=f'{label} (ECE={ece:.3f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to DSLLaDA checkpoint dir')
    parser.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu',
                        help='HF dataset name for calibration eval')
    parser.add_argument('--dataset_config', default='sample-10BT')
    parser.add_argument('--n_batches', type=int, default=100,
                        help='Number of batches to evaluate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--n_bins', type=int, default=10)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'LLaDA-XDLM'))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from dsl_llada.core.dsl_modules import DSLLaDA
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    print(f"Loading checkpoint: {args.checkpoint}")
    # DSLLaDA checkpoints save the full wrapper; load with torch.load or HF API
    llada = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = DSLLaDA(llada).to(args.device)
    model.eval()

    print(f"Loading dataset: {args.dataset} ({args.dataset_config})")
    ds = load_dataset(args.dataset, args.dataset_config, split='train', streaming=True)

    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.seq_len,
                         padding='max_length', return_tensors='pt')

    def collate(batch):
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch])
        return {'input_ids': input_ids}

    ds_mapped = ds.map(lambda x: {'input_ids': tokenizer.encode(
        x['text'], truncation=True, max_length=args.seq_len,
        padding='max_length')[:args.seq_len]})
    ds_mapped = ds_mapped.take(args.n_batches * args.batch_size)

    dataloader = DataLoader(list(ds_mapped), batch_size=args.batch_size, collate_fn=collate)

    print("Computing calibration...")
    bin_conf, bin_acc, ece = compute_calibration(
        model, dataloader, n_bins=args.n_bins, device=args.device)

    print(f"ECE: {ece:.4f}")
    for c, a in zip(bin_conf, bin_acc):
        print(f"  conf={c:.2f}  acc={a:.2f}")

    plot_calibration(bin_conf, bin_acc, ece, save_path=args.save_path)


if __name__ == '__main__':
    main()
