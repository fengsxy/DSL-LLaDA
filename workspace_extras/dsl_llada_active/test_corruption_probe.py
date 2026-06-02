"""Probing experiment: Token Corruption Robustness.

Core question: Does DSL training teach the backbone to correct wrong tokens,
not just fill in masks?

Design:
- Take real text, replace X% of tokens with random WRONG tokens (not MASK)
- Feed corrupted text to model, get predictions for ALL positions
- Measure: can the model correct the corrupted tokens?
- Compare original LLaDA vs ckpt-1000 (DSL-trained)

If ckpt-1000 can correct errors while original can't, it proves DSL training
gave the backbone a genuinely new ability: error correction.
"""
import os, sys, torch, json
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336

# Diverse text samples — longer for more statistical power
TEXTS = [
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the park, and children were playing on the swings while their parents watched from nearby benches.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Deep learning, a further subset, uses neural networks with many layers.",
    "To make a chocolate cake, you need flour, sugar, cocoa powder, eggs, butter, and milk. Preheat the oven to 350 degrees Fahrenheit and grease a nine-inch round cake pan.",
    "The capital of France is Paris. It is known for the Eiffel Tower, the Louvre Museum, and its rich cultural heritage that spans over two thousand years of history.",
    "In mathematics, the Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides. This fundamental theorem has countless applications.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure. Ice melts at 0 degrees Celsius. These phase transitions are fundamental to understanding thermodynamics.",
    "The human body has 206 bones. The largest bone is the femur, located in the thigh. The smallest bone is the stapes, found in the middle ear.",
    "Python is a popular programming language known for its simple syntax and versatility. It is widely used in data science, web development, and artificial intelligence research.",
    "The Great Wall of China is one of the most famous structures in the world. It stretches over 13,000 miles and was built over many centuries to protect against northern invasions.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This process is essential for nearly all life on Earth.",
    "Albert Einstein developed the theory of general relativity, which describes gravity as the curvature of spacetime caused by mass and energy. This theory has been confirmed by numerous experiments.",
    "The Amazon River is the largest river in the world by volume. It flows through South America and its basin covers about 40 percent of the continent.",
    "Quantum computing uses quantum mechanical phenomena such as superposition and entanglement to perform calculations. It has the potential to solve problems that are intractable for classical computers.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime. His works explore themes of love, power, jealousy, and the human condition that remain relevant today.",
    "The periodic table organizes chemical elements by their atomic number and electron configuration. Elements in the same column share similar chemical properties due to their valence electron structure.",
    "Neural networks are inspired by the structure of biological brains. They consist of layers of interconnected nodes that process information through weighted connections and activation functions.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second. Nothing can travel faster than light according to Einstein's theory of special relativity.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly burning fossil fuels, have been the main driver since the Industrial Revolution.",
    "The Fibonacci sequence starts with 0 and 1, and each subsequent number is the sum of the two preceding ones. This sequence appears frequently in nature, from flower petals to spiral galaxies.",
    "DNA carries the genetic instructions for the development and functioning of all known living organisms. Its double helix structure was discovered by Watson and Crick in 1953.",
]

CORRUPTION_RATES = [0.1, 0.2, 0.3, 0.5, 0.7]


def corrupt_tokens(input_ids, rate, vocab_size=126336):
    """Replace `rate` fraction of tokens with random wrong tokens."""
    B, L = input_ids.shape
    mask = torch.rand(B, L, device=input_ids.device) < rate
    # Generate random tokens (avoid MASK_ID and special tokens)
    random_tokens = torch.randint(100, vocab_size - 100, (B, L), device=input_ids.device)
    # Make sure replacement is actually different from original
    same = (random_tokens == input_ids)
    random_tokens[same] = (random_tokens[same] + 1) % (vocab_size - 100) + 100

    corrupted = input_ids.clone()
    corrupted[mask] = random_tokens[mask]
    return corrupted, mask


def eval_correction(model, tokenizer, texts, device, corruption_rates):
    """Evaluate model's ability to correct corrupted tokens."""
    results = {}

    for rate in corruption_rates:
        all_corrupted_acc = []  # accuracy on corrupted positions
        all_clean_acc = []      # accuracy on clean positions
        all_corrupted_conf = [] # confidence on corrupted positions
        all_clean_conf = []     # confidence on clean positions
        all_corrected_to_wrong = []  # did model change correct→wrong on clean positions?
        total_corrupted = 0
        total_clean = 0

        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
            input_ids = tokens['input_ids'].to(device)
            L = input_ids.shape[1]
            if L < 10:
                continue

            corrupted, corrupt_mask = corrupt_tokens(input_ids, rate)

            with torch.no_grad():
                # Feed corrupted text (no MASK tokens at all!)
                out = model(corrupted)
                logits = out.logits[0].float()
                probs = F.softmax(logits, dim=-1)
                pred_ids = probs.argmax(dim=-1)
                pred_conf = probs.max(dim=-1).values

            corrupt_mask = corrupt_mask[0]  # (L,)
            gold = input_ids[0]             # (L,)

            # Corrupted positions: can model recover the original token?
            if corrupt_mask.any():
                corrupted_correct = (pred_ids[corrupt_mask] == gold[corrupt_mask]).float()
                corrupted_confidence = pred_conf[corrupt_mask]
                all_corrupted_acc.append(corrupted_correct.mean().item())
                all_corrupted_conf.append(corrupted_confidence.mean().item())
                total_corrupted += corrupt_mask.sum().item()

            # Clean positions: does model keep the correct token?
            clean_mask = ~corrupt_mask
            if clean_mask.any():
                clean_correct = (pred_ids[clean_mask] == gold[clean_mask]).float()
                clean_confidence = pred_conf[clean_mask]
                all_clean_acc.append(clean_correct.mean().item())
                all_clean_conf.append(clean_confidence.mean().item())
                total_clean += clean_mask.sum().item()

                # Did the model change a correct token to wrong?
                changed = (pred_ids[clean_mask] != corrupted[0, clean_mask])
                if changed.any():
                    # Of the ones it changed, how many were changed to correct?
                    # (corrupted[clean] == gold[clean], so "changed" means model disagreed with input)
                    pass

        results[rate] = {
            'corruption_rate': rate,
            'corrupted_acc': np.mean(all_corrupted_acc) if all_corrupted_acc else 0,
            'corrupted_conf': np.mean(all_corrupted_conf) if all_corrupted_conf else 0,
            'clean_acc': np.mean(all_clean_acc) if all_clean_acc else 0,
            'clean_conf': np.mean(all_clean_conf) if all_clean_conf else 0,
            'n_corrupted': total_corrupted,
            'n_clean': total_clean,
        }

    return results


def eval_with_mask_input(model, tokenizer, texts, device, corruption_rates):
    """Control experiment: same positions, but use MASK instead of wrong tokens.
    This is what original LLaDA was trained for."""
    results = {}

    for rate in corruption_rates:
        all_acc = []
        all_conf = []

        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
            input_ids = tokens['input_ids'].to(device)
            L = input_ids.shape[1]
            if L < 10:
                continue

            mask = torch.rand(1, L, device=input_ids.device) < rate
            masked_ids = input_ids.clone()
            masked_ids[0, mask[0]] = MASK_ID

            with torch.no_grad():
                out = model(masked_ids)
                logits = out.logits[0].float()
                probs = F.softmax(logits, dim=-1)
                pred_ids = probs.argmax(dim=-1)
                pred_conf = probs.max(dim=-1).values

            if mask[0].any():
                correct = (pred_ids[mask[0]] == input_ids[0, mask[0]]).float()
                all_acc.append(correct.mean().item())
                all_conf.append(pred_conf[mask[0]].mean().item())

        results[rate] = {
            'mask_rate': rate,
            'mask_acc': np.mean(all_acc) if all_acc else 0,
            'mask_conf': np.mean(all_conf) if all_conf else 0,
        }

    return results


def print_results(name, corrupt_results, mask_results):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'Rate':>6s} | {'Corrupt→Fix':>12s} {'Conf':>6s} | {'Clean Keep':>11s} {'Conf':>6s} | {'MASK→Fill':>10s} {'Conf':>6s}")
    print(f"  {'-'*66}")
    for rate in CORRUPTION_RATES:
        cr = corrupt_results[rate]
        mr = mask_results[rate]
        print(f"  {rate:>5.0%}  | "
              f"{cr['corrupted_acc']:>11.1%} {cr['corrupted_conf']:>6.3f} | "
              f"{cr['clean_acc']:>10.1%} {cr['clean_conf']:>6.3f} | "
              f"{mr['mask_acc']:>9.1%} {mr['mask_conf']:>6.3f}")


def main():
    import argparse
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

    print(f"\nTesting on {len(TEXTS)} texts, corruption rates: {CORRUPTION_RATES}")
    print(f"Corruption = random wrong token (NOT mask)")

    torch.manual_seed(42)
    corrupt_results = eval_correction(model, tokenizer, TEXTS, device, CORRUPTION_RATES)
    torch.manual_seed(42)
    mask_results = eval_with_mask_input(model, tokenizer, TEXTS, device, CORRUPTION_RATES)

    ckpt_name = args.checkpoint.replace('/', '_').replace('.', '_')
    print_results(f"{ckpt_name}", corrupt_results, mask_results)

    # Save results
    all_results = {
        'checkpoint': args.checkpoint,
        'corrupt': {str(k): v for k, v in corrupt_results.items()},
        'mask': {str(k): v for k, v in mask_results.items()},
    }
    os.makedirs('results/corruption', exist_ok=True)
    output_path = args.output or f'results/corruption/corruption_{ckpt_name}.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
