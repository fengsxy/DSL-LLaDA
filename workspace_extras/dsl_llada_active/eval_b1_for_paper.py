"""Quick eval of b1 (β=1) on context robustness curves.

Re-runs the eval_sde_gen_formal exp 2.2 protocol but for the b1 model only,
to fill in numbers in tab:context_robust. Existing data file already has
original/mdm_cpt/sm_b2/replace; we add b1.

Output: eval_results/sde_gen_formal/multi_mask_robustness_b1.json
"""
import json
import os
import sys
import torch
import numpy as np

sys.path.insert(0, '/home/ubuntu/efs/RMDM/LLaDA')
from transformers import AutoModelForCausalLM, AutoTokenizer

MASK_ID = 126336
device = 'cuda:0'
seed = 42

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
texts_data = json.load(open('/home/ubuntu/efs/RMDM/eval_data/texts_100.json'))

ckpt = '/home/ubuntu/efs/RMDM/checkpoints/pertoken_b1_d100_1k/checkpoint-1000'
print(f'Loading b1 from {ckpt}', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device).eval()

mask_rates = [0.3, 0.5, 0.7]
corruption_rates = [0, 10, 20, 30]

torch.manual_seed(seed)
np.random.seed(seed)

results = {}
for mr in mask_rates:
    for cr in corruption_rates:
        total_correct = 0
        total_masked = 0
        for text_obj in texts_data:
            text = text_obj['text'] if isinstance(text_obj, dict) else text_obj
            ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).input_ids.to(device)
            L = ids.shape[1]
            if L < 20: continue

            # Set RNG per (text, mr, cr) for reproducibility
            rng = np.random.RandomState(seed + L + int(mr*100) + cr)
            n_mask = max(1, int(L * mr))
            mask_pos = rng.choice(L, n_mask, replace=False)
            mask_pos.sort()

            # Corrupt some clean positions
            clean_pos = np.array([i for i in range(L) if i not in set(mask_pos)])
            if len(clean_pos) > 0 and cr > 0:
                n_corr = max(0, int(len(clean_pos) * cr / 100))
                if n_corr > 0:
                    corr_idx = rng.choice(len(clean_pos), n_corr, replace=False)
                    corr_pos = clean_pos[corr_idx]
                    rand_tokens = torch.randint(100, 126000, (n_corr,), device=device)
                    corrupted_ids = ids.clone()
                    corrupted_ids[0, corr_pos] = rand_tokens
                else:
                    corrupted_ids = ids.clone()
            else:
                corrupted_ids = ids.clone()

            input_ids = corrupted_ids.clone()
            input_ids[0, mask_pos] = MASK_ID

            with torch.no_grad():
                logits = model(input_ids).logits

            preds = logits[0, mask_pos].argmax(dim=-1)
            gold = ids[0, mask_pos]
            total_correct += (preds == gold).sum().item()
            total_masked += len(mask_pos)

        acc = total_correct / total_masked if total_masked > 0 else 0
        key = f'mask{int(mr*100)}_corrupt{cr}'
        results[key] = round(acc, 4)
        print(f'  b1 mask={int(mr*100)}% corrupt={cr}%: acc={acc:.4f}', flush=True)

out = '/home/ubuntu/efs/RMDM/eval_results/sde_gen_formal/multi_mask_robustness_b1.json'
with open(out, 'w') as f:
    json.dump({'mask_rates': mask_rates, 'corruption_rates': corruption_rates, 'seed': seed, 'results': {'b1': results}}, f, indent=2)
print(f'Saved to {out}')
