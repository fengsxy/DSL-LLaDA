"""Show corruption correction: Original vs Random-48 vs PCA-128.

Usage:
    python dsl_llada/show_corruption_pca.py [gpu_id]
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336
ORIGINAL = 'GSAI-ML/LLaDA-8B-Instruct'
SOFTMASKER = './checkpoints/dsl_1000step/checkpoint-1000'
PCA128 = './checkpoints/dsl_pca128_20260312_171035/checkpoint-1000'

TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower, the Louvre Museum, and its rich cultural heritage.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure. Ice melts at 0 degrees Celsius.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the park.",
]

R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'; B = '\033[1m'; D = '\033[2m'; X = '\033[0m'


def corrupt_and_predict(model, tokenizer, text, rate, seed=42):
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    input_ids = tokens['input_ids'].cuda()
    L = input_ids.shape[1]

    torch.manual_seed(seed)
    mask = torch.rand(L, device='cuda') < rate
    random_toks = torch.randint(100, MASK_ID - 100, (L,), device='cuda')
    same = (random_toks == input_ids[0])
    random_toks[same] = (random_toks[same] + 1) % (MASK_ID - 100) + 100

    corrupted = input_ids.clone()
    corrupted[0, mask] = random_toks[mask]

    with torch.no_grad():
        logits = model(corrupted).logits[0].float()
        probs = F.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)
        pred_conf = probs.max(dim=-1).values

    results = []
    for i in range(L):
        gold = input_ids[0, i].item()
        corrupt = corrupted[0, i].item()
        pred = pred_ids[i].item()
        conf = pred_conf[i].item()
        results.append({
            'pos': i, 'gold': tokenizer.decode([gold]), 'gold_id': gold,
            'corrupt': tokenizer.decode([corrupt]), 'corrupt_id': corrupt,
            'is_corrupted': mask[i].item(),
            'pred': tokenizer.decode([pred]), 'pred_id': pred,
            'confidence': conf,
            'fixed': mask[i].item() and (pred == gold),
            'kept_clean': (not mask[i].item()) and (pred == gold),
            'broke_clean': (not mask[i].item()) and (pred != gold),
        })
    return results


tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)

print('Loading Original LLaDA...')
model_orig = AutoModel.from_pretrained(ORIGINAL, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

print('Loading Softmasker (random-48, ckpt-1000)...')
model_sm = AutoModel.from_pretrained(SOFTMASKER, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

print('Loading PCA-128 (ckpt-1000)...')
model_pca = AutoModel.from_pretrained(PCA128, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

models = [
    ('Orig', model_orig),
    ('Rand48', model_sm),
    ('PCA128', model_pca),
]

for rate in [0.2, 0.5]:
    print(f'\n{"#"*90}')
    print(f'  CORRUPTION RATE = {rate:.0%} (random wrong tokens, NOT mask)')
    print(f'{"#"*90}')

    for text in TEXTS:
        print(f'\n{B}Original:{X} "{text[:80]}..."')

        all_results = {}
        for name, model in models:
            all_results[name] = corrupt_and_predict(model, tokenizer, text, rate)

        # Show corrupted input
        ro = all_results['Orig']
        corrupted_str = ''
        for r in ro:
            if r['is_corrupted']:
                corrupted_str += f'{R}{r["corrupt"]}{X}'
            else:
                corrupted_str += r['gold']
        print(f'{D}Corrupted:{X} {corrupted_str[:150]}...')
        print()

        # Header
        print(f'  {"Pos":>3} {"Gold":>12} {"→Corrupt":>12} │', end='')
        for name, _ in models:
            print(f' {name+" pred":>12} {"conf":>5} {"":>3} │', end='')
        print()
        print(f'  {"─"*3} {"─"*12} {"─"*12} ┤', end='')
        for _ in models:
            print(f' {"─"*12} {"─"*5} {"─"*3} ┤', end='')
        print()

        stats = {name: {'fixed': 0, 'kept': 0} for name, _ in models}
        n_corrupt = 0
        n_clean = 0

        for i in range(len(ro)):
            r0 = all_results['Orig'][i]
            if r0['is_corrupted']:
                n_corrupt += 1
                gold = r0['gold'].replace('\n', '\\n')
                corrupt = r0['corrupt'].replace('\n', '\\n')

                print(f'  {r0["pos"]:>3} {gold:>12} {R}→{corrupt:>11}{X} │', end='')
                for name, _ in models:
                    r = all_results[name][i]
                    p = r['pred'].replace('\n', '\\n')
                    status = f'{G}✓fix{X}' if r['fixed'] else f'{R}✗{X}   '
                    if r['fixed']:
                        stats[name]['fixed'] += 1
                    print(f' {p:>12} {r["confidence"]:>5.2f} {status} │', end='')
                print()
            else:
                n_clean += 1
                for name, _ in models:
                    r = all_results[name][i]
                    if r['kept_clean']:
                        stats[name]['kept'] += 1

        print(f'  {"─"*90}')
        print(f'  Corrupted→Fixed:', end='')
        for name, _ in models:
            f = stats[name]['fixed']
            print(f'  {name} {f}/{n_corrupt} ({f/n_corrupt:.0%})', end=' │')
        print()
        print(f'  Clean preserved: ', end='')
        for name, _ in models:
            k = stats[name]['kept']
            print(f'  {name} {k}/{n_clean} ({k/n_clean:.0%})', end=' │')
        print()
