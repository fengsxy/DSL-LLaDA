"""Show detailed corruption correction cases.

Input: real text → replace 20% tokens with random wrong ones → model predicts
Compare: Original LLaDA vs Softmasker on the SAME corrupted positions.

Usage:
    python dsl_llada/show_corruption_cases.py [gpu_id]
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336
ORIGINAL = 'GSAI-ML/LLaDA-8B-Instruct'
SOFTMASKER = './checkpoints/dsl_1000step/checkpoint-1000'

TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower, the Louvre Museum, and its rich cultural heritage.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure. Ice melts at 0 degrees Celsius.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the park.",
]

R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'; B = '\033[1m'; D = '\033[2m'; X = '\033[0m'
BG_R = '\033[41m'; BG_G = '\033[42m'


def corrupt_and_predict(model, tokenizer, text, rate, seed=42):
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    input_ids = tokens['input_ids'].cuda()
    L = input_ids.shape[1]

    torch.manual_seed(seed)
    mask = torch.rand(L, device='cuda') < rate
    # Random wrong tokens
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
            'pos': i,
            'gold': tokenizer.decode([gold]),
            'gold_id': gold,
            'corrupt': tokenizer.decode([corrupt]),
            'corrupt_id': corrupt,
            'is_corrupted': mask[i].item(),
            'pred': tokenizer.decode([pred]),
            'pred_id': pred,
            'confidence': conf,
            'fixed': mask[i].item() and (pred == gold),
            'kept_clean': (not mask[i].item()) and (pred == gold),
            'broke_clean': (not mask[i].item()) and (pred != gold),
        })
    return results


tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)
print('Loading Original LLaDA...')
model_orig = AutoModel.from_pretrained(ORIGINAL, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
print('Loading Softmasker (ckpt-1000)...')
model_sm = AutoModel.from_pretrained(SOFTMASKER, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

for rate in [0.2, 0.5]:
    print(f'\n{"#"*80}')
    print(f'  CORRUPTION RATE = {rate:.0%} (random wrong tokens, NOT mask)')
    print(f'{"#"*80}')

    for text in TEXTS:
        print(f'\n{B}Original:{X} "{text[:80]}..."')

        ro = corrupt_and_predict(model_orig, tokenizer, text, rate)
        rs = corrupt_and_predict(model_sm, tokenizer, text, rate)

        # Show corrupted input
        corrupted_str = ''
        for r in ro:
            if r['is_corrupted']:
                corrupted_str += f'{R}{r["corrupt"]}{X}'
            else:
                corrupted_str += r['gold']
        print(f'{D}Corrupted:{X} {corrupted_str[:120]}...')
        print()

        # Show only corrupted positions (interesting ones)
        print(f'  {"Pos":>3} {"Gold":>12} {"→Corrupt":>12} │ {"Orig pred":>12} {"conf":>5} {"":>3} │ {"SM pred":>12} {"conf":>5} {"":>3}')
        print(f'  {"─"*3} {"─"*12} {"─"*12} ┤ {"─"*12} {"─"*5} {"─"*3} ┤ {"─"*12} {"─"*5} {"─"*3}')

        orig_fixed = 0
        sm_fixed = 0
        orig_kept = 0
        sm_kept = 0
        n_corrupt = 0
        n_clean = 0

        for o, s in zip(ro, rs):
            if o['is_corrupted']:
                n_corrupt += 1
                gold = o['gold'].replace('\n', '\\n')
                corrupt = o['corrupt'].replace('\n', '\\n')
                op = o['pred'].replace('\n', '\\n')
                sp = s['pred'].replace('\n', '\\n')

                o_status = f'{G}✓fix{X}' if o['fixed'] else f'{R}✗{X}   '
                s_status = f'{G}✓fix{X}' if s['fixed'] else f'{R}✗{X}   '
                if o['fixed']:
                    orig_fixed += 1
                if s['fixed']:
                    sm_fixed += 1

                highlight = ''
                if o['fixed'] != s['fixed']:
                    highlight = '  ←' + (Y + 'SM' + X if s['fixed'] else Y + 'Orig' + X)

                print(f'  {o["pos"]:>3} {gold:>12} {R}→{corrupt:>11}{X} │ {op:>12} {o["confidence"]:>5.2f} {o_status} │ {sp:>12} {s["confidence"]:>5.2f} {s_status}{highlight}')
            else:
                n_clean += 1
                if o['kept_clean']:
                    orig_kept += 1
                if s['kept_clean']:
                    sm_kept += 1

        print(f'  {"─"*75}')
        print(f'  Corrupted→Fixed: Original {orig_fixed}/{n_corrupt} ({orig_fixed/n_corrupt:.0%}) │ Softmasker {sm_fixed}/{n_corrupt} ({sm_fixed/n_corrupt:.0%})')
        print(f'  Clean preserved:  Original {orig_kept}/{n_clean} ({orig_kept/n_clean:.0%}) │ Softmasker {sm_kept}/{n_clean} ({sm_kept/n_clean:.0%})')
