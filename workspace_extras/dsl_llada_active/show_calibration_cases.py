"""Show detailed calibration cases: input → masked → model prediction.

Compares Original LLaDA vs Softmasker (ckpt-1000) on the same masked positions.

Usage:
    python dsl_llada/show_calibration_cases.py [gpu_id]
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336
ORIGINAL = 'GSAI-ML/LLaDA-8B-Instruct'
SOFTMASKER = './checkpoints/dsl_1000step/checkpoint-1000'

# Same texts used in calibration test
TEXTS = [
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the park.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "To make a chocolate cake, you need flour, sugar, cocoa powder, eggs, butter, and milk. Preheat the oven to 350 degrees.",
    "The capital of France is Paris. It is known for the Eiffel Tower, the Louvre Museum, and its rich cultural heritage.",
    "In mathematics, the Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
]

R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'; B = '\033[1m'; D = '\033[2m'; X = '\033[0m'
BG_R = '\033[41m'; BG_G = '\033[42m'; BG_Y = '\033[43m'


def predict_masked(model, tokenizer, text, mask_ratio=0.5, seed=42):
    """Mask tokens and get predictions with confidence."""
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    input_ids = tokens['input_ids'].cuda()
    L = input_ids.shape[1]

    torch.manual_seed(seed)
    mask = torch.rand(L, device='cuda') < mask_ratio
    if mask.all():
        mask[0] = False

    masked_ids = input_ids.clone()
    masked_ids[0, mask] = MASK_ID

    with torch.no_grad():
        logits = model(masked_ids).logits[0].float()
        probs = F.softmax(logits, dim=-1)

    results = []
    for i in range(L):
        tok_str = tokenizer.decode([input_ids[0, i].item()])
        is_masked = mask[i].item()
        pred_prob, pred_id = probs[i].max(dim=-1)
        pred_str = tokenizer.decode([pred_id.item()])
        correct = (pred_id.item() == input_ids[0, i].item())
        # Also get top-3
        top3_probs, top3_ids = probs[i].topk(3)
        top3 = [(tokenizer.decode([top3_ids[j].item()]), top3_probs[j].item()) for j in range(3)]
        results.append({
            'pos': i,
            'token': tok_str,
            'masked': is_masked,
            'pred': pred_str,
            'pred_id': pred_id.item(),
            'gold_id': input_ids[0, i].item(),
            'confidence': pred_prob.item(),
            'correct': correct,
            'top3': top3,
        })
    return results


def show_case(text, results_orig, results_sm, mask_ratio):
    """Print side-by-side comparison for masked positions."""
    print(f'\n{B}Text:{X} "{text[:80]}..."')
    print(f'{D}mask_ratio={mask_ratio}{X}\n')

    print(f'  {"Pos":>3} {"Token":>12} {"Masked":>6} │ {"Original pred":>14} {"conf":>6} {"✓":>2} │ {"Softmasker pred":>16} {"conf":>6} {"✓":>2}')
    print(f'  {"─"*3} {"─"*12} {"─"*6} ┤ {"─"*14} {"─"*6} {"─"*2} ┤ {"─"*16} {"─"*6} {"─"*2}')

    n_masked = 0
    orig_correct = 0
    sm_correct = 0
    orig_confs = []
    sm_confs = []

    for ro, rs in zip(results_orig, results_sm):
        if not ro['masked']:
            continue
        n_masked += 1

        tok = ro['token'].replace('\n', '\\n')
        # Original
        op = ro['pred'].replace('\n', '\\n')
        oc = ro['confidence']
        o_ok = '✓' if ro['correct'] else '✗'
        o_color = G if ro['correct'] else R
        # Softmasker
        sp = rs['pred'].replace('\n', '\\n')
        sc = rs['confidence']
        s_ok = '✓' if rs['correct'] else '✗'
        s_color = G if rs['correct'] else R

        orig_correct += ro['correct']
        sm_correct += rs['correct']
        orig_confs.append(oc)
        sm_confs.append(sc)

        # Highlight interesting cases: one right one wrong, or confidence differs
        highlight = ''
        if ro['correct'] != rs['correct']:
            highlight = '  ←' + (Y + ' SM wins' + X if rs['correct'] else Y + ' Orig wins' + X)

        print(f'  {ro["pos"]:>3} {tok:>12} {"[MASK]":>6} │ {o_color}{op:>14}{X} {oc:>6.3f} {o_color}{o_ok:>2}{X} │ {s_color}{sp:>16}{X} {sc:>6.3f} {s_color}{s_ok:>2}{X}{highlight}')

    print(f'  {"─"*80}')
    print(f'  Masked: {n_masked} tokens')
    print(f'  Accuracy:    Original {orig_correct}/{n_masked} ({orig_correct/n_masked:.1%})  │  Softmasker {sm_correct}/{n_masked} ({sm_correct/n_masked:.1%})')
    print(f'  Avg conf:    Original {sum(orig_confs)/len(orig_confs):.3f}  │  Softmasker {sum(sm_confs)/len(sm_confs):.3f}')


# Load models
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)

print(f'Loading Original LLaDA...')
model_orig = AutoModel.from_pretrained(ORIGINAL, trust_remote_code=True,
                                        dtype=torch.bfloat16).cuda().eval()

print(f'Loading Softmasker (ckpt-1000)...')
model_sm = AutoModel.from_pretrained(SOFTMASKER, trust_remote_code=True,
                                      dtype=torch.bfloat16).cuda().eval()

print(f'\n{"="*80}')
print(f' Calibration Case Study: Original LLaDA vs Softmasker (ckpt-1000)')
print(f' Task: Fill in [MASK] tokens (standard LLaDA inference, no DSL)')
print(f'{"="*80}')

for mask_ratio in [0.3, 0.5]:
    print(f'\n{"#"*80}')
    print(f'  MASK RATIO = {mask_ratio}')
    print(f'{"#"*80}')

    for text in TEXTS:
        results_orig = predict_masked(model_orig, tokenizer, text, mask_ratio=mask_ratio)
        results_sm = predict_masked(model_sm, tokenizer, text, mask_ratio=mask_ratio)
        show_case(text, results_orig, results_sm, mask_ratio)
