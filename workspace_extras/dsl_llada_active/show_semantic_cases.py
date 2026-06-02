"""Test semantic error correction: wrong but plausible tokens.

Unlike random corruption, these are meaningful errors (math/factual).
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from transformers import AutoTokenizer, AutoModel

ORIGINAL = 'GSAI-ML/LLaDA-8B-Instruct'
SOFTMASKER = './checkpoints/dsl_1000step/checkpoint-1000'

tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)

print('Loading Original LLaDA...')
model_orig = AutoModel.from_pretrained(ORIGINAL, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
print('Loading Softmasker (ckpt-1000)...')
model_sm = AutoModel.from_pretrained(SOFTMASKER, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'; B = '\033[1m'; D = '\033[2m'; X = '\033[0m'

cases = [
    {
        'wrong': 'One plus one equals three.',
        'right': 'One plus one equals two.',
        'label': 'Math error: 1+1=3',
    },
    {
        'wrong': 'The capital of France is England.',
        'right': 'The capital of France is Paris.',
        'label': 'Factual error: France→England',
    },
    {
        'wrong': '2 + 2 = 5',
        'right': '2 + 2 = 4',
        'label': 'Math error: 2+2=5',
    },
    {
        'wrong': 'Water boils at 50 degrees Celsius.',
        'right': 'Water boils at 100 degrees Celsius.',
        'label': 'Factual error: 50→100',
    },
    {
        'wrong': 'The Earth orbits around Mars.',
        'right': 'The Earth orbits around the Sun.',
        'label': 'Factual error: Mars→Sun',
    },
]

@torch.no_grad()
def predict(model, text):
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
    logits = model(ids).logits[0].float()
    probs = F.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)
    pred_conf = probs.max(dim=-1).values
    tokens = []
    for i in range(ids.shape[1]):
        tokens.append({
            'input': tokenizer.decode([ids[0, i].item()]),
            'input_id': ids[0, i].item(),
            'pred': tokenizer.decode([pred_ids[i].item()]),
            'pred_id': pred_ids[i].item(),
            'conf': pred_conf[i].item(),
            'changed': pred_ids[i].item() != ids[0, i].item(),
        })
    return tokens

print(f'\n{"="*80}')
print(f'  Semantic Error Correction: Can the model fix meaningful errors?')
print(f'{"="*80}\n')

for case in cases:
    print(f'{B}{case["label"]}{X}')
    print(f'  Wrong input: "{case["wrong"]}"')
    print(f'  Correct:     "{case["right"]}"')
    print()

    ro = predict(model_orig, case['wrong'])
    rs = predict(model_sm, case['wrong'])

    # Also predict the correct version to see if models agree
    ro_right = predict(model_orig, case['right'])
    rs_right = predict(model_sm, case['right'])

    print(f'  {"Pos":>3} {"Input token":>14} │ {"Orig pred":>14} {"conf":>5} {"chg":>3} │ {"SM pred":>14} {"conf":>5} {"chg":>3}')
    print(f'  {"─"*3} {"─"*14} ┤ {"─"*14} {"─"*5} {"─"*3} ┤ {"─"*14} {"─"*5} {"─"*3}')

    for i, (o, s) in enumerate(zip(ro, rs)):
        inp = o['input'].replace('\n', '\\n')
        op = o['pred'].replace('\n', '\\n')
        sp = s['pred'].replace('\n', '\\n')
        o_mark = f'{Y}→{X}' if o['changed'] else '  '
        s_mark = f'{Y}→{X}' if s['changed'] else '  '
        o_color = Y if o['changed'] else ''
        s_color = Y if s['changed'] else ''
        o_end = X if o['changed'] else ''
        s_end = X if s['changed'] else ''
        print(f'  {i:>3} {inp:>14} │ {o_color}{op:>14}{o_end} {o["conf"]:>5.3f} {o_mark} │ {s_color}{sp:>14}{s_end} {s["conf"]:>5.3f} {s_mark}')

    print()
    # Summary
    orig_changes = [o for o in ro if o['changed']]
    sm_changes = [s for s in rs if s['changed']]
    print(f'  Original LLaDA changes: {len(orig_changes)} tokens', end='')
    if orig_changes:
        print(f' — ' + ', '.join(f'"{c["input"]}"→"{c["pred"]}"' for c in orig_changes))
    else:
        print(f' — {R}keeps error as-is{X}')
    print(f'  Softmasker changes:     {len(sm_changes)} tokens', end='')
    if sm_changes:
        print(f' — ' + ', '.join(f'"{c["input"]}"→"{c["pred"]}"' for c in sm_changes))
    else:
        print(f' — {R}keeps error as-is{X}')
    print(f'\n{"─"*80}\n')
