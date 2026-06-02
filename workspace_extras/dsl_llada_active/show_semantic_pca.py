"""Test semantic error correction: Original vs Rand48 vs PCA128."""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from transformers import AutoTokenizer, AutoModel

ORIGINAL = 'GSAI-ML/LLaDA-8B-Instruct'
SOFTMASKER = './checkpoints/dsl_1000step/checkpoint-1000'
PCA128 = './checkpoints/dsl_pca128_20260312_171035/checkpoint-1000'

tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)
print('Loading Original LLaDA...')
model_orig = AutoModel.from_pretrained(ORIGINAL, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
print('Loading Rand48...')
model_sm = AutoModel.from_pretrained(SOFTMASKER, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
print('Loading PCA128...')
model_pca = AutoModel.from_pretrained(PCA128, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'; B = '\033[1m'; D = '\033[2m'; X = '\033[0m'

cases = [
    ('One plus one equals three.', 'One plus one equals two.', '1+1=3'),
    ('The capital of France is England.', 'The capital of France is Paris.', 'France→England'),
    ('2 + 2 = 5', '2 + 2 = 4', '2+2=5'),
    ('Water boils at 50 degrees Celsius.', 'Water boils at 100 degrees Celsius.', '50→100'),
    ('The Earth orbits around Mars.', 'The Earth orbits around the Sun.', 'Mars→Sun'),
]

models = [('Orig', model_orig), ('Rand48', model_sm), ('PCA128', model_pca)]

@torch.no_grad()
def predict(model, text):
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
    logits = model(ids).logits[0].float()
    probs = F.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)
    pred_conf = probs.max(dim=-1).values
    return [(tokenizer.decode([ids[0,i].item()]),
             tokenizer.decode([pred_ids[i].item()]),
             pred_conf[i].item(),
             pred_ids[i].item() != ids[0,i].item())
            for i in range(ids.shape[1])]

print(f'\n{"="*90}')
print(f'  Semantic Error Correction: Original vs Rand48 vs PCA128')
print(f'{"="*90}\n')

for wrong, right, label in cases:
    print(f'{B}{label}{X}: "{Y}{wrong}{X}" (correct: "{G}{right}{X}")')

    for name, model in models:
        results = predict(model, wrong)
        changes = [(inp, pred, conf) for inp, pred, conf, changed in results if changed]
        if changes:
            desc = ', '.join(f'"{i}"→"{p}" ({c:.2f})' for i, p, c in changes)
            print(f'  {name:>8}: {Y}changes{X} {desc}')
        else:
            # Find the error token confidence
            wrong_toks = tokenizer(wrong, return_tensors='pt', add_special_tokens=False)['input_ids']
            right_toks = tokenizer(right, return_tensors='pt', add_special_tokens=False)['input_ids']
            # Find differing positions
            diffs = []
            for inp, pred, conf, _ in results:
                diffs.append(f'{inp}({conf:.2f})')
            # Just show the key error token confidence
            error_confs = []
            for inp, pred, conf, _ in results:
                error_confs.append((inp.strip(), conf))
            key = error_confs[-2] if len(error_confs) > 1 else error_confs[-1]  # usually the error is near end
            print(f'  {name:>8}: {R}keeps error{X} — all tokens unchanged (key: "{key[0]}" conf={key[1]:.3f})')
    print()
