"""Can DSL noise + converter shift semantic errors toward correct answers?

If embedding space captures semantics, then:
  noisy("England") at moderate SNR → converter uncertain → backbone uses context → "Paris"?

Test: feed noise_embed + noise through converter, check if backbone prediction shifts.
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '0'

from transformers import AutoTokenizer, AutoModel
from dsl_llada.dsl_modules import attach_dsl_modules, noisy_embedding
import safetensors.torch

ORIGINAL = 'GSAI-ML/LLaDA-8B-Instruct'
PCA_CKPT = './checkpoints/dsl_pca128_20260312_171035/checkpoint-1000'
RAND_CKPT = './checkpoints/dsl_1000step/checkpoint-1000'

tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)

R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'; B = '\033[1m'; D = '\033[2m'; X = '\033[0m'

def load_dsl_model(ckpt_path, noise_dim, noise_init):
    model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    attach_dsl_modules(model, noise_dim=noise_dim, freeze_ff_out=True, noise_init=noise_init)

    dsl_keys = {'converter.backbone_embedding.weight', 'converter.backbone_embedding.bias',
                'converter.beta', 'converter.embed.weight', 'converter.logit_bias', 'noise_embed.weight'}
    for cf in os.listdir(ckpt_path):
        if cf.endswith('.safetensors'):
            state = safetensors.torch.load_file(os.path.join(ckpt_path, cf))
            for k, v in state.items():
                if k in dsl_keys:
                    parts = k.split('.')
                    obj = model
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    param = getattr(obj, parts[-1])
                    if isinstance(param, torch.nn.Parameter):
                        param.data.copy_(v.to(param.device))
                    else:
                        setattr(obj, parts[-1], v.to(param.device))
    model.converter.cuda()
    model.noise_embed.cuda()
    return model


cases = [
    ('One plus one equals three.', 'two', 'three'),
    ('The capital of France is England.', 'Paris', 'England'),
    ('Water boils at 50 degrees Celsius.', '100', '50'),
    ('The Earth orbits around Mars.', 'the Sun', 'Mars'),
]


@torch.no_grad()
def test_with_dsl_noise(model, text, snr_values, noise_dim, label):
    """Feed text through noise_embed + noise + converter + backbone."""
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
    B, L = ids.shape

    # First: normal inference (no DSL)
    logits_clean = model(ids).logits[0].float()
    pred_clean = logits_clean.argmax(dim=-1)

    print(f'  {label}:')

    # Show clean prediction
    tokens_str = []
    for i in range(L):
        t = tokenizer.decode([ids[0, i].item()])
        p = tokenizer.decode([pred_clean[i].item()])
        if pred_clean[i].item() != ids[0, i].item():
            tokens_str.append(f'{Y}{t}→{p}{X}')
        else:
            tokens_str.append(t)
    print(f'    {"No noise":>12}: {" ".join(tokens_str)}')

    # With DSL noise at various SNR
    for snr_val in snr_values:
        snr = torch.full((B, L), float(snr_val), device='cuda', dtype=torch.float32)

        # Average over multiple noise samples for stability
        n_samples = 5
        pred_counts = torch.zeros(L, model.config.vocab_size, device='cuda')

        for _ in range(n_samples):
            z_noisy = noisy_embedding(model.noise_embed, ids, snr)
            # Through converter
            h = model.converter(z_noisy)
            # Through backbone
            wte_dtype = model.model.transformer.wte.weight.dtype
            logits = model(input_ids=ids, inputs_embeds=h.to(dtype=wte_dtype)).logits[0].float()
            pred_counts.scatter_add_(1, logits.argmax(dim=-1).unsqueeze(1),
                                      torch.ones(L, 1, device='cuda'))

        # Most common prediction across samples
        pred_ids = pred_counts.argmax(dim=-1)

        # Also get the converter's top prediction to see if it's confused
        z_noisy_single = noisy_embedding(model.noise_embed, ids, snr)
        converter_probs = model.converter.get_token_probs(z_noisy_single)[0]  # (L, V+1)
        conv_top = converter_probs.argmax(dim=-1)
        conv_conf = converter_probs.max(dim=-1).values

        tokens_str = []
        for i in range(L):
            t = tokenizer.decode([ids[0, i].item()])
            p = tokenizer.decode([pred_ids[i].item()])
            cc = conv_conf[i].item()
            ct = tokenizer.decode([conv_top[i].item()]) if conv_top[i].item() < 126336 else '[M]'
            correct_conv = conv_top[i].item() == ids[0, i].item()

            if pred_ids[i].item() != ids[0, i].item():
                tokens_str.append(f'{Y}{t}→{p}{X}')
            else:
                tokens_str.append(t)

        # Converter accuracy for this SNR
        conv_acc = (conv_top == ids[0]).float().mean().item()
        print(f'    {"SNR="+str(snr_val):>12}: {" ".join(tokens_str)}  {D}(conv_acc={conv_acc:.0%}){X}')


# Load models
print('Loading Rand48 with DSL modules...')
model_rand = load_dsl_model(RAND_CKPT, noise_dim=48, noise_init='random')
print('Loading PCA128 with DSL modules...')
model_pca = load_dsl_model(PCA_CKPT, noise_dim=128, noise_init='pca')

snr_values = [50, 20, 10, 5, 3, 1]

print(f'\n{"="*90}')
print(f'  Semantic Error + DSL Noise: Does noise help the model "rethink"?')
print(f'{"="*90}')

for wrong, correct_word, error_word in cases:
    print(f'\n{B}"{wrong}"{X}  (should be "{G}{correct_word}{X}" not "{R}{error_word}{X}")\n')

    test_with_dsl_noise(model_rand, wrong, snr_values, 48, 'Rand48')
    print()
    test_with_dsl_noise(model_pca, wrong, snr_values, 128, 'PCA128')
    print(f'\n{"─"*90}')
