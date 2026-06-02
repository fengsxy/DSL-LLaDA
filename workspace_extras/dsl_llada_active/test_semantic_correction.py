"""Test semantic error correction: can the model fix semantically wrong but plausible tokens?
E.g. "The capital of France is London" → should correct to "Paris"

Usage: CUDA_VISIBLE_DEVICES=0,1 python test_semantic_correction.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MASK_ID = 126336
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

CASES = [
    # (corrupted_text, gold_word, corrupt_word, category)
    # Geography
    ('The capital of France is London.', 'Paris', 'London', 'geo'),
    ('The capital of Japan is Beijing.', 'Tokyo', 'Beijing', 'geo'),
    ('The capital of Italy is Madrid.', 'Rome', 'Madrid', 'geo'),
    ('The capital of Germany is Vienna.', 'Berlin', 'Vienna', 'geo'),
    ('The capital of China is Seoul.', 'Beijing', 'Seoul', 'geo'),
    ('The largest ocean is the Atlantic Ocean.', 'Pacific', 'Atlantic', 'geo'),
    ('Mount Everest is located in Africa.', 'Asia', 'Africa', 'geo'),

    # Science
    ('Water is made of hydrogen and nitrogen.', 'oxygen', 'nitrogen', 'sci'),
    ('Einstein developed the theory of evolution.', 'relativity', 'evolution', 'sci'),
    ('The chemical symbol for gold is Ag.', 'Au', 'Ag', 'sci'),
    ('Light travels faster than electricity.', 'sound', 'electricity', 'sci'),
    ('The Earth revolves around Mars.', 'Sun', 'Mars', 'sci'),

    # Math
    ('Two plus two equals five.', 'four', 'five', 'math'),
    ('A triangle has four sides.', 'three', 'four', 'math'),
    ('A square has five corners.', 'four', 'five', 'math'),

    # Common knowledge
    ('The sun rises in the west.', 'east', 'west', 'common'),
    ('Shakespeare wrote War and Peace.', 'Tolstoy', 'Shakespeare', 'common'),
    ('Neil Armstrong was the first person to walk on Mars.', 'Moon', 'Mars', 'common'),
    ('The Great Wall is located in Japan.', 'China', 'Japan', 'common'),
    ('Christmas is celebrated on December 31st.', '25th', '31st', 'common'),
]


def load_model(path, device):
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map=device)
    model.eval()
    return model


def test_correction(model, device, text, gold_word, corrupt_word):
    ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    # Try with and without leading space
    corrupt_variants = [corrupt_word, ' ' + corrupt_word, corrupt_word.lower(), ' ' + corrupt_word.lower()]
    gold_variants = [gold_word, ' ' + gold_word, gold_word.lower(), ' ' + gold_word.lower()]

    ids_list = ids[0].tolist()
    pos = -1
    corrupt_ids = None
    gold_ids = None

    for cv, gv in zip(corrupt_variants, gold_variants):
        cids = tokenizer(cv, add_special_tokens=False)['input_ids']
        for i in range(len(ids_list) - len(cids) + 1):
            if ids_list[i:i + len(cids)] == cids:
                pos = i
                corrupt_ids = cids
                gold_ids = tokenizer(gv, add_special_tokens=False)['input_ids']
                break
        if pos != -1:
            break

    if pos == -1:
        return None, None, None

    # Replace corrupt with MASK
    masked = ids.clone()
    for j in range(len(corrupt_ids)):
        masked[0, pos + j] = MASK_ID

    with torch.no_grad():
        logits = model(masked).logits

    pred_ids = logits[0, pos:pos + len(corrupt_ids)].argmax(dim=-1).tolist()
    pred_word = tokenizer.decode(pred_ids).strip()
    gold_word_decoded = tokenizer.decode(gold_ids).strip()

    # Also get top-3 for first position
    top3v, top3i = logits[0, pos].topk(3)
    top3_words = [tokenizer.decode([i.item()]).strip() for i in top3i]

    correct = (pred_word.lower() == gold_word_decoded.lower())
    return correct, pred_word, top3_words


def main():
    print("Loading Original LLaDA on cuda:0...")
    model_orig = load_model('GSAI-ML/LLaDA-8B-Instruct', 'cuda:0')

    print("Loading Semantic β=2 on cuda:1...")
    model_sem = load_model('./checkpoints/semantic_b05_d100_10k/checkpoint-4000', 'cuda:1')

    print()
    print(f"{'Corrupted text':<50s} | {'Gold':>8s} | {'Orig pred':>10s} | {'Sem pred':>10s} | Orig top3 / Sem top3")
    print("=" * 140)

    orig_ok = sem_ok = total = 0

    for text, gold, corrupt, cat in CASES:
        r_o, p_o, t3_o = test_correction(model_orig, 'cuda:0', text, gold, corrupt)
        r_s, p_s, t3_s = test_correction(model_sem, 'cuda:1', text, gold, corrupt)

        if r_o is None:
            print(f"  SKIP: can't find '{corrupt}' in tokenized text")
            continue

        total += 1
        if r_o: orig_ok += 1
        if r_s: sem_ok += 1

        mo = '✓' if r_o else '✗'
        ms = '✓' if r_s else '✗'
        short = text[:48] + '..' if len(text) > 48 else text
        t3o_str = ','.join(t3_o) if t3_o else ''
        t3s_str = ','.join(t3_s) if t3_s else ''
        print(f"{short:<50s} | {gold:>8s} | {mo} {p_o:>8s} | {ms} {p_s:>8s} | {t3o_str} / {t3s_str}")

    print("=" * 140)
    print(f"Original: {orig_ok}/{total} ({orig_ok/total:.0%})    Semantic β=2: {sem_ok}/{total} ({sem_ok/total:.0%})")


if __name__ == '__main__':
    main()
