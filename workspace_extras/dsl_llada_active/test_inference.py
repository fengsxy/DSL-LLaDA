"""Test remasking inference on DSL-trained LLaDA checkpoint.

Loads the checkpoint, runs standard LLaDA remasking generation (no DSL noise),
and compares output with the original model.

Usage:
    python dsl_llada/test_inference.py [checkpoint_dir]
"""
import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from generate import generate


def load_model(path, device='cuda:0'):
    """Load LLaDA model (original or DSL-trained checkpoint)."""
    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    return model


def test_remasking(model, tokenizer, device='cuda:0'):
    """Test standard LLaDA remasking generation."""
    prompts = [
        "What is 2 + 3?",
        "The capital of France is",
        "Explain briefly what machine learning is.",
    ]

    messages = [{"role": "user", "content": p} for p in prompts]
    formatted = [tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False) for m in messages]

    results = []
    for i, prompt_text in enumerate(formatted):
        encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        out = generate(
            model, input_ids, attention_mask,
            steps=64, gen_length=128, block_length=128,
            temperature=0., cfg_scale=0., remasking='low_confidence'
        )
        decoded = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        results.append((prompts[i], decoded))
        print(f"\n--- Prompt {i+1}: {prompts[i]}")
        print(f"    Output: {decoded[:200]}")

    return results


def test_infilling(model, tokenizer, device='cuda:0'):
    """Test infilling: mask some tokens and check if model can recover them."""
    text = "The quick brown fox jumps over the lazy dog"
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    L = input_ids.shape[1]

    mask_id = 126336
    # Mask 50% of tokens randomly
    mask_positions = torch.rand(L) < 0.5
    masked_ids = input_ids.clone()
    masked_ids[0, mask_positions] = mask_id

    with torch.no_grad():
        logits = model(masked_ids).logits
    preds = logits.argmax(dim=-1)

    # Check recovery accuracy on masked positions
    correct = (preds[0, mask_positions] == input_ids[0, mask_positions]).float().mean().item()
    n_masked = mask_positions.sum().item()

    print(f"\n--- Infilling test ---")
    print(f"    Original: {text}")
    print(f"    Masked {n_masked}/{L} tokens")
    print(f"    Recovery accuracy: {correct:.2%}")

    recovered = tokenizer.decode(preds[0], skip_special_tokens=True)
    print(f"    Recovered: {recovered[:200]}")

    return correct


if __name__ == '__main__':
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'GSAI-ML/LLaDA-8B-Instruct'
    device = 'cuda:0'

    print(f"Loading model from: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    tokenizer.padding_side = 'left'

    model = load_model(checkpoint, device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    print("\n=== Remasking Generation ===")
    test_remasking(model, tokenizer, device)

    print("\n=== Infilling Test ===")
    test_infilling(model, tokenizer, device)
