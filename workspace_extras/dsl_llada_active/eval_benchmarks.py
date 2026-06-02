"""Evaluate LLaDA on MBPP and ARC-Challenge benchmarks.

Uses remasking-based generation (not autoregressive).

Usage:
    python dsl_llada/eval_benchmarks.py \
        --checkpoint GSAI-ML/LLaDA-8B-Instruct \
        --dataset mbpp --steps 64 --gpu 0 \
        --output results/mbpp_llada_64step.json

    python dsl_llada/eval_benchmarks.py \
        --checkpoint GSAI-ML/LLaDA-8B-Instruct \
        --dataset arc --steps 64 --gpu 0 \
        --output results/arc_llada_64step.json
"""
import argparse
import ast
import json
import os
import re
import sys
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from generate import generate

MASK_ID = 126336


def load_model(path, device):
    """Load LLaDA model (original or finetuned checkpoint)."""
    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    return model.to(device).eval()


def load_mbpp():
    """Load MBPP test set."""
    from datasets import load_dataset
    ds = load_dataset('google-research-datasets/mbpp', split='test')
    problems = []
    for item in ds:
        problems.append({
            'text': item['text'],
            'code': item['code'],
            'test_list': item['test_list'],
        })
    return problems


def load_arc():
    """Load ARC-Challenge test set."""
    from datasets import load_dataset
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    problems = []
    for item in ds:
        choices_text = item['choices']['text']
        choices_label = item['choices']['label']
        problems.append({
            'question': item['question'],
            'choices_text': choices_text,
            'choices_label': choices_label,
            'answer_key': item['answerKey'],
        })
    return problems


def format_mbpp_prompt(problem):
    """Format MBPP problem as a prompt."""
    prompt = (
        f"Write a Python function to solve the following problem.\n\n"
        f"Problem: {problem['text']}\n\n"
        f"Test cases:\n"
    )
    for tc in problem['test_list'][:3]:
        prompt += f"  {tc}\n"
    prompt += (
        f"\nWrite ONLY the function definition. "
        f"Put your code inside a ```python``` code block.\n\n"
        f"Answer:"
    )
    return prompt


def format_arc_prompt(problem):
    """Format ARC-Challenge problem as a multiple-choice prompt."""
    prompt = f"Question: {problem['question']}\n\nChoices:\n"
    for label, text in zip(problem['choices_label'], problem['choices_text']):
        prompt += f"  {label}. {text}\n"
    prompt += (
        f"\nChoose the correct answer. "
        f"Reply with ONLY the letter (A, B, C, or D) of the correct answer.\n\n"
        f"Answer:"
    )
    return prompt


def extract_code_block(text):
    """Extract Python code from model output.

    Tries ```python ... ``` blocks first, then ``` ... ```,
    then falls back to looking for lines starting with 'def '.
    """
    # Try ```python ... ```
    match = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: extract from first 'def ' to end (or next blank line after block)
    match = re.search(r'(def\s+\w+.*)', text, re.DOTALL)
    if match:
        code = match.group(1)
        # Try to find end of function (heuristic: stop at double newline
        # or when indentation returns to zero after the def)
        lines = code.split('\n')
        func_lines = [lines[0]]
        for line in lines[1:]:
            if line.strip() == '' and len(func_lines) > 1:
                # Keep one blank line, but stop at second
                func_lines.append(line)
                continue
            if line and not line[0].isspace() and not line.startswith('def'):
                break
            func_lines.append(line)
        return '\n'.join(func_lines).strip()

    return text.strip()


def check_syntax_valid(code):
    """Check if code is syntactically valid Python using ast.parse()."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_has_function(code):
    """Check if code contains at least one function definition."""
    return bool(re.search(r'^\s*def\s+\w+', code, re.MULTILINE))


def extract_answer_letter(text):
    """Extract answer letter (A/B/C/D/1/2/3/4) from model output."""
    text = text.strip()

    # Direct single letter answer
    if len(text) == 1 and text.upper() in 'ABCDE12345':
        return text.upper()

    # "The answer is X" pattern
    match = re.search(r'(?:answer|correct)\s*(?:is|:)\s*\(?([A-Ea-e1-5])\)?', text, re.I)
    if match:
        return match.group(1).upper()

    # Letter followed by period or parenthesis at start
    match = re.match(r'\(?([A-Ea-e])\)?[\.\s\)]', text)
    if match:
        return match.group(1).upper()

    # First occurrence of a standalone letter A-E
    match = re.search(r'\b([A-Ea-e])\b', text)
    if match:
        return match.group(1).upper()

    # Last resort: first letter character
    for ch in text:
        if ch.upper() in 'ABCDE':
            return ch.upper()

    return ''


def generate_response(model, tokenizer, prompt_text, steps, device,
                      gen_length=512, block_length=None):
    """Generate a response using LLaDA remasking-based generation."""
    if block_length is None:
        block_length = gen_length

    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)

    with torch.no_grad():
        attention_mask = torch.ones_like(input_ids)
        out = generate(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0., cfg_scale=0., remasking='low_confidence')

    generated = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return generated


def evaluate_mbpp(model, tokenizer, problems, steps, device,
                  max_problems=None, gen_length=512, block_length=None):
    """Evaluate on MBPP. Reports syntax-valid rate as proxy for pass@1."""
    if max_problems:
        problems = problems[:max_problems]

    results = []
    syntax_valid = 0
    has_func = 0
    total = 0

    for i, prob in enumerate(tqdm(problems, desc=f'MBPP {steps}-step')):
        try:
            prompt = format_mbpp_prompt(prob)
            output = generate_response(
                model, tokenizer, prompt, steps, device,
                gen_length=gen_length, block_length=block_length)

            code = extract_code_block(output)
            is_syntax_valid = check_syntax_valid(code)
            is_has_func = check_has_function(code)

            syntax_valid += int(is_syntax_valid)
            has_func += int(is_has_func)
            total += 1

            results.append({
                'idx': i,
                'problem': prob['text'][:200],
                'extracted_code': code[:500],
                'syntax_valid': is_syntax_valid,
                'has_function': is_has_func,
                'output': output[:800],
            })

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(problems)}, "
                      f"syntax_valid={syntax_valid/total:.4f}, "
                      f"has_func={has_func/total:.4f}")

        except Exception as e:
            print(f"  Error on problem {i}: {e}")
            results.append({
                'idx': i, 'problem': prob['text'][:200],
                'extracted_code': '', 'syntax_valid': False,
                'has_function': False, 'output': f'ERROR: {e}',
            })
            total += 1

    syntax_rate = syntax_valid / total if total > 0 else 0
    func_rate = has_func / total if total > 0 else 0
    return syntax_rate, func_rate, results


def evaluate_arc(model, tokenizer, problems, steps, device,
                 max_problems=None, gen_length=32, block_length=None):
    """Evaluate on ARC-Challenge. Reports accuracy."""
    if max_problems:
        problems = problems[:max_problems]

    results = []
    correct = 0
    total = 0

    for i, prob in enumerate(tqdm(problems, desc=f'ARC {steps}-step')):
        try:
            prompt = format_arc_prompt(prob)
            output = generate_response(
                model, tokenizer, prompt, steps, device,
                gen_length=gen_length, block_length=block_length)

            pred = extract_answer_letter(output)
            gold = prob['answer_key']
            is_correct = pred == gold

            correct += int(is_correct)
            total += 1

            results.append({
                'idx': i,
                'question': prob['question'][:200],
                'choices': dict(zip(prob['choices_label'], prob['choices_text'])),
                'gold': gold,
                'pred': pred,
                'correct': is_correct,
                'output': output[:300],
            })

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(problems)}, acc={correct/total:.4f}")

        except Exception as e:
            print(f"  Error on problem {i}: {e}")
            results.append({
                'idx': i, 'question': prob['question'][:200],
                'gold': prob['answer_key'], 'pred': '', 'correct': False,
                'output': f'ERROR: {e}',
            })
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLaDA on MBPP and ARC-Challenge')
    parser.add_argument('--checkpoint', default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Model checkpoint path or HF model ID')
    parser.add_argument('--dataset', choices=['mbpp', 'arc'], required=True,
                        help='Benchmark to evaluate on')
    parser.add_argument('--steps', type=int, default=64,
                        help='Number of remasking steps')
    parser.add_argument('--gen_length', type=int, default=512,
                        help='Max generation length in tokens')
    parser.add_argument('--block_length', type=int, default=None,
                        help='Block length for remasking (default=gen_length)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--max_problems', type=int, default=None,
                        help='Limit number of problems (for quick testing)')
    parser.add_argument('--output', default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    if args.block_length is None:
        args.block_length = args.gen_length

    # ARC only needs short outputs (just a letter)
    if args.dataset == 'arc' and args.gen_length == 512:
        args.gen_length = 32
        args.block_length = 32
        print("Note: ARC uses short generation (32 tokens). "
              "Override with --gen_length if needed.")

    device = f'cuda:{args.gpu}'
    os.makedirs('results', exist_ok=True)

    if args.output is None:
        ckpt_name = args.checkpoint.replace('/', '_').replace('.', '_')
        args.output = f'results/{args.dataset}_{ckpt_name}_{args.steps}step.json'

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    tokenizer.padding_side = 'left'

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'mbpp':
        problems = load_mbpp()
    else:
        problems = load_arc()
    print(f"  {len(problems)} problems loaded")

    # Evaluate
    print(f"\nEvaluating: {args.steps}-step on {args.dataset} "
          f"(gen_length={args.gen_length}, block_length={args.block_length})")
    t0 = time.time()

    if args.dataset == 'mbpp':
        syntax_rate, func_rate, results = evaluate_mbpp(
            model, tokenizer, problems, args.steps, device,
            max_problems=args.max_problems,
            gen_length=args.gen_length, block_length=args.block_length)
        elapsed = time.time() - t0

        summary = {
            'checkpoint': args.checkpoint,
            'dataset': 'mbpp',
            'steps': args.steps,
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'syntax_valid_rate': syntax_rate,
            'has_function_rate': func_rate,
            'n_syntax_valid': sum(r['syntax_valid'] for r in results),
            'n_has_function': sum(r['has_function'] for r in results),
            'n_total': len(results),
            'elapsed_sec': elapsed,
            'results': results,
        }

        print(f"\n{'='*50}")
        print(f"Result: MBPP | {args.steps}-step | {args.checkpoint}")
        print(f"Syntax valid rate: {syntax_rate:.4f} "
              f"({summary['n_syntax_valid']}/{summary['n_total']})")
        print(f"Has function rate: {func_rate:.4f} "
              f"({summary['n_has_function']}/{summary['n_total']})")

    else:  # arc
        accuracy, results = evaluate_arc(
            model, tokenizer, problems, args.steps, device,
            max_problems=args.max_problems,
            gen_length=args.gen_length, block_length=args.block_length)
        elapsed = time.time() - t0

        summary = {
            'checkpoint': args.checkpoint,
            'dataset': 'arc-challenge',
            'steps': args.steps,
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'accuracy': accuracy,
            'n_correct': sum(r['correct'] for r in results),
            'n_total': len(results),
            'elapsed_sec': elapsed,
            'results': results,
        }

        print(f"\n{'='*50}")
        print(f"Result: ARC-Challenge | {args.steps}-step | {args.checkpoint}")
        print(f"Accuracy: {accuracy:.4f} "
              f"({summary['n_correct']}/{summary['n_total']})")

    print(f"Time: {elapsed:.0f}s")

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
