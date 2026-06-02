"""Evaluate DSL-LLaDA reasoning on GSM8K and MATH-500.

Supports standard remasking and DSL two-stage generation.
Single-GPU per run; use eval_reasoning_parallel.sh to dispatch across 8 GPUs.

Usage:
    python dsl_llada/eval/eval_reasoning.py \
        --checkpoint GSAI-ML/LLaDA-8B-Instruct \
        --method standard --steps 64 \
        --dataset gsm8k --gpu 0 \
        --output results/gsm8k_original_std64.json
"""
import argparse
import json
import os
import re
import sys
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'LLaDA'))
from generate import generate

# XDLM generate with k1 (clean token replacement)
import importlib.util
_xdm_spec = importlib.util.spec_from_file_location(
    "generate_xdm_module",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'LLaDA-XDLM', 'generate.py'))
_xdm_mod = importlib.util.module_from_spec(_xdm_spec)
_xdm_spec.loader.exec_module(_xdm_mod)
generate_xdm = _xdm_mod.generate_xdm

MASK_ID = 126336


def load_model(path, device):
    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    return model.to(device).eval()


def attach_dsl(model, checkpoint, device):
    """Attach DSL modules from checkpoint if not already present."""
    if hasattr(model, 'noise_embed'):
        return
    sys.path.insert(0, os.path.dirname(__file__))
    from dsl_llada.core.dsl_modules import attach_dsl_modules
    attach_dsl_modules(model, freeze_ff_out=True)

    import safetensors.torch
    import glob
    shard_files = sorted(glob.glob(os.path.join(checkpoint, 'model-*.safetensors')))
    for sf in shard_files:
        sd = safetensors.torch.load_file(sf, device=str(device))
        for k, v in sd.items():
            if k.startswith('converter.') or k.startswith('noise_embed.'):
                parts = k.split('.')
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                param = getattr(obj, parts[-1])
                if isinstance(param, torch.nn.Parameter):
                    param.data.copy_(v)
                else:
                    setattr(obj, parts[-1], v)
        del sd
    model.noise_embed = model.noise_embed.to(device)
    model.converter = model.converter.to(device)


def load_gsm8k():
    """Load GSM8K test set."""
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    problems = []
    for item in ds:
        answer_text = item['answer']
        # Extract final number after ####
        match = re.search(r'####\s*(.+)', answer_text)
        gold = match.group(1).strip().replace(',', '') if match else ''
        problems.append({
            'question': item['question'],
            'gold': gold,
        })
    return problems


def load_mbpp():
    """Load MBPP sanitized test set (257 problems)."""
    from datasets import load_dataset
    ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
    problems = []
    for item in ds:
        problems.append({
            'question': item['prompt'],
            'gold': item['code'],
            'test_list': item['test_list'],
            'test_imports': item.get('test_imports', []),
            'task_id': item['task_id'],
        })
    return problems


def load_math500():
    """Load MATH-500: 500 problems sampled from hendrycks MATH test set."""
    from datasets import load_dataset, concatenate_datasets
    import random
    configs = ['algebra', 'counting_and_probability', 'geometry',
               'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    all_ds = []
    for c in configs:
        all_ds.append(load_dataset('EleutherAI/hendrycks_math', c, split='test'))
    full = concatenate_datasets(all_ds)
    # Deterministic sample of 500
    random.seed(42)
    indices = random.sample(range(len(full)), min(500, len(full)))
    problems = []
    for i in indices:
        item = full[i]
        # Extract answer from \boxed{} in solution
        solution = item.get('solution', '')
        matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
        gold = matches[-1].strip() if matches else ''
        problems.append({
            'question': item['problem'],
            'gold': gold,
        })
    return problems


def extract_number(text):
    """Extract the final number from model output (GSM8K format)."""
    # Try #### format first
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Try "the answer is X" pattern
    match = re.search(r'(?:answer|result)\s*(?:is|=)\s*\$?([+-]?[\d,]+\.?\d*)', text, re.I)
    if match:
        return match.group(1).replace(',', '')
    # Last number in the text
    numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return ''


def extract_boxed(text):
    """Extract content from \\boxed{...}."""
    # Find last \boxed{...}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    # Fallback: try to find the answer after "answer is"
    match = re.search(r'(?:answer|result)\s*(?:is|=)\s*(.+?)(?:\.|$)', text, re.I)
    if match:
        return match.group(1).strip()
    return ''


def extract_code(text):
    """Extract Python code from model output."""
    # Try ```python ... ``` blocks first
    matches = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Try to find def ... lines
    lines = text.split('\n')
    code_lines = []
    in_func = False
    for line in lines:
        if line.strip().startswith('def '):
            in_func = True
        if in_func:
            code_lines.append(line)
            # End on empty line after function body (with content)
            if line.strip() == '' and len(code_lines) > 2:
                # Check if next non-empty content is another def or not indented
                continue
    if code_lines:
        return '\n'.join(code_lines).strip()
    # Fallback: return entire text as code
    return text.strip()


def check_mbpp(code_output, test_list, test_imports=None, timeout=5):
    """Check if generated code passes MBPP test cases."""
    import subprocess, tempfile
    code = extract_code(code_output)

    # Build test script
    parts = []
    if test_imports:
        for imp in test_imports:
            parts.append(imp)
    parts.append(code)
    parts.append('')  # separator
    for test in test_list:
        parts.append(test)
    script = '\n'.join(parts)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
            f.write(script)
            f.flush()
            result = subprocess.run(
                ['python3', f.name],
                capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def normalize_answer(s):
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    s = s.replace(',', '').replace('$', '').replace('%', '')
    s = s.replace(' ', '')
    # Remove trailing period
    s = s.rstrip('.')
    return s


def dsl_two_stage_generate(model, prompt, gen_length=512, total_steps=8,
                           switch_ratio=0.5, snr_max=100.0):
    """Two-stage generation using DSL noise path."""
    from dsl_llada.core.dsl_modules import noisy_embedding
    B, P = prompt.shape
    seq_len = P + gen_length
    device = prompt.device

    x = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt
    prompt_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    prompt_mask[:, :P] = True
    switch_step = int(total_steps * switch_ratio)

    for step in range(total_steps):
        snrs = torch.zeros(B, seq_len, device=device)
        snrs[x != MASK_ID] = snr_max

        with torch.no_grad():
            z_noisy = noisy_embedding(model.noise_embed, x, snrs)
            h = model.converter(z_noisy).to(dtype=torch.bfloat16)
            out = model(input_ids=x, inputs_embeds=h)
            probs = F.softmax(out.logits.float(), dim=-1)

        confidence, predicted = probs.max(dim=-1)
        mask_positions = (x == MASK_ID) & (~prompt_mask)

        if step < switch_step:
            n_masked = mask_positions[0].sum().item()
            n_to_unmask = max(1, int(n_masked * (step + 1) / switch_step))
            conf_masked = confidence.clone()
            conf_masked[~mask_positions] = -float('inf')
            n_to_unmask = min(n_to_unmask, int(mask_positions.sum(dim=-1).min().item()))
            if n_to_unmask > 0:
                _, top_idx = conf_masked.topk(n_to_unmask, dim=-1)
                x.scatter_(1, top_idx, predicted.gather(1, top_idx))
        else:
            progress = (step - switch_step) / max(1, total_steps - switch_step)
            threshold = 0.9 - 0.5 * progress
            new_x = predicted.clone()
            new_x[confidence < threshold] = MASK_ID
            new_x[prompt_mask] = x[prompt_mask]
            x = new_x

    return x


def generate_answer(model, tokenizer, question, method, steps, device,
                    gen_length=512, block_length=None, dataset_type='gsm8k',
                    save_trajectory=False, **kwargs):
    """Generate answer for a single question. Returns (text, trajectory_or_None)."""
    if block_length is None:
        block_length = gen_length

    if dataset_type == 'gsm8k':
        prompt_text = (
            f"Solve the following math problem step by step. "
            f"Show your work and put your final answer after ####.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    elif dataset_type == 'mbpp':
        test_examples = kwargs.get('test_list', [])
        test_str = '\n'.join(test_examples[:3]) if test_examples else ''
        prompt_text = (
            f"You are an expert Python programmer. Write a function to solve the following task.\n\n"
            f"{question}\n\n"
            f"Your code should satisfy these tests:\n{test_str}\n\n"
            f"```python\n"
        )
    else:  # math500
        prompt_text = (
            f"Solve the following math problem. "
            f"Show your work and put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {question}\n\nSolution:"
        )

    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)

    trajectory = None
    with torch.no_grad():
        if method == 'dsl':
            out = dsl_two_stage_generate(
                model, input_ids, gen_length=gen_length,
                total_steps=steps, switch_ratio=0.5)
        elif method == 'xdm':  # XDLM generate with k1 clean token replacement
            attention_mask = torch.ones_like(input_ids)
            k1 = kwargs.get('k1', 0.1)
            out = generate_xdm(
                model, input_ids, attention_mask,
                steps=steps, gen_length=gen_length, block_length=block_length,
                temperature=0., cfg_scale=0., remasking='low_confidence',
                confidence_eos_eot_inf=kwargs.get('confidence_eos_eot_inf', False),
                k1=k1)
        else:  # standard or d3im remasking
            attention_mask = torch.ones_like(input_ids)
            remasking_strategy = method if method in ('d3im', 'd3im_margin', 'remdm_conf', 'remdm_cap') else 'low_confidence'
            eos_ratio = kwargs.get('eos_suppress_ratio', 0.0)
            result = generate(
                model, input_ids, attention_mask,
                steps=steps, gen_length=gen_length, block_length=block_length,
                temperature=0., cfg_scale=0., remasking=remasking_strategy,
                eos_suppress_ratio=eos_ratio,
                confidence_eos_eot_inf=kwargs.get('confidence_eos_eot_inf', False),
                save_trajectory=save_trajectory)
            if save_trajectory:
                out, trajectory = result
            else:
                out = result

    generated = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return generated, trajectory


def evaluate(model, tokenizer, problems, method, steps, device,
             dataset_type='gsm8k', max_problems=None,
             gen_length=512, block_length=None, save_trajectory=False, **kwargs):
    """Evaluate on a list of problems."""
    if max_problems:
        problems = problems[:max_problems]

    results = []
    trajectories = [] if save_trajectory else None
    correct = 0
    total = 0

    for i, prob in enumerate(tqdm(problems, desc=f'{dataset_type} {method}-{steps}step')):
        try:
            extra_kwargs = dict(kwargs)
            if dataset_type == 'mbpp':
                extra_kwargs['test_list'] = prob.get('test_list', [])
            output, traj = generate_answer(
                model, tokenizer, prob['question'], method, steps, device,
                gen_length=gen_length, block_length=block_length,
                dataset_type=dataset_type, save_trajectory=save_trajectory,
                **extra_kwargs)

            if dataset_type == 'gsm8k':
                pred = extract_number(output)
                gold = prob['gold']
                is_correct = normalize_answer(pred) == normalize_answer(gold)
            elif dataset_type == 'mbpp':
                pred = extract_code(output)
                gold = prob['gold']
                is_correct = check_mbpp(output, prob['test_list'], prob.get('test_imports'))
            else:
                pred = extract_boxed(output)
                gold = prob['gold']
                is_correct = normalize_answer(pred) == normalize_answer(gold)

            correct += int(is_correct)
            total += 1

            results.append({
                'idx': i,
                'question': prob['question'][:200],
                'gold': gold[:200] if isinstance(gold, str) else gold,
                'pred': pred[:200] if isinstance(pred, str) else pred,
                'correct': is_correct,
                'output': output[:500],
            })

            if save_trajectory and traj is not None:
                trajectories.append(traj)

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(problems)}, acc={correct/total:.4f}")

        except Exception as e:
            print(f"  Error on problem {i}: {e}")
            results.append({
                'idx': i, 'question': prob['question'][:200],
                'gold': prob['gold'], 'pred': '', 'correct': False,
                'output': f'ERROR: {e}',
            })
            total += 1
            if save_trajectory:
                trajectories.append(None)

    accuracy = correct / total if total > 0 else 0
    return accuracy, results, trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--method', choices=['standard', 'dsl', 'd3im', 'd3im_margin', 'remdm_conf', 'remdm_cap', 'xdm'], default='standard')
    parser.add_argument('--k1', type=float, default=0.1, help='XDLM k1 parameter for clean token replacement ratio')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--dataset', choices=['gsm8k', 'math500', 'mbpp'], default='gsm8k')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--gen_length', type=int, default=512,
                        help='Max generation length in tokens')
    parser.add_argument('--block_length', type=int, default=None,
                        help='Block length for remasking (default=gen_length)')
    parser.add_argument('--max_problems', type=int, default=None,
                        help='Limit number of problems (for quick test)')
    parser.add_argument('--eos_suppress_ratio', type=float, default=0.0,
                        help='Suppress EOS/EoT logits in first X%% of steps (0.0=off, 0.8=first 80%%)')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                        help='Set confidence of EOS and EoT tokens to -inf (see LLaDA Appendix B.4)')
    parser.add_argument('--save_trajectory', action='store_true',
                        help='Save per-step token predictions and confidence')
    args = parser.parse_args()
    if args.block_length is None:
        args.block_length = args.gen_length

    device = f'cuda:{args.gpu}'
    os.makedirs('results', exist_ok=True)

    if args.output is None:
        ckpt_name = args.checkpoint.replace('/', '_').replace('.', '_')
        args.output = f'results/{args.dataset}_{ckpt_name}_{args.method}{args.steps}.json'

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    tokenizer.padding_side = 'left'

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    if args.method == 'dsl' and args.checkpoint != 'GSAI-ML/LLaDA-8B-Instruct':
        print("Attaching DSL modules...")
        attach_dsl(model, args.checkpoint, device)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'gsm8k':
        problems = load_gsm8k()
    elif args.dataset == 'mbpp':
        problems = load_mbpp()
    else:
        problems = load_math500()
    print(f"  {len(problems)} problems loaded")

    print(f"\nEvaluating: {args.method} {args.steps}-step on {args.dataset} "
          f"(gen_length={args.gen_length}, block_length={args.block_length})"
          f"{' [saving trajectory]' if args.save_trajectory else ''}")
    t0 = time.time()
    accuracy, results, trajectories = evaluate(
        model, tokenizer, problems, args.method, args.steps, device,
        dataset_type=args.dataset, max_problems=args.max_problems,
        gen_length=args.gen_length, block_length=args.block_length,
        save_trajectory=args.save_trajectory,
        eos_suppress_ratio=args.eos_suppress_ratio,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        k1=args.k1)
    elapsed = time.time() - t0

    summary = {
        'checkpoint': args.checkpoint,
        'method': args.method,
        'steps': args.steps,
        'dataset': args.dataset,
        'gen_length': args.gen_length,
        'block_length': args.block_length,
        'accuracy': accuracy,
        'n_correct': sum(r['correct'] for r in results),
        'n_total': len(results),
        'elapsed_sec': elapsed,
        'results': results,
    }

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save trajectories as .pt file (compact binary)
    if args.save_trajectory and trajectories:
        traj_path = args.output.replace('.json', '_traj.pt')
        valid_trajs = [t for t in trajectories if t is not None]
        if valid_trajs:
            merged = {
                'tokens': torch.stack([t['tokens'] for t in valid_trajs]),    # (N, steps, gen_len)
                'confidence': torch.stack([t['confidence'] for t in valid_trajs]),
                'x_state': torch.stack([t['x_state'] for t in valid_trajs]),
            }
            torch.save(merged, traj_path)
            n_traj = len(valid_trajs)
            traj_size_mb = os.path.getsize(traj_path) / 1024 / 1024
            print(f"Trajectories: {n_traj} saved to {traj_path} ({traj_size_mb:.1f} MB)")

    print(f"\n{'='*50}")
    print(f"Result: {args.dataset} | {args.method}-{args.steps}step | {args.checkpoint}")
    print(f"Accuracy: {accuracy:.4f} ({summary['n_correct']}/{summary['n_total']})")
    print(f"Time: {elapsed:.0f}s")
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
