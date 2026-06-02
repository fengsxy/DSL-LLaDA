"""Unified evaluation script for DSL-LLaDA.

Reads model config from dsl_llada/configs/registry.json, loads model once,
runs 4 test suites (T1 corruption, T2 generation, T3 mask prediction,
T4 reasoning), saves structured JSON results.

Usage:
    python dsl_llada/eval/eval_unified.py --model_key sm_b2 --gpu 0
    python dsl_llada/eval/eval_unified.py --model_key sm_b2 --tests t1,t4 --gpu 0
    python dsl_llada/eval/eval_unified.py --model_key sm_b2 --tests t2 --gpu 0 --sde_beta_infer 1.5
    python dsl_llada/eval/eval_unified.py --model_key original --gpu 0 --skip_existing
    python dsl_llada/eval/eval_unified.py --aggregate
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "external", "LLaDA"))
from generate import generate

MASK_ID = 126336
EVAL_DATA_DIR = os.path.join(_root, "eval_data")
EVAL_RESULTS_DIR = os.path.join(_root, "eval_results")
REGISTRY_PATH = os.environ.get(
    "DSL_LLADA_REGISTRY",
    os.path.join(_root, "dsl_llada", "configs", "registry.json"),
)


# ===================================================================
# Core infrastructure
# ===================================================================

def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def resolve_checkpoint(entry):
    """Return absolute path (local) or HF model id."""
    path = entry["path"]
    if entry.get("type") == "local":
        abs_path = os.path.join(_root, path)
        if os.path.isdir(abs_path):
            return abs_path
        # Try as-is (maybe already absolute)
        if os.path.isdir(path):
            return path
        print(f"  WARNING: local checkpoint not found: {abs_path}")
        return abs_path
    return path  # HF id


def load_model(checkpoint, device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    return model.to(device).eval()


def attach_dsl(model, checkpoint_dir, device, dsl_config):
    """Attach DSL converter/noise_embed from checkpoint safetensors."""
    if hasattr(model, "noise_embed"):
        return

    # Set env vars BEFORE importing attach_dsl_modules
    if dsl_config:
        os.environ["DSL_NOISE_DIM"] = str(dsl_config.get("noise_dim", 48))
        os.environ["DSL_BETA_INIT"] = str(dsl_config.get("beta_init", 5.0))
        os.environ["DSL_NOISE_INIT"] = str(dsl_config.get("noise_init", "random"))

    sys.path.insert(0, _script_dir)
    from dsl_llada.core.dsl_modules import attach_dsl_modules
    attach_dsl_modules(model, freeze_ff_out=True)

    # Load trained weights from safetensors shards
    import safetensors.torch
    shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
    for sf in shard_files:
        sd = safetensors.torch.load_file(sf, device=str(device))
        for k, v in sd.items():
            if k.startswith("converter.") or k.startswith("noise_embed."):
                parts = k.split(".")
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
    # Ensure ALL sub-modules (including backbone_embedding) are on device
    for name, param in model.named_parameters():
        if param.device != torch.device(device):
            param.data = param.data.to(device)
    for name, buf in model.named_buffers():
        if buf.device != torch.device(device):
            buf.data = buf.data.to(device)
    print(f"  DSL modules attached from {checkpoint_dir}")


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    tok.padding_side = "left"
    return tok


def eval_data_hash():
    """MD5 of all eval_data json files for reproducibility."""
    h = hashlib.md5()
    for fn in sorted(os.listdir(EVAL_DATA_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(EVAL_DATA_DIR, fn), "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def save_result(model_key, filename, data):
    """Save JSON result to eval_results/{model_key}/."""
    out_dir = os.path.join(EVAL_RESULTS_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    # Backup existing
    if os.path.exists(out_path):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = out_path + f".bak_{ts}"
        os.rename(out_path, bak)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")
    return out_path


def result_exists(model_key, filename):
    return os.path.exists(os.path.join(EVAL_RESULTS_DIR, model_key, filename))


def write_meta(model_key, entry, tests_run, duration, data_hash):
    meta = {
        "model_key": model_key,
        "description": entry.get("description", ""),
        "checkpoint": entry.get("path", ""),
        "tests_run": tests_run,
        "duration_sec": round(duration, 1),
        "data_hash": data_hash,
        "timestamp": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    save_result(model_key, "meta.json", meta)


# ===================================================================
# Data loaders
# ===================================================================

def _load_json(name):
    with open(os.path.join(EVAL_DATA_DIR, name)) as f:
        return json.load(f)


# ===================================================================
# T1: Token Corruption
# ===================================================================

def corrupt_tokens_random(tokens, rate, seed):
    """Replace rate% of tokens with random token_id in [100, 126000).

    Returns (corrupted_list, positions, details).
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)
    n_corrupt = max(1, int(n * rate))
    positions = sorted(rng.choice(n, n_corrupt, replace=False).tolist())
    corrupted = list(tokens)
    details = []
    for pos in positions:
        new_tok = rng.randint(100, 126000)
        while new_tok == tokens[pos]:
            new_tok = rng.randint(100, 126000)
        details.append({"pos": pos, "original": tokens[pos], "replacement": int(new_tok)})
        corrupted[pos] = int(new_tok)
    return corrupted, positions, details


def build_wte_topk(model, tokenizer, tokens_set, k=5):
    """Precompute wte cosine top-k for a set of unique token ids.

    Returns dict: {token_id: [{"id": int, "cos": float}, ...]}.
    """
    wte = model.model.transformer.wte.weight.detach().float()  # (V, d)
    wte_norm = F.normalize(wte, dim=-1)
    result = {}
    token_list = sorted(tokens_set)
    # Process in chunks to avoid OOM
    chunk_sz = 512
    for start in range(0, len(token_list), chunk_sz):
        chunk = token_list[start : start + chunk_sz]
        idx = torch.tensor(chunk, dtype=torch.long, device=wte.device)
        vecs = wte_norm[idx]  # (chunk, d)
        sims = vecs @ wte_norm.T  # (chunk, V)
        # Zero out self-similarity
        for i, tid in enumerate(chunk):
            sims[i, tid] = -1.0
        topk_vals, topk_idx = sims.topk(k, dim=-1)
        for i, tid in enumerate(chunk):
            neighbors = []
            for j in range(k):
                neighbors.append({
                    "id": int(topk_idx[i, j].item()),
                    "cos": round(float(topk_vals[i, j].item()), 4),
                })
            result[tid] = neighbors
    return result


def corrupt_tokens_semantic(tokens, rate, wte_topk, seed):
    """Replace rate% of tokens with wte cosine top-5 neighbor.

    Falls back to random if token not in topk map.
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)
    n_corrupt = max(1, int(n * rate))
    positions = sorted(rng.choice(n, n_corrupt, replace=False).tolist())
    corrupted = list(tokens)
    details = []
    for pos in positions:
        orig_tok = tokens[pos]
        neighbors = wte_topk.get(orig_tok)
        if neighbors:
            choice_idx = rng.randint(0, len(neighbors))
            new_tok = neighbors[choice_idx]["id"]
            cos = neighbors[choice_idx]["cos"]
        else:
            new_tok = rng.randint(100, 126000)
            while new_tok == orig_tok:
                new_tok = rng.randint(100, 126000)
            cos = None
        details.append({
            "pos": pos, "original": orig_tok,
            "replacement": int(new_tok), "cos_sim": cos,
        })
        corrupted[pos] = int(new_tok)
    return corrupted, positions, details


def eval_corruption_single(model, tokens_tensor, gold_tokens, corrupted_positions, device):
    """Single forward pass to evaluate corruption fix.

    Returns (fix_rate, clean_preserved).
    """
    input_ids = tokens_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    preds = out.logits[0].argmax(dim=-1).cpu()
    gold = torch.tensor(gold_tokens, dtype=torch.long)

    # Fix rate: fraction of corrupted positions where model predicts the original token
    corrupted_set = set(corrupted_positions)
    fixed = sum(1 for pos in corrupted_positions if preds[pos].item() == gold[pos].item())
    fix_rate = fixed / len(corrupted_positions) if corrupted_positions else 0.0

    # Clean preserved: fraction of non-corrupted positions where prediction matches gold
    clean_positions = [i for i in range(len(gold_tokens)) if i not in corrupted_set]
    if clean_positions:
        clean_correct = sum(1 for pos in clean_positions if preds[pos].item() == gold[pos].item())
        clean_preserved = clean_correct / len(clean_positions)
    else:
        clean_preserved = 1.0

    return fix_rate, clean_preserved


def run_t1(model, tokenizer, device, model_key):
    """Run T1: token corruption tests."""
    print("\n" + "=" * 60)
    print("T1: Token Corruption")
    print("=" * 60)

    texts_data = _load_json("texts_100.json")
    result = {
        "test": "t1_corruption",
        "model": model_key,
        "timestamp": datetime.datetime.now().isoformat(),
        "subtests": {},
    }

    # T1a: random corruption at 10%, 30%, 50%
    for rate_pct in [10, 30, 50]:
        rate = rate_pct / 100.0
        key = f"random_{rate_pct}"
        fix_rates, clean_rates = [], []

        for item in tqdm(texts_data, desc=f"T1a random {rate_pct}%"):
            gold_tokens = item["tokens"]
            seed = 42 + item["id"]
            corrupted, positions, _ = corrupt_tokens_random(gold_tokens, rate, seed)
            tokens_tensor = torch.tensor(corrupted, dtype=torch.long)
            fix_r, clean_r = eval_corruption_single(model, tokens_tensor, gold_tokens, positions, device)
            fix_rates.append(fix_r)
            clean_rates.append(clean_r)

        result["subtests"][key] = {
            "fix_rate": round(float(np.mean(fix_rates)), 4),
            "clean_preserved": round(float(np.mean(clean_rates)), 4),
            "n_texts": len(texts_data),
        }
        print(f"  {key}: fix={np.mean(fix_rates):.3f}, clean={np.mean(clean_rates):.3f}")

    # T1b: semantic auto corruption at 10%, 30%
    # Collect all unique tokens
    all_tokens = set()
    for item in texts_data:
        all_tokens.update(item["tokens"])
    print(f"  Building wte top-k for {len(all_tokens)} unique tokens...")
    wte_topk = build_wte_topk(model, tokenizer, all_tokens, k=5)

    for rate_pct in [10, 30]:
        rate = rate_pct / 100.0
        key = f"semantic_auto_{rate_pct}"
        fix_rates, clean_rates = [], []

        for item in tqdm(texts_data, desc=f"T1b semantic {rate_pct}%"):
            gold_tokens = item["tokens"]
            seed = 42 + item["id"]
            corrupted, positions, _ = corrupt_tokens_semantic(gold_tokens, rate, wte_topk, seed)
            tokens_tensor = torch.tensor(corrupted, dtype=torch.long)
            fix_r, clean_r = eval_corruption_single(model, tokens_tensor, gold_tokens, positions, device)
            fix_rates.append(fix_r)
            clean_rates.append(clean_r)

        result["subtests"][key] = {
            "fix_rate": round(float(np.mean(fix_rates)), 4),
            "clean_preserved": round(float(np.mean(clean_rates)), 4),
            "n_texts": len(texts_data),
        }
        print(f"  {key}: fix={np.mean(fix_rates):.3f}, clean={np.mean(clean_rates):.3f}")

    # T1c: semantic manual corruption
    manual_data = _load_json("semantic_corruption_manual.json")
    manual_results = []
    by_difficulty = {}
    by_category = {}
    by_corruption_type = {}

    for item in tqdm(manual_data, desc="T1c semantic manual"):
        try:
            orig_enc = tokenizer(item["original_text"], add_special_tokens=False, return_tensors="pt")
            corr_enc = tokenizer(item["corrupted_text"], add_special_tokens=False, return_tensors="pt")

            orig_ids = orig_enc["input_ids"][0].tolist()
            corr_ids = corr_enc["input_ids"][0].tolist()

            # Find corrupted positions by diffing
            min_len = min(len(orig_ids), len(corr_ids))
            positions = [i for i in range(min_len) if orig_ids[i] != corr_ids[i]]
            # Also mark extra/missing tokens at the end
            if len(corr_ids) > len(orig_ids):
                positions.extend(range(len(orig_ids), len(corr_ids)))

            if not positions:
                continue

            tokens_tensor = torch.tensor(corr_ids[:min_len], dtype=torch.long)
            gold_tokens = orig_ids[:min_len]
            positions = [p for p in positions if p < min_len]

            if not positions:
                continue

            fix_r, clean_r = eval_corruption_single(model, tokens_tensor, gold_tokens, positions, device)

            manual_results.append({
                "id": item["id"],
                "fix_rate": fix_r,
                "clean_preserved": clean_r,
                "difficulty": item.get("difficulty", "unknown"),
                "category": item.get("category", "unknown"),
                "corruption_type": item["corruptions"][0].get("corruption_type", "unknown") if item.get("corruptions") else "unknown",
            })

            # Aggregate by difficulty
            diff = item.get("difficulty", "unknown")
            by_difficulty.setdefault(diff, []).append(fix_r)

            # Aggregate by category
            cat = item.get("category", "unknown")
            by_category.setdefault(cat, []).append(fix_r)

            # Aggregate by corruption type
            ctype = item["corruptions"][0].get("corruption_type", "unknown") if item.get("corruptions") else "unknown"
            by_corruption_type.setdefault(ctype, []).append(fix_r)

        except Exception as e:
            print(f"  Error on manual item {item.get('id', '?')}: {e}")
            continue

    overall_fix = float(np.mean([r["fix_rate"] for r in manual_results])) if manual_results else 0.0
    overall_clean = float(np.mean([r["clean_preserved"] for r in manual_results])) if manual_results else 0.0

    result["subtests"]["semantic_manual"] = {
        "overall": {
            "fix_rate": round(overall_fix, 4),
            "clean_preserved": round(overall_clean, 4),
            "n_cases": len(manual_results),
        },
        "by_difficulty": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_difficulty.items())},
        "by_category": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_category.items())},
        "by_corruption_type": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_corruption_type.items())},
    }
    print(f"  semantic_manual: fix={overall_fix:.3f}, clean={overall_clean:.3f} ({len(manual_results)} cases)")
    for d, vals in sorted(by_difficulty.items()):
        print(f"    {d}: fix={np.mean(vals):.3f} (n={len(vals)})")

    save_result(model_key, "t1_corruption.json", result)
    return result


# ===================================================================
# T3: Mask Prediction
# ===================================================================

def compute_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error."""
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (n_bin / total) * abs(avg_acc - avg_conf)
    return float(ece)


def run_t3(model, tokenizer, device, model_key):
    """Run T3: mask prediction at 30%, 50%, 70%."""
    print("\n" + "=" * 60)
    print("T3: Mask Prediction")
    print("=" * 60)

    texts_data = _load_json("texts_100.json")
    result = {
        "test": "t3_mask",
        "model": model_key,
        "timestamp": datetime.datetime.now().isoformat(),
        "mask_rates": {},
    }

    for rate_pct in [30, 50, 70]:
        rate = rate_pct / 100.0
        all_conf, all_acc = [], []
        total_correct = 0
        total_tokens = 0

        for item in tqdm(texts_data, desc=f"T3 mask {rate_pct}%"):
            gold_tokens = item["tokens"]
            n = len(gold_tokens)
            seed = 42 + item["id"]
            rng = np.random.RandomState(seed)
            n_mask = max(1, int(n * rate))
            mask_positions = sorted(rng.choice(n, n_mask, replace=False).tolist())

            input_ids = list(gold_tokens)
            for pos in mask_positions:
                input_ids[pos] = MASK_ID
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(input_ids=input_tensor)

            logits = out.logits[0].float()
            probs = F.softmax(logits, dim=-1)

            for pos in mask_positions:
                conf = probs[pos].max().item()
                pred = probs[pos].argmax().item()
                correct = int(pred == gold_tokens[pos])
                all_conf.append(conf)
                all_acc.append(correct)
                total_correct += correct
                total_tokens += 1

        accuracy = total_correct / total_tokens if total_tokens else 0.0
        ece = compute_ece(all_conf, all_acc)
        avg_conf = float(np.mean(all_conf)) if all_conf else 0.0

        result["mask_rates"][str(rate_pct)] = {
            "accuracy": round(accuracy, 4),
            "ece": round(ece, 4),
            "avg_confidence": round(avg_conf, 4),
            "n_tokens": total_tokens,
        }
        print(f"  mask {rate_pct}%: acc={accuracy:.3f}, ece={ece:.4f}, conf={avg_conf:.3f} ({total_tokens} tokens)")

    save_result(model_key, "t3_mask.json", result)
    return result


# ===================================================================
# T4: Reasoning (GSM8K + MATH)
# ===================================================================

def extract_number(text):
    """Extract final number from GSM8K output."""
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"(?:answer|result)\s*(?:is|=)\s*\$?([+-]?[\d,]+\.?\d*)", text, re.I)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def extract_boxed(text):
    """Extract from \\boxed{...}."""
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    match = re.search(r"(?:answer|result)\s*(?:is|=)\s*(.+?)(?:\.|$)", text, re.I)
    if match:
        return match.group(1).strip()
    return ""


def normalize_answer(s):
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "")
    s = s.rstrip(".")
    return s


def run_t4(model, tokenizer, device, model_key):
    """Run T4: reasoning (GSM8K 100 + MATH 100)."""
    print("\n" + "=" * 60)
    print("T4: Reasoning")
    print("=" * 60)

    result = {
        "test": "t4_reasoning",
        "model": model_key,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    for dataset_name in ["gsm8k", "math"]:
        data = _load_json(f"{dataset_name}_100.json")
        cases = []
        correct_count = 0

        for item in tqdm(data, desc=f"T4 {dataset_name}"):
            try:
                if dataset_name == "gsm8k":
                    prompt_text = (
                        f"Solve the following math problem step by step. "
                        f"Show your work and put your final answer after ####.\n\n"
                        f"Question: {item['question']}\n\nAnswer:"
                    )
                else:
                    prompt_text = (
                        f"Solve the following math problem. "
                        f"Show your work and put your final answer in \\boxed{{}}.\n\n"
                        f"Problem: {item['question']}\n\nSolution:"
                    )

                messages = [{"role": "user", "content": prompt_text}]
                formatted = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    out = generate(
                        model, input_ids, attention_mask,
                        steps=64, gen_length=256,
                        block_length=256, temperature=0.0,
                        cfg_scale=0.0, remasking="low_confidence",
                    )
                generated_text = tokenizer.decode(
                    out[0, input_ids.shape[1]:], skip_special_tokens=True
                )

                if dataset_name == "gsm8k":
                    pred = extract_number(generated_text)
                else:
                    pred = extract_boxed(generated_text)

                gold = str(item["gold_answer"])
                is_correct = normalize_answer(pred) == normalize_answer(gold)
                correct_count += int(is_correct)

                cases.append({
                    "id": item["id"],
                    "original_idx": item.get("original_idx", -1),
                    "gold": gold[:200],
                    "pred": pred[:200],
                    "correct": is_correct,
                    "text": generated_text[:500],
                })

            except Exception as e:
                print(f"  Error on {dataset_name} item {item.get('id', '?')}: {e}")
                cases.append({
                    "id": item.get("id", -1),
                    "original_idx": item.get("original_idx", -1),
                    "gold": str(item.get("gold_answer", "")),
                    "pred": "",
                    "correct": False,
                    "text": f"ERROR: {e}",
                })

        acc = correct_count / len(data) if data else 0.0
        result[dataset_name] = {
            "accuracy": round(acc, 4),
            "n": len(data),
            "correct": correct_count,
            "cases": cases,
        }
        print(f"  {dataset_name}: {correct_count}/{len(data)} = {acc:.3f}")

    save_result(model_key, "t4_reasoning.json", result)
    return result


# ===================================================================
# T5: Trip Planning (Order Robustness, Multi-Step)
# ===================================================================

def _trip_build_templates(categories):
    """Build total_first and total_last JSON instruction templates."""
    cat_fields = ", ".join(f'"{c}": <number>' for c in categories)
    day_schema = f'{{"day": <N>, {cat_fields}}}'
    total_first = (
        f'Respond with ONLY a valid JSON: {{"total_cost": <number>, '
        f'"daily_breakdown": [{day_schema}, ...]}}. '
        f'total_cost MUST equal the sum of all daily costs '
        f'({"+".join(categories)} for each day). Stay within budget.'
    )
    total_last = (
        f'Respond with ONLY a valid JSON: {{"daily_breakdown": [{day_schema}, ...], '
        f'"total_cost": <number>}}. '
        f'total_cost MUST equal the sum of all daily costs '
        f'({"+".join(categories)} for each day). Stay within budget.'
    )
    return total_first, total_last


def _trip_check(text, budget, num_days, categories):
    """Check if generated JSON is valid and internally consistent."""
    try:
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return {"valid": False, "reason": "no_json"}
        data = json.loads(m.group())
        total = data.get("total_cost", 0)
        daily = data.get("daily_breakdown", [])
        if not isinstance(total, (int, float)) or not isinstance(daily, list):
            return {"valid": False, "reason": "bad_types"}
        actual = sum(d.get(k, 0) for d in daily for k in categories)
        consistent = abs(total - actual) < 1
        within_budget = total <= budget
        correct_days = len(daily) == num_days
        ok = consistent and within_budget and correct_days
        return {
            "valid": True, "ok": ok,
            "total_cost": total, "actual_sum": actual,
            "consistent": consistent, "within_budget": within_budget,
            "correct_days": correct_days, "n_days_gen": len(daily),
            "diff": abs(total - actual),
        }
    except json.JSONDecodeError:
        return {"valid": False, "reason": "json_parse_error"}
    except Exception as e:
        return {"valid": False, "reason": str(e)}


def _t5_generate_remask(model, tokenizer, prompt_text, device, steps):
    """Trip planning generation via standard remasking."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = generate(
            model, input_ids, attention_mask,
            steps=steps, gen_length=512,
            block_length=512, temperature=0.0,
            cfg_scale=0.0, remasking="low_confidence",
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def _t5_generate_sde(model, tokenizer, prompt_text, device, nfe, sde_config):
    """Trip planning generation via SDE (continuous space)."""
    # Use gen_length=512 for trip planning (longer JSON)
    cfg = dict(sde_config)
    # NFE = steps * 2 for heun, so steps = nfe // 2
    solver = cfg.get("solver", "heun")
    cfg["steps"] = nfe // 2 if solver == "heun" else nfe
    text = generate_sde(model, tokenizer, prompt_text, device, cfg)
    return text


def _t5_run_config(model, tokenizer, device, problems, method, steps_or_nfe,
                    order, sde_config=None):
    """Run one (method, steps, order) configuration and return results dict."""
    cases = []
    ok_by_level = {}

    desc = f"T5 {method} nfe={steps_or_nfe} {order}"
    for prob in tqdm(problems, desc=desc):
        cats = prob["categories"]
        tf, tl = _trip_build_templates(cats)
        template = tf if order == "total_first" else tl
        prompt_text = f"{prob['query']}\n\n{template}"

        try:
            if method == "remask":
                gen_text = _t5_generate_remask(
                    model, tokenizer, prompt_text, device, steps_or_nfe
                )
            else:
                gen_text = _t5_generate_sde(
                    model, tokenizer, prompt_text, device, steps_or_nfe,
                    sde_config
                )
        except Exception as e:
            gen_text = f"ERROR: {e}"

        check = _trip_check(gen_text, prob["budget"], prob["days"], cats)
        level = prob["level"]
        ok_by_level.setdefault(level, {"ok": 0, "total": 0})
        ok_by_level[level]["total"] += 1
        if check.get("ok"):
            ok_by_level[level]["ok"] += 1

        cases.append({
            "level": level,
            "query": prob["query"][:100],
            "order": order,
            "ok": check.get("ok", False),
            "check": check,
            "text": gen_text[:600],
        })

    total_ok = sum(v["ok"] for v in ok_by_level.values())
    total_n = sum(v["total"] for v in ok_by_level.values())

    parts = [
        f"{lv}:{ok_by_level.get(lv, {}).get('ok', 0)}/{ok_by_level.get(lv, {}).get('total', 0)}"
        for lv in ["D1", "D2", "D3", "D4"]
    ]
    print(f"  {method} nfe={steps_or_nfe} {order}: {' '.join(parts)} (total {total_ok}/{total_n})")

    return {
        "accuracy": round(total_ok / total_n, 4) if total_n else 0,
        "ok": total_ok,
        "total": total_n,
        "by_level": ok_by_level,
        "cases": cases,
    }


def run_t5(model, tokenizer, device, model_key, entry=None,
           checkpoint_dir=None, methods=None):
    """Run T5: trip planning order robustness at multiple step counts.

    Args:
        methods: list of "remask" and/or "sde". Default: ["remask"].
                 SDE requires DSL modules attached.
    """
    print("\n" + "=" * 60)
    print("T5: Trip Planning (Order Robustness)")
    print("=" * 60)

    if methods is None:
        methods = ["remask"]

    problems = _load_json("trip_planning_80.json")
    nfe_counts = [8, 16, 32, 64]

    # Attach DSL if needed for SDE
    _is_dsl = entry and entry.get("dsl") and entry.get("type") == "local"
    if "sde" in methods and _is_dsl and checkpoint_dir:
        if not hasattr(model, "noise_embed") or model.noise_embed is None:
            dsl_config = entry.get("dsl_config", {})
            attach_dsl(model, checkpoint_dir, device, dsl_config)

    sde_config = entry.get("sde_config", {}) if entry else {}

    result = {
        "test": "t5_trip_planning",
        "model": model_key,
        "timestamp": datetime.datetime.now().isoformat(),
        "nfe_counts": nfe_counts,
        "methods": methods,
        "n_problems": len(problems),
        "by_nfe": {},
    }

    for nfe in nfe_counts:
        nfe_result = {}
        for method in methods:
            if method == "sde" and not _is_dsl:
                continue
            for order in ["total_last", "total_first"]:
                key = f"{method}_{order}"
                nfe_result[key] = _t5_run_config(
                    model, tokenizer, device, problems, method, nfe,
                    order, sde_config=sde_config
                )
        result["by_nfe"][str(nfe)] = nfe_result

    save_result(model_key, "t5_trip_planning.json", result)
    return result


# ===================================================================
# T2: Generation Quality
# ===================================================================

def compute_gen_ppl(texts, device):
    """GPT-2 Large perplexity on generated texts (truncate to 128 words)."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    gpt2_name = "gpt2-large"
    print(f"  Loading {gpt2_name} for GenPPL...")
    gpt2_tok = GPT2TokenizerFast.from_pretrained(gpt2_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_name).to(device).eval()

    nlls = []
    for text in texts:
        # Truncate to 128 words
        words = text.split()[:128]
        truncated = " ".join(words)
        if not truncated.strip():
            continue
        enc = gpt2_tok(truncated, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = gpt2_model(input_ids, labels=input_ids)
            nlls.append(out.loss.item())

    del gpt2_model, gpt2_tok
    torch.cuda.empty_cache()

    if not nlls:
        return float("nan")
    return float(np.exp(np.mean(nlls)))


def compute_distinct_n(texts, n):
    """Fraction of unique n-grams across all texts."""
    total_ngrams = []
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            total_ngrams.append(tuple(words[i : i + n]))
    if not total_ngrams:
        return 0.0
    return len(set(total_ngrams)) / len(total_ngrams)


def compute_rep_rate(texts):
    """Fraction of consecutive repeated tokens."""
    total = 0
    repeated = 0
    for text in texts:
        words = text.split()
        for i in range(1, len(words)):
            total += 1
            if words[i] == words[i - 1]:
                repeated += 1
    return repeated / total if total > 0 else 0.0


def generate_remask(model, tokenizer, prompt_text, device,
                    block_length=256, steps=64, suppress_eos=False):
    """Generate using remasking. Returns generated text."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    eos_ratio = 1.0 if suppress_eos else 0.0

    with torch.no_grad():
        out = generate(
            model, input_ids, attention_mask,
            steps=steps, gen_length=256,
            block_length=block_length, temperature=0.0,
            cfg_scale=0.0, remasking="low_confidence",
            eos_suppress_ratio=eos_ratio,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def generate_sde(model, tokenizer, prompt_text, device, sde_config):
    """SDE generation in continuous space.

    Simplified Euler-Maruyama / Heun solver.
    """
    from dsl_llada.core.dsl_modules import noisy_embedding

    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    B, P = prompt_ids.shape
    gen_length = 256
    seq_len = P + gen_length

    steps = sde_config.get("steps", 32)
    schedule = sde_config.get("schedule", [3, 100])
    noise_scale = sde_config.get("noise_scale", 0.05)
    beta_infer = sde_config.get("beta_infer", 2.5)
    solver = sde_config.get("solver", "heun")

    noise_dim = model.noise_embed.weight.shape[1]

    # Override converter beta for inference
    orig_beta = model.converter.beta.data.clone()
    model.converter.beta.data.fill_(beta_infer)

    try:
        # SNR schedule: log-linear from schedule[0] to schedule[1]
        snr_lo, snr_hi = schedule
        snr_schedule = torch.exp(
            torch.linspace(math.log(snr_lo), math.log(snr_hi), steps + 1, device=device)
        )

        # Initialize: prompt positions get high-SNR embeddings, gen positions get random
        prompt_snr = torch.full((1, P), snr_hi, device=device)
        z_prompt = noisy_embedding(model.noise_embed, prompt_ids, prompt_snr).float()

        # Random init for generation positions
        z_gen = torch.randn(1, gen_length, noise_dim, dtype=torch.float32, device=device)

        for step_idx in range(steps):
            snr_t = snr_schedule[step_idx]
            snr_next = snr_schedule[step_idx + 1]

            # Concatenate prompt + generation embeddings
            z_full = torch.cat([z_prompt, z_gen], dim=1)

            # Forward pass through converter + backbone
            h = model.converter(z_full).to(dtype=torch.bfloat16)
            with torch.no_grad():
                out = model(input_ids=torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device),
                            inputs_embeds=h)
            logits = out.logits.float()

            # Get predicted token probabilities and embedding
            probs = F.softmax(logits[:, P:, :model.noise_embed.weight.shape[0]], dim=-1)
            # Predicted clean embedding: weighted sum of noise_embed
            embed_w = model.noise_embed.weight.float()  # (V, d)
            x_hat = torch.matmul(probs, embed_w)  # (1, gen_length, d)

            # Drift: move towards predicted clean embedding
            dt = (snr_next - snr_t) / snr_hi
            drift = (x_hat - z_gen) * dt.abs()

            if solver == "heun" and step_idx < steps - 1:
                # Predictor step
                z_gen_pred = z_gen + drift + noise_scale * math.sqrt(abs(dt.item())) * torch.randn_like(z_gen)

                # Corrector: re-evaluate at predicted position
                z_full_pred = torch.cat([z_prompt, z_gen_pred], dim=1)
                h_pred = model.converter(z_full_pred).to(dtype=torch.bfloat16)
                with torch.no_grad():
                    out_pred = model(
                        input_ids=torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device),
                        inputs_embeds=h_pred,
                    )
                logits_pred = out_pred.logits.float()
                probs_pred = F.softmax(logits_pred[:, P:, :model.noise_embed.weight.shape[0]], dim=-1)
                x_hat_pred = torch.matmul(probs_pred, embed_w)
                drift_pred = (x_hat_pred - z_gen_pred) * dt.abs()

                # Average drift
                z_gen = z_gen + 0.5 * (drift + drift_pred) + noise_scale * math.sqrt(abs(dt.item())) * torch.randn_like(z_gen)
            else:
                z_gen = z_gen + drift + noise_scale * math.sqrt(abs(dt.item())) * torch.randn_like(z_gen)

        # Final decode: use converter to get logits, argmax
        z_full = torch.cat([z_prompt, z_gen], dim=1)
        h = model.converter(z_full).to(dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(
                input_ids=torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device),
                inputs_embeds=h,
            )
        final_tokens = out.logits[0, P:].argmax(dim=-1)
        text = tokenizer.decode(final_tokens, skip_special_tokens=True)

    finally:
        # Restore original beta
        model.converter.beta.data.copy_(orig_beta)

    nfe = steps * 2 if solver == "heun" else steps
    return text, nfe


def run_t2(model, tokenizer, device, model_key, entry, sde_override=None,
           checkpoint_dir=None):
    """Run T2: generation quality."""
    print("\n" + "=" * 60)
    print("T2: Generation Quality")
    print("=" * 60)

    prompts_data = _load_json("sde_prompts_100.json")
    gen_methods = entry.get("gen_methods", ["remask_free"])

    result = {
        "test": "t2_generation",
        "model": model_key,
        "timestamp": datetime.datetime.now().isoformat(),
        "methods": {},
    }

    for method in gen_methods:
        print(f"\n  Method: {method}")
        texts = []

        for item in tqdm(prompts_data, desc=f"T2 {method}"):
            prompt_text = item["prompt"]
            try:
                if method == "sde":
                    # Lazily attach DSL modules only when SDE is needed
                    if not hasattr(model, "noise_embed"):
                        if checkpoint_dir and entry.get("dsl"):
                            print("    Attaching DSL modules for SDE...")
                            attach_dsl(model, checkpoint_dir, device, entry.get("dsl_config"))
                        else:
                            print("    Skipping SDE: no DSL modules available")
                            break
                    sde_config = dict(entry.get("sde_config", {}))
                    if sde_override:
                        sde_config.update(sde_override)
                    text, _ = generate_sde(model, tokenizer, prompt_text, device, sde_config)
                elif method == "remask_free":
                    text = generate_remask(model, tokenizer, prompt_text, device,
                                           block_length=256, steps=64, suppress_eos=False)
                elif method == "remask_suppress_block32":
                    text = generate_remask(model, tokenizer, prompt_text, device,
                                           block_length=32, steps=64, suppress_eos=True)
                else:
                    text = generate_remask(model, tokenizer, prompt_text, device)

                texts.append(text)

            except Exception as e:
                print(f"    Error on prompt {item.get('id', '?')}: {e}")
                texts.append("")

        if not texts or all(not t for t in texts):
            result["methods"][method] = {"error": "no texts generated"}
            continue

        valid_texts = [t for t in texts if t.strip()]

        # Compute metrics
        print(f"    Computing GenPPL ({len(valid_texts)} texts)...")
        gen_ppl = compute_gen_ppl(valid_texts, device) if valid_texts else float("nan")
        d2 = compute_distinct_n(valid_texts, 2)
        d3 = compute_distinct_n(valid_texts, 3)
        rep = compute_rep_rate(valid_texts)
        avg_len = float(np.mean([len(t.split()) for t in valid_texts])) if valid_texts else 0.0

        # MAUVE (optional) — compare against human-written text (WikiText-103)
        mauve_score = None
        try:
            import mauve
            # Use eval texts_100 as human reference distribution
            ref_data = _load_json("texts_100.json")
            ref_texts = [item["text"] for item in ref_data[:len(valid_texts)]]
            mauve_result = mauve.compute_mauve(
                p_text=ref_texts, q_text=valid_texts,
                device_id=int(str(device).split(":")[-1]) if ":" in str(device) else 0,
                max_text_length=256,
            )
            mauve_score = round(float(mauve_result.mauve), 4)
        except (ImportError, Exception) as e:
            pass

        method_result = {
            "gen_ppl": round(gen_ppl, 2) if not math.isnan(gen_ppl) else None,
            "mauve": mauve_score,
            "d2": round(d2, 4),
            "d3": round(d3, 4),
            "rep_rate": round(rep, 4),
            "avg_len": round(avg_len, 1),
            "n_generated": len(valid_texts),
            "generated_texts": valid_texts,  # Save for post-hoc MAUVE
        }

        if method == "sde":
            sde_config = dict(entry.get("sde_config", {}))
            if sde_override:
                sde_config.update(sde_override)
            method_result["nfe"] = sde_config.get("steps", 32) * (2 if sde_config.get("solver") == "heun" else 1)
            method_result["sde_params"] = sde_config

        result["methods"][method] = method_result
        print(f"    GenPPL={gen_ppl:.2f}, D2={d2:.4f}, D3={d3:.4f}, rep={rep:.4f}, avg_len={avg_len:.0f}")

    save_result(model_key, "t2_generation.json", result)
    return result


# ===================================================================
# Aggregate: summary tables
# ===================================================================

def aggregate_results():
    """Read all results and generate Markdown summary tables."""
    print("\n" + "=" * 60)
    print("Aggregating results")
    print("=" * 60)

    registry = load_registry()
    model_keys = list(registry.keys())

    # Load all results
    all_results = {}
    for mk in model_keys:
        all_results[mk] = {}
        result_dir = os.path.join(EVAL_RESULTS_DIR, mk)
        if not os.path.isdir(result_dir):
            continue
        for fn in os.listdir(result_dir):
            if fn.endswith(".json") and not fn.startswith("meta"):
                with open(os.path.join(result_dir, fn)) as f:
                    all_results[mk][fn.replace(".json", "")] = json.load(f)

    lines = []
    lines.append("# DSL-LLaDA Evaluation Summary")
    lines.append(f"\nGenerated: {datetime.datetime.now().isoformat()}\n")

    # --- Status table ---
    lines.append("## Status\n")
    tests = ["t1_corruption", "t2_generation", "t3_mask", "t4_reasoning", "t5_trip_planning"]
    header = "| Model | " + " | ".join(tests) + " |"
    sep = "|---|" + "|".join(["---"] * len(tests)) + "|"
    lines.append(header)
    lines.append(sep)
    for mk in model_keys:
        row = f"| {mk} |"
        for t in tests:
            has = t in all_results.get(mk, {})
            row += " Y |" if has else " - |"
        lines.append(row)
    lines.append("")

    # --- T1: Random Corruption ---
    lines.append("## T1: Random Corruption\n")
    header = "| Model | Fix@10% | Clean@10% | Fix@30% | Clean@30% | Fix@50% | Clean@50% |"
    sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for mk in model_keys:
        t1 = all_results.get(mk, {}).get("t1_corruption", {})
        subs = t1.get("subtests", {})
        row = f"| {mk}"
        for rate in [10, 30, 50]:
            k = f"random_{rate}"
            d = subs.get(k, {})
            fix = d.get("fix_rate")
            clean = d.get("clean_preserved")
            row += f" | {fix*100:.1f}%" if fix is not None else " | -"
            row += f" | {clean*100:.1f}%" if clean is not None else " | -"
        row += " |"
        lines.append(row)
    lines.append("")

    # --- T1: Semantic Corruption Auto ---
    lines.append("## T1: Semantic Corruption (Auto)\n")
    header = "| Model | Fix@10% | Clean@10% | Fix@30% | Clean@30% |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for mk in model_keys:
        t1 = all_results.get(mk, {}).get("t1_corruption", {})
        subs = t1.get("subtests", {})
        row = f"| {mk}"
        for rate in [10, 30]:
            k = f"semantic_auto_{rate}"
            d = subs.get(k, {})
            fix = d.get("fix_rate")
            clean = d.get("clean_preserved")
            row += f" | {fix*100:.1f}%" if fix is not None else " | -"
            row += f" | {clean*100:.1f}%" if clean is not None else " | -"
        row += " |"
        lines.append(row)
    lines.append("")

    # --- T1: Semantic Corruption Manual ---
    lines.append("## T1: Semantic Corruption (Manual)\n")
    header = "| Model | Fix% | Clean% | Easy | Medium | Hard |"
    sep = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for mk in model_keys:
        t1 = all_results.get(mk, {}).get("t1_corruption", {})
        subs = t1.get("subtests", {})
        sm = subs.get("semantic_manual", {})
        overall = sm.get("overall", {})
        by_diff = sm.get("by_difficulty", {})
        fix = overall.get("fix_rate")
        clean = overall.get("clean_preserved")
        easy = by_diff.get("easy")
        medium = by_diff.get("medium")
        hard = by_diff.get("hard")
        row = f"| {mk}"
        row += f" | {fix*100:.1f}%" if fix is not None else " | -"
        row += f" | {clean*100:.1f}%" if clean is not None else " | -"
        row += f" | {easy*100:.1f}%" if easy is not None else " | -"
        row += f" | {medium*100:.1f}%" if medium is not None else " | -"
        row += f" | {hard*100:.1f}%" if hard is not None else " | -"
        row += " |"
        lines.append(row)
    lines.append("")

    # --- T3: Mask Prediction ---
    lines.append("## T3: Mask Prediction\n")
    header = "| Model | Acc@30% | ECE@30% | Acc@50% | ECE@50% | Acc@70% | ECE@70% |"
    sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for mk in model_keys:
        t3 = all_results.get(mk, {}).get("t3_mask", {})
        rates = t3.get("mask_rates", {})
        row = f"| {mk}"
        for rate in [30, 50, 70]:
            d = rates.get(str(rate), {})
            acc = d.get("accuracy")
            ece = d.get("ece")
            row += f" | {acc*100:.1f}%" if acc is not None else " | -"
            row += f" | {ece:.4f}" if ece is not None else " | -"
        row += " |"
        lines.append(row)
    lines.append("")

    # --- T4: Reasoning ---
    lines.append("## T4: Reasoning\n")
    header = "| Model | GSM8K | MATH |"
    sep = "|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for mk in model_keys:
        t4 = all_results.get(mk, {}).get("t4_reasoning", {})
        gsm = t4.get("gsm8k", {})
        math_r = t4.get("math", {})
        gsm_acc = gsm.get("accuracy")
        math_acc = math_r.get("accuracy")
        row = f"| {mk}"
        row += f" | {gsm_acc*100:.1f}%" if gsm_acc is not None else " | -"
        row += f" | {math_acc*100:.1f}%" if math_acc is not None else " | -"
        row += " |"
        lines.append(row)
    lines.append("")

    # --- T2: Generation Quality ---
    for gen_method in ["sde", "remask_free", "remask_suppress_block32"]:
        lines.append(f"## T2: Generation ({gen_method})\n")
        header = "| Model | GenPPL | MAUVE | D2 | D3 | RepRate | AvgLen |"
        sep = "|---|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for mk in model_keys:
            t2 = all_results.get(mk, {}).get("t2_generation", {})
            methods = t2.get("methods", {})
            m = methods.get(gen_method, {})
            ppl = m.get("gen_ppl")
            mauve_val = m.get("mauve")
            d2 = m.get("d2")
            d3 = m.get("d3")
            rep = m.get("rep_rate")
            avg_len = m.get("avg_len")
            row = f"| {mk}"
            row += f" | {ppl:.2f}" if ppl is not None else " | -"
            row += f" | {mauve_val:.4f}" if mauve_val is not None else " | -"
            row += f" | {d2:.4f}" if d2 is not None else " | -"
            row += f" | {d3:.4f}" if d3 is not None else " | -"
            row += f" | {rep:.4f}" if rep is not None else " | -"
            row += f" | {avg_len:.0f}" if avg_len is not None else " | -"
            row += " |"
            lines.append(row)
        lines.append("")

    summary_text = "\n".join(lines)

    # Save and print
    summary_path = os.path.join(EVAL_RESULTS_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(summary_text)
    print(f"\nSaved to {summary_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified DSL-LLaDA evaluation")
    parser.add_argument("--model_key", type=str, help="Model key from registry.json")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--tests", type=str, default="t1,t2,t3,t4,t5",
                        help="Comma-separated test names to run")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip tests whose result files already exist")
    parser.add_argument("--aggregate", action="store_true",
                        help="Generate summary tables from existing results")
    parser.add_argument("--sde_beta_infer", type=float, default=None,
                        help="Override SDE beta_infer")
    parser.add_argument("--sde_steps", type=int, default=None,
                        help="Override SDE steps")
    parser.add_argument("--sde_noise_scale", type=float, default=None,
                        help="Override SDE noise_scale")

    args = parser.parse_args()

    # --- Aggregate mode ---
    if args.aggregate:
        aggregate_results()
        return

    # --- Eval mode ---
    if not args.model_key:
        parser.error("--model_key is required (or use --aggregate)")

    registry = load_registry()
    if args.model_key not in registry:
        print(f"ERROR: model_key '{args.model_key}' not in registry. Available: {list(registry.keys())}")
        sys.exit(1)

    entry = registry[args.model_key]
    checkpoint = resolve_checkpoint(entry)
    device = f"cuda:{args.gpu}"

    # Parse requested tests
    test_names = [t.strip() for t in args.tests.split(",")]
    test_to_file = {
        "t1": "t1_corruption.json",
        "t2": "t2_generation.json",
        "t3": "t3_mask.json",
        "t4": "t4_reasoning.json",
        "t5": "t5_trip_planning.json",
    }

    if args.skip_existing:
        orig = list(test_names)
        test_names = [t for t in test_names if not result_exists(args.model_key, test_to_file.get(t, ""))]
        skipped = set(orig) - set(test_names)
        if skipped:
            print(f"Skipping existing: {skipped}")
    if not test_names:
        print("All requested tests already exist. Nothing to do.")
        return

    print(f"Model:  {args.model_key} ({entry.get('description', '')})")
    print(f"Ckpt:   {checkpoint}")
    print(f"Device: {device}")
    print(f"Tests:  {test_names}")

    # Load model + tokenizer
    print(f"\nLoading model...")
    t0_load = time.time()
    model = load_model(checkpoint, device)
    tokenizer = load_tokenizer()
    print(f"  Model loaded in {time.time() - t0_load:.1f}s")

    # DSL modules are only needed for T2 SDE generation.
    # For T1/T3/T4 (standard forward pass and remasking), the backbone works
    # with plain token IDs — attaching DSL modules can cause device mismatches.
    # We attach DSL lazily only when T2 SDE is actually run (inside run_t2).
    _is_dsl = entry.get("dsl") and entry.get("type") == "local"

    # Build SDE override dict
    sde_override = {}
    if args.sde_beta_infer is not None:
        sde_override["beta_infer"] = args.sde_beta_infer
    if args.sde_steps is not None:
        sde_override["steps"] = args.sde_steps
    if args.sde_noise_scale is not None:
        sde_override["noise_scale"] = args.sde_noise_scale

    # Run tests (t1, t3, t4, t2 — t2 last because it loads GPT-2)
    t0_eval = time.time()
    tests_run = []
    run_order = ["t1", "t3", "t4", "t5", "t2"]

    for t in run_order:
        if t not in test_names:
            continue
        try:
            if t == "t1":
                run_t1(model, tokenizer, device, args.model_key)
            elif t == "t3":
                run_t3(model, tokenizer, device, args.model_key)
            elif t == "t4":
                run_t4(model, tokenizer, device, args.model_key)
            elif t == "t5":
                t5_methods = ["remask"]
                if _is_dsl:
                    t5_methods.append("sde")
                run_t5(model, tokenizer, device, args.model_key,
                       entry=entry,
                       checkpoint_dir=checkpoint if _is_dsl else None,
                       methods=t5_methods)
            elif t == "t2":
                run_t2(model, tokenizer, device, args.model_key, entry,
                       sde_override=sde_override if sde_override else None,
                       checkpoint_dir=checkpoint if _is_dsl else None)
            tests_run.append(t)
        except Exception as e:
            print(f"\nERROR running {t}: {e}")
            traceback.print_exc()

    duration = time.time() - t0_eval
    d_hash = eval_data_hash()
    write_meta(args.model_key, entry, tests_run, duration, d_hash)

    print(f"\n{'=' * 60}")
    print(f"Done: {args.model_key} | tests={tests_run} | {duration:.0f}s")
    print(f"Results in: {os.path.join(EVAL_RESULTS_DIR, args.model_key)}/")


if __name__ == "__main__":
    main()
