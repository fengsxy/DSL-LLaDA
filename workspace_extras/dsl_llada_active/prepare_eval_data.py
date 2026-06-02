#!/usr/bin/env python3
"""Prepare static eval datasets for DSL-LLaDA unified evaluation.

Generates four JSON files in eval_data/:
  - texts_100.json       : 100 WikiText texts, pre-tokenized with LLaDA tokenizer
  - sde_prompts_100.json : 100 generation prompts (5 templates x 20 topics)
  - gsm8k_100.json       : 100 fixed GSM8K problems (seed=42 sample)
  - math_100.json         : 100 fixed MATH problems (seed=42 sample)

Usage:
    python dsl_llada/prepare_eval_data.py
    python dsl_llada/prepare_eval_data.py --only texts --n_texts 1000
"""

import argparse
import json
import os
import random
import re


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eval_data")


def _is_wikitext_heading(text):
    stripped = text.strip()
    return stripped.startswith("=") and stripped.endswith("=")


def prepare_texts(n_texts=100, split="test", dataset_name="Salesforce/wikitext",
                  dataset_config="wikitext-103-raw-v1", out_name=None):
    """Prepare n non-heading WikiText texts, tokenized with LLaDA."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(dataset_name, dataset_config, split=split)
    texts = []
    for item in ds:
        text = item.get("text", "").strip()
        if not text or _is_wikitext_heading(text):
            continue
        texts.append(text)
        if len(texts) >= n_texts:
            break
    if len(texts) < n_texts:
        raise RuntimeError(f"Only found {len(texts)} usable texts, requested {n_texts}")

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    records = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        records.append({
            "id": i,
            "text": text,
            "tokens": tokens,
            "n_tokens": len(tokens),
        })

    out_name = out_name or f"texts_{n_texts}.json"
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"{out_name}: {len(records)} entries written to {out_path}")
    return records


def prepare_texts_100():
    """Backward-compatible wrapper for the original 100-text eval file."""
    return prepare_texts(n_texts=100, out_name="texts_100.json")


def prepare_sde_prompts_100():
    """Generate 100 SDE prompts from 5 templates x 20 topics."""
    templates = [
        "Write a short paragraph about {topic}.",
        "Explain {topic} in simple terms.",
        "Describe {topic} and its significance.",
        "What is {topic}?",
        "Summarize the key aspects of {topic}.",
    ]

    topics = [
        "photosynthesis",
        "the solar system",
        "machine learning",
        "the water cycle",
        "ancient Rome",
        "quantum mechanics",
        "climate change",
        "the human brain",
        "the Industrial Revolution",
        "DNA and genetics",
        "black holes",
        "democracy",
        "the internet",
        "plate tectonics",
        "the Renaissance",
        "antibiotics",
        "artificial intelligence",
        "the periodic table",
        "evolution",
        "cryptocurrency",
    ]

    records = []
    idx = 0
    for template in templates:
        for topic in topics:
            records.append({
                "id": idx,
                "template": template,
                "topic": topic,
                "prompt": template.format(topic=topic),
            })
            idx += 1

    out_path = os.path.join(OUT_DIR, "sde_prompts_100.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"sde_prompts_100.json: {len(records)} entries written to {out_path}")
    return records


def prepare_gsm8k_100():
    """Sample 100 GSM8K test problems with seed=42."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")

    random.seed(42)
    indices = random.sample(range(len(ds)), 100)

    records = []
    for i, orig_idx in enumerate(indices):
        item = ds[orig_idx]
        # Extract gold answer from "#### X" pattern
        answer_text = item["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        gold = match.group(1).strip() if match else ""

        records.append({
            "id": i,
            "original_idx": orig_idx,
            "question": item["question"],
            "gold_answer": gold,
        })

    out_path = os.path.join(OUT_DIR, "gsm8k_100.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"gsm8k_100.json: {len(records)} entries written to {out_path}")
    return records


def prepare_math_100():
    """Sample 100 MATH test problems (all 7 configs merged) with seed=42."""
    from datasets import load_dataset

    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    # Merge all configs
    all_items = []
    for config in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        for j in range(len(ds)):
            all_items.append(ds[j])

    random.seed(42)
    indices = random.sample(range(len(all_items)), 100)

    records = []
    for i, orig_idx in enumerate(indices):
        item = all_items[orig_idx]

        # Extract gold answer from last \boxed{...} in solution
        solution = item["solution"]
        # Find all \boxed{...} occurrences, handling nested braces
        gold = _extract_last_boxed(solution)

        records.append({
            "id": i,
            "original_idx": orig_idx,
            "question": item["problem"],
            "gold_answer": gold,
        })

    out_path = os.path.join(OUT_DIR, "math_100.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"math_100.json: {len(records)} entries written to {out_path}")
    return records


def _extract_last_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    # Find all positions of \boxed{
    positions = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not positions:
        return ""

    # Take the last occurrence
    start = positions[-1]
    # Find the opening brace
    brace_start = text.index("{", start)

    # Match braces to find the closing one
    depth = 0
    for idx in range(brace_start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : idx]
    # Fallback: return everything after \boxed{ to end
    return text[brace_start + 1 :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "texts"], default="all")
    parser.add_argument("--n_texts", type=int, default=100)
    parser.add_argument("--texts_split", default="test")
    parser.add_argument("--texts_out", default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Preparing eval_data/ static datasets")
    print("=" * 60)

    if args.only in ("all", "texts"):
        prepare_texts(n_texts=args.n_texts, split=args.texts_split, out_name=args.texts_out)
    if args.only == "all":
        prepare_sde_prompts_100()
        prepare_gsm8k_100()
        prepare_math_100()

    print("=" * 60)
    print("Done. All files written to", OUT_DIR)


if __name__ == "__main__":
    main()
