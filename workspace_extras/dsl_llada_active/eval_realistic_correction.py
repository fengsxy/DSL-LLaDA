"""Selective-correction evaluation.

This supplements the random-token corruption table with more reviewer-facing
corruption types: spelling/OCR/homophone edits, entity/number/factual swaps,
and semantic-neighbor substitutions. It reuses the repository registry and
model loading code from eval_unified.py.
"""
import argparse
import datetime
import difflib
import json
import os
import random
import re
import sys

import numpy as np
import torch
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)

from eval_unified import (  # noqa: E402
    attach_dsl,
    corrupt_tokens_random,
    eval_corruption_single,
    load_model,
    load_registry,
    load_tokenizer,
    resolve_checkpoint,
)


def _load_json(name):
    with open(os.path.join(_root, "eval_data", name)) as f:
        return json.load(f)


def _make_model(model_key, gpu):
    registry = load_registry()
    if model_key not in registry:
        raise SystemExit(f"Unknown model_key={model_key}; available={sorted(registry)}")

    entry = registry[model_key]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(entry)
    print(f"Loading {model_key} from {checkpoint}", flush=True)
    if entry.get("dsl"):
        dsl_config = entry.get("dsl_config", {})
        os.environ["DSL_NOISE_DIM"] = str(dsl_config.get("noise_dim", 48))
        os.environ["DSL_BETA_INIT"] = str(dsl_config.get("beta_init", 5.0))
        os.environ["DSL_NOISE_INIT"] = str(dsl_config.get("noise_init", "random"))
    model = load_model(checkpoint, device)
    if entry.get("dsl"):
        attach_dsl(model, checkpoint, device, entry.get("dsl_config", {}))
    model.eval()
    tokenizer = load_tokenizer()
    return model, tokenizer, device, entry


def _noisy_edit(word, rng):
    if len(word) < 5:
        return word
    choices = []
    if len(word) > 5:
        i = rng.randint(1, len(word) - 2)
        choices.append(word[:i] + word[i + 1:])
    i = rng.randint(1, len(word) - 2)
    choices.append(word[:i] + word[i + 1] + word[i] + word[i + 2:])
    replacements = [("rn", "m"), ("cl", "d"), ("vv", "w"), ("0", "O"), ("1", "l")]
    for old, new in replacements:
        if old in word:
            choices.append(word.replace(old, new, 1))
    return rng.choice(choices)


def build_noisy_edit_token_cases(tokenizer, texts_data, limit, rate, seed):
    """Generate tokenizer-aware spelling/OCR edit cases from eval texts.

    We only keep edits that re-tokenize to exactly one token, making the
    correction metric identical to the random-token corruption protocol while
    using plausible character-level edits instead of uniform random IDs.
    """
    rng = random.Random(seed)
    cases = []
    for item in texts_data:
        tokens = item["tokens"]
        candidate_idx = [
            i for i, tid in enumerate(tokens)
            if re.match(r"^[A-Za-z][A-Za-z-]{3,}$", tokenizer.decode([tid]).strip())
        ]
        if not candidate_idx:
            continue
        target_n = max(1, int(len(candidate_idx) * rate))
        chosen = list(candidate_idx)
        rng.shuffle(chosen)
        corrupted = list(tokens)
        corruptions = []
        for idx in chosen:
            raw_old = tokenizer.decode([tokens[idx]])
            old = raw_old.strip()
            prefix = raw_old[:len(raw_old) - len(raw_old.lstrip())]
            replacement = None
            replacement_ids = None
            for _ in range(8):
                new = _noisy_edit(old, rng)
                if new == old:
                    continue
                candidates = [prefix + new, new]
                if not prefix:
                    candidates.append(" " + new)
                for text in dict.fromkeys(candidates):
                    new_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    if len(new_ids) == 1 and new_ids[0] != tokens[idx]:
                        replacement = text
                        replacement_ids = new_ids
                        break
                if replacement_ids is not None:
                    break
            if replacement_ids is None:
                continue
            corrupted[idx] = int(replacement_ids[0])
            corruptions.append({
                "token_index": idx,
                "original_token": int(tokens[idx]),
                "replacement_token": int(replacement_ids[0]),
                "original_text": old,
                "replacement_text": replacement.strip(),
            })
            if len(corruptions) >= target_n:
                break
        if not corruptions:
            continue
        cases.append({
            "id": item["id"],
            "suite": "noisy_edit",
            "category": "spelling_ocr",
            "gold_tokens": tokens,
            "corrupted_tokens": corrupted,
            "positions": [c["token_index"] for c in corruptions],
            "corruptions": corruptions,
        })
        if len(cases) >= limit:
            break
    return cases


def build_manual_cases():
    cases = []

    noisy_edit_pairs = [
        ("Their car was parked outside the museum during the storm.",
         "There car was parked outside the museum during the storm."),
        ("The report came from the committee after months of review.",
         "The report came form the committee after months of review."),
        ("The new policy had a strong effect on public schools.",
         "The new policy had a strong affect on public schools."),
        ("The principal announced the schedule before classes began.",
         "The principle announced the schedule before classes began."),
        ("The team decided to accept the revised contract.",
         "The team decided to except the revised contract."),
        ("The jacket was too loose after the athlete lost weight.",
         "The jacket was too lose after the athlete lost weight."),
        ("The hikers passed the old bridge before sunset.",
         "The hikers past the old bridge before sunset."),
        ("The weather changed quickly along the northern coast.",
         "The whether changed quickly along the northern coast."),
        ("The room was quiet after the lecture ended.",
         "The room was quite after the lecture ended."),
        ("The result was better than the analysts expected.",
         "The result was better then the analysts expected."),
        ("Modern cameras can record high resolution video.",
         "Modem cameras can record high resolution video."),
        ("The clear signal reached the receiver without delay.",
         "The dear signal reached the receiver without delay."),
        ("The farmer stored grain in the barn near the road.",
         "The farmer stored grain in the bam near the road."),
        ("The child drew a small clock on the worksheet.",
         "The child drew a small dock on the worksheet."),
        ("The nurse wrote the dosage on the chart.",
         "The nurse wrote the dosage on the chait."),
        ("The company will hire more engineers next year.",
         "The company will higher more engineers next year."),
        ("The council reviewed the site plan on Tuesday.",
         "The counsel reviewed the site plan on Tuesday."),
        ("The scientist cited several papers in the appendix.",
         "The scientist sighted several papers in the appendix."),
        ("The library shelf was filled with reference books.",
         "The library shelve was filled with reference books."),
        ("The mayor said the city would build a new station.",
         "The mayor said the city wood build a new station."),
        ("The medicine should reduce the patient's pain.",
         "The medicine should deduce the patient's pain."),
        ("The teacher asked students to write a summary.",
         "The teacher asked students to right a summary."),
        ("The aircraft began its descent after crossing the river.",
         "The aircraft began its decent after crossing the river."),
        ("The editor checked every line of the article.",
         "The editor checked every lime of the article."),
        ("The court heard the appeal in a public session.",
         "The court heard the apple in a public session."),
        ("The data source was listed below the table.",
         "The data sauce was listed below the table."),
        ("The plant needs enough light to grow indoors.",
         "The plant needs enough flight to grow indoors."),
        ("The museum opened a new gallery for visitors.",
         "The museum opened a new galley for visitors."),
        ("The doctor noted a mild fever in the record.",
         "The doctor noted a mild fewer in the record."),
        ("The model learned the correct label from context.",
         "The model learned the collect label from context."),
    ]
    for idx, (original, corrupted) in enumerate(noisy_edit_pairs):
        cases.append({
            "id": f"noisy_edit_{idx}",
            "suite": "noisy_edit_manual",
            "category": "spelling_ocr_homophone",
            "original_text": original,
            "corrupted_text": corrupted,
            "corruptions": [],
        })

    for item in _load_json("semantic_corruption_manual.json"):
        ctype = "semantic_neighbor"
        category = item.get("category", "semantic")
        corruption_types = [
            c.get("corruption_type", "") for c in item.get("corruptions", [])
        ]
        if any("number" in x or "date" in x for x in corruption_types):
            ctype = "entity_number"
        elif category in {"geography", "history", "science", "math"}:
            ctype = "entity_number"
        cases.append({
            "id": f"manual_{item['id']}",
            "suite": ctype,
            "category": category,
            "original_text": item["original_text"],
            "corrupted_text": item["corrupted_text"],
            "corruptions": item.get("corruptions", []),
        })

    for item in _load_json("correction_benchmark_50.json"):
        category = item.get("category", "factual")
        suite = "entity_number" if category in {"math", "science", "geography", "history"} else "semantic_neighbor"
        cases.append({
            "id": f"bench_{item['id']}",
            "suite": suite,
            "category": category,
            "original_text": item["original"],
            "corrupted_text": item["corrupted"],
            "corruptions": item.get("error_descriptions", []),
        })

    return cases


def token_diff_positions(orig_ids, corr_ids):
    """Return aligned corrupted and clean positions in corrupted-token space."""
    matcher = difflib.SequenceMatcher(a=orig_ids, b=corr_ids, autojunk=False)
    corrupt_pairs = []
    clean_pairs = []
    skipped_ops = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for oi, cj in zip(range(i1, i2), range(j1, j2)):
                clean_pairs.append((oi, cj))
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for oi, cj in zip(range(i1, i2), range(j1, j2)):
                corrupt_pairs.append((oi, cj))
        else:
            skipped_ops += 1

    return corrupt_pairs, clean_pairs, skipped_ops


def eval_text_cases(model, tokenizer, device, cases, max_length):
    per_case = []
    for case in tqdm(cases, desc="text corruption cases"):
        orig = tokenizer(
            case["original_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        corr = tokenizer(
            case["corrupted_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if len(orig) < 2 or len(corr) < 2:
            continue
        corrupt_pairs, clean_pairs, skipped_ops = token_diff_positions(orig, corr)
        if not corrupt_pairs:
            continue

        input_ids = torch.tensor(corr, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_ids=input_ids).logits[0].argmax(dim=-1).detach().cpu().tolist()

        fixed = sum(1 for oi, cj in corrupt_pairs if cj < len(pred) and pred[cj] == orig[oi])
        clean_ok = sum(1 for oi, cj in clean_pairs if cj < len(pred) and pred[cj] == orig[oi])
        fix = fixed / len(corrupt_pairs)
        clean = clean_ok / len(clean_pairs) if clean_pairs else 1.0
        per_case.append({
            "id": case["id"],
            "suite": case["suite"],
            "category": case.get("category", "unknown"),
            "fix_rate": fix,
            "clean_preserved": clean,
            "n_corrupt_tokens": len(corrupt_pairs),
            "n_clean_tokens": len(clean_pairs),
            "skipped_diff_ops": skipped_ops,
        })
    return per_case


def eval_random_suite(model, texts_data, device, rates, seed):
    per_case = []
    for rate in rates:
        for item in tqdm(texts_data, desc=f"random {int(rate * 100)}%"):
            gold_tokens = item["tokens"]
            corrupted, positions, _ = corrupt_tokens_random(gold_tokens, rate, seed + item["id"])
            tokens_tensor = torch.tensor(corrupted, dtype=torch.long)
            fix, clean = eval_corruption_single(model, tokens_tensor, gold_tokens, positions, device)
            per_case.append({
                "id": f"random_{int(rate * 100)}_{item['id']}",
                "suite": f"random_token_{int(rate * 100)}",
                "category": "random_token",
                "fix_rate": fix,
                "clean_preserved": clean,
                "n_corrupt_tokens": len(positions),
                "n_clean_tokens": len(gold_tokens) - len(positions),
                "skipped_diff_ops": 0,
            })
    return per_case


def eval_token_cases(model, cases, device):
    per_case = []
    for case in tqdm(cases, desc="token corruption cases"):
        gold_tokens = case["gold_tokens"]
        corrupted = case["corrupted_tokens"]
        positions = case["positions"]
        if not positions:
            continue
        tokens_tensor = torch.tensor(corrupted, dtype=torch.long)
        fix, clean = eval_corruption_single(model, tokens_tensor, gold_tokens, positions, device)
        per_case.append({
            "id": f"{case['suite']}_{case['id']}",
            "suite": case["suite"],
            "category": case.get("category", "unknown"),
            "fix_rate": fix,
            "clean_preserved": clean,
            "n_corrupt_tokens": len(positions),
            "n_clean_tokens": len(gold_tokens) - len(positions),
            "skipped_diff_ops": 0,
        })
    return per_case


def aggregate(per_case):
    grouped = {}
    for row in per_case:
        grouped.setdefault(row["suite"], []).append(row)

    summary = {}
    for suite, rows in sorted(grouped.items()):
        fix = float(np.mean([r["fix_rate"] for r in rows]))
        clean = float(np.mean([r["clean_preserved"] for r in rows]))
        overcorrect = 1.0 - clean
        summary[suite] = {
            "fix_rate": round(fix, 4),
            "clean_preserved": round(clean, 4),
            "overcorrect": round(overcorrect, 4),
            "selective_score": round(fix - overcorrect, 4),
            "n_cases": len(rows),
            "n_corrupt_tokens": int(sum(r["n_corrupt_tokens"] for r in rows)),
            "n_clean_tokens": int(sum(r["n_clean_tokens"] for r in rows)),
            "skipped_diff_ops": int(sum(r["skipped_diff_ops"] for r in rows)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit_text_cases", type=int, default=100)
    parser.add_argument("--texts_file", default="texts_100.json",
                        help="Eval text JSON under eval_data/ or an absolute path")
    parser.add_argument("--typo_rate", type=float, default=0.10)
    parser.add_argument("--random_rates", default="0.10,0.30,0.50")
    parser.add_argument("--suites", default="random,noisy_edit,manual",
                        help="Comma-separated suites: random,noisy_edit,manual")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer, device, entry = _make_model(args.model_key, args.gpu)
    random_rates = [float(x) for x in args.random_rates.split(",") if x.strip()]
    suites = {x.strip() for x in args.suites.split(",") if x.strip()}

    texts_path = args.texts_file
    if not os.path.isabs(texts_path):
        texts_path = os.path.join(_root, "eval_data", texts_path)
    with open(texts_path) as f:
        texts_data = json.load(f)[:args.limit_text_cases]
    per_case = []
    if "random" in suites:
        per_case.extend(eval_random_suite(model, texts_data, device, random_rates, args.seed))

    if "noisy_edit" in suites:
        noisy_edit_cases = build_noisy_edit_token_cases(
            tokenizer, texts_data, args.limit_text_cases, args.typo_rate, args.seed
        )
        per_case.extend(eval_token_cases(model, noisy_edit_cases, device))

    if "manual" in suites:
        text_cases = build_manual_cases()
        per_case.extend(eval_text_cases(model, tokenizer, device, text_cases, args.max_length))

    payload = {
        "test": "selective_correction",
        "model": args.model_key,
        "description": entry.get("description", ""),
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": args.seed,
        "texts_file": args.texts_file,
        "max_length": args.max_length,
        "random_rates": random_rates,
        "typo_rate": args.typo_rate,
        "suites": sorted(suites),
        "summary": aggregate(per_case),
        "per_case": per_case,
    }

    out = args.out or os.path.join(
        _root, "eval_results", args.model_key, "realistic_correction.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
