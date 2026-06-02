#!/usr/bin/env python3
"""
Failure Mode Analysis for Masked Diffusion LM Generation Quality.

Computes a suite of quantifiable, reproducible degradation metrics across 5 methods:
  - b1_sde: beta=1 SDE generation
  - original_remask: Original LLaDA remask
  - b1_remask: beta=1 remask
  - xdlm_remask: XDLM remask
  - mdm_cpt_remask: MDM CPT remask

Metrics cover: truncation, word-level repetition, phrase-level repetition,
sentence-level repetition, pattern degeneration, and information density.
"""

import json
import re
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
import statistics

###############################################################################
# 1. Metric Definitions
###############################################################################

def word_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def sent_tokenize(text: str) -> list[str]:
    """Split text into sentences (period/exclamation/question + space or end)."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


# ---------- M1: Truncation / Incompleteness ----------

def metric_truncation(text: str, gen_length: int = 256) -> dict:
    """
    M1: Truncation metrics.
    - word_count: number of words
    - char_count: number of characters
    - is_short: word_count < 20 (severely truncated)
    - incomplete_sentence: last sentence doesn't end with proper punctuation
    - short_ratio: word_count / expected_words (expected ~ gen_length * 0.75 tokens ~ words)
    """
    words = word_tokenize(text)
    wc = len(words)
    cc = len(text.strip())
    is_short = wc < 20
    # Check if text ends with sentence-ending punctuation
    stripped = text.strip()
    incomplete = not bool(re.search(r'[.!?"\']$', stripped)) if stripped else True
    return {
        "word_count": wc,
        "char_count": cc,
        "is_short": is_short,
        "incomplete_sentence": incomplete,
    }


# ---------- M2: Word-level Repetition ----------

def metric_word_repetition(text: str) -> dict:
    """
    M2: Consecutive word repetition.
    - max_consec_repeat: longest run of the same word (e.g., "the the the" = 3)
    - consec_repeat_ratio: fraction of words that are part of a consecutive repeat run >= 2
    - num_consec_runs: number of distinct consecutive repeat runs (length >= 2)
    """
    words = word_tokenize(text)
    if not words:
        return {"max_consec_repeat": 0, "consec_repeat_ratio": 0.0, "num_consec_runs": 0}

    max_run = 1
    cur_run = 1
    repeat_words = 0
    num_runs = 0

    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur_run += 1
        else:
            if cur_run >= 2:
                repeat_words += cur_run
                num_runs += 1
            max_run = max(max_run, cur_run)
            cur_run = 1
    # Handle last run
    if cur_run >= 2:
        repeat_words += cur_run
        num_runs += 1
    max_run = max(max_run, cur_run)

    return {
        "max_consec_repeat": max_run,
        "consec_repeat_ratio": repeat_words / len(words) if words else 0.0,
        "num_consec_runs": num_runs,
    }


# ---------- M3: Phrase-level Repetition (n-gram) ----------

def get_ngrams(words: list[str], n: int) -> list[tuple]:
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def metric_ngram_repetition(text: str, ns=(2, 3, 4, 5)) -> dict:
    """
    M3: N-gram repetition metrics.
    For each n in ns:
    - rep_n: 1 - (unique n-grams / total n-grams)  [0 = no repetition, 1 = all same]
    - max_ngram_freq_n: frequency of the most repeated n-gram
    """
    words = word_tokenize(text)
    result = {}
    for n in ns:
        ngrams = get_ngrams(words, n)
        if not ngrams:
            result[f"rep_{n}"] = 0.0
            result[f"max_ngram_freq_{n}"] = 0
            continue
        counts = Counter(ngrams)
        unique = len(counts)
        total = len(ngrams)
        result[f"rep_{n}"] = 1.0 - unique / total
        result[f"max_ngram_freq_{n}"] = counts.most_common(1)[0][1]
    return result


# ---------- M4: Sentence-level Repetition ----------

def normalize_sentence(s: str) -> str:
    """Normalize for comparison: lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', s.lower().strip())


def metric_sentence_repetition(text: str) -> dict:
    """
    M4: Sentence-level repetition.
    - exact_sent_repeat_ratio: fraction of sentences that are exact duplicates of a previous sentence
    - near_sent_repeat_ratio: fraction of sentences with Jaccard similarity > 0.8 to a previous sentence
    - max_sent_repeat_count: maximum number of times any single sentence appears
    """
    sents = sent_tokenize(text)
    if len(sents) <= 1:
        return {
            "exact_sent_repeat_ratio": 0.0,
            "near_sent_repeat_ratio": 0.0,
            "max_sent_repeat_count": 1,
            "num_sentences": len(sents),
        }

    normed = [normalize_sentence(s) for s in sents]

    # Exact repeats
    counts = Counter(normed)
    exact_repeats = sum(c - 1 for c in counts.values())
    max_sent_count = counts.most_common(1)[0][1]

    # Near repeats (Jaccard > 0.8)
    word_sets = [set(word_tokenize(s)) for s in normed]
    near_repeats = 0
    for i in range(1, len(word_sets)):
        for j in range(i):
            if word_sets[i] and word_sets[j]:
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0 and intersection / union > 0.8:
                    near_repeats += 1
                    break  # count each sentence only once

    return {
        "exact_sent_repeat_ratio": exact_repeats / len(sents),
        "near_sent_repeat_ratio": near_repeats / len(sents),
        "max_sent_repeat_count": max_sent_count,
        "num_sentences": len(sents),
    }


# ---------- M5: Pattern Degeneration ----------

def metric_pattern_degeneration(text: str) -> dict:
    """
    M5: Fixed-pattern degeneration detection.
    - comma_spam_ratio: fraction of characters that are commas
    - dot_spam_ratio: fraction of characters that are periods
    - newline_spam_ratio: fraction of characters that are newlines
    - punctuation_ratio: fraction of all characters that are punctuation
    - longest_punct_run: longest consecutive run of punctuation/whitespace only
    - has_degenerate_pattern: True if any spam ratio > 0.15 or longest_punct_run > 20
    """
    if not text.strip():
        return {
            "comma_spam_ratio": 0.0,
            "dot_spam_ratio": 0.0,
            "newline_spam_ratio": 0.0,
            "punctuation_ratio": 0.0,
            "longest_punct_run": 0,
            "has_degenerate_pattern": True,
        }

    total = len(text)
    comma_count = text.count(',')
    dot_count = text.count('.')
    newline_count = text.count('\n')
    punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\')

    # Longest run of non-alphanumeric characters
    runs = re.findall(r'[^a-zA-Z0-9]+', text)
    longest_punct_run = max((len(r) for r in runs), default=0)

    comma_ratio = comma_count / total
    dot_ratio = dot_count / total
    newline_ratio = newline_count / total
    punct_ratio = punct_count / total

    has_degen = (comma_ratio > 0.15 or dot_ratio > 0.15 or
                 newline_ratio > 0.15 or longest_punct_run > 20)

    return {
        "comma_spam_ratio": comma_ratio,
        "dot_spam_ratio": dot_ratio,
        "newline_spam_ratio": newline_ratio,
        "punctuation_ratio": punct_ratio,
        "longest_punct_run": longest_punct_run,
        "has_degenerate_pattern": has_degen,
    }


# ---------- M6: Information Density ----------

def metric_information_density(text: str) -> dict:
    """
    M6: Information density / lexical diversity.
    - type_token_ratio (TTR): unique words / total words
    - hapax_ratio: words appearing exactly once / total words
    - new_word_rate_second_half: fraction of words in 2nd half that didn't appear in 1st half
    - content_word_ratio: fraction of words that are NOT in a stopword list
    - compression_ratio: len(text) / len(set(text.split())) — crude info density
    """
    words = word_tokenize(text)
    if not words:
        return {
            "type_token_ratio": 0.0,
            "hapax_ratio": 0.0,
            "new_word_rate_second_half": 0.0,
            "content_word_ratio": 0.0,
        }

    counts = Counter(words)
    unique = len(counts)
    total = len(words)
    hapax = sum(1 for c in counts.values() if c == 1)

    ttr = unique / total

    # New word rate in second half
    mid = total // 2
    first_half_vocab = set(words[:mid])
    second_half = words[mid:]
    if second_half:
        new_in_second = sum(1 for w in second_half if w not in first_half_vocab)
        new_word_rate = new_in_second / len(second_half)
    else:
        new_word_rate = 0.0

    # Content word ratio (exclude common English stopwords)
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if', 'while',
        'because', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
        'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
        'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
        'whom', 'whose',
    }
    content_words = sum(1 for w in words if w not in STOPWORDS)
    content_ratio = content_words / total

    return {
        "type_token_ratio": ttr,
        "hapax_ratio": hapax / total,
        "new_word_rate_second_half": new_word_rate,
        "content_word_ratio": content_ratio,
    }


# ---------- M7: Composite Severity Score ----------

def metric_composite_severity(m1, m2, m3, m4, m5, m6) -> dict:
    """
    M7: Composite failure severity score (0-100, higher = worse).
    Combines multiple signals into a single degradation score.

    Components (each 0-1, weighted):
    - truncation_score (0.10): 1 if is_short, 0.5 if incomplete
    - word_repeat_score (0.15): min(1, consec_repeat_ratio * 5)
    - ngram_repeat_score (0.20): min(1, rep_4 * 2)
    - sent_repeat_score (0.15): exact_sent_repeat_ratio + near_sent_repeat_ratio (capped at 1)
    - pattern_degen_score (0.20): 1 if has_degenerate_pattern, else max(comma,dot,newline)*5
    - info_density_score (0.20): 1 - type_token_ratio (low diversity = high score)
    """
    trunc = 0.0
    if m1["is_short"]:
        trunc = 1.0
    elif m1["incomplete_sentence"]:
        trunc = 0.3

    word_rep = min(1.0, m2["consec_repeat_ratio"] * 5)
    ngram_rep = min(1.0, m3.get("rep_4", 0) * 2)
    sent_rep = min(1.0, m4["exact_sent_repeat_ratio"] + m4["near_sent_repeat_ratio"])

    if m5["has_degenerate_pattern"]:
        pattern = 1.0
    else:
        pattern = min(1.0, max(m5["comma_spam_ratio"], m5["dot_spam_ratio"], m5["newline_spam_ratio"]) * 5)

    info = 1.0 - m6["type_token_ratio"]

    score = (0.10 * trunc + 0.15 * word_rep + 0.20 * ngram_rep +
             0.15 * sent_rep + 0.20 * pattern + 0.20 * info)

    return {
        "severity_score": round(score * 100, 2),
        "truncation_component": round(trunc, 4),
        "word_repeat_component": round(word_rep, 4),
        "ngram_repeat_component": round(ngram_rep, 4),
        "sent_repeat_component": round(sent_rep, 4),
        "pattern_degen_component": round(pattern, 4),
        "info_density_component": round(info, 4),
    }


###############################################################################
# 2. Analyze a single text
###############################################################################

def analyze_text(text: str) -> dict:
    m1 = metric_truncation(text)
    m2 = metric_word_repetition(text)
    m3 = metric_ngram_repetition(text)
    m4 = metric_sentence_repetition(text)
    m5 = metric_pattern_degeneration(text)
    m6 = metric_information_density(text)
    m7 = metric_composite_severity(m1, m2, m3, m4, m5, m6)
    return {**m1, **m2, **m3, **m4, **m5, **m6, **m7}


###############################################################################
# 3. Aggregation helpers
###############################################################################

def aggregate_metrics(results: list[dict]) -> dict:
    """Compute mean and std for each numeric metric across a list of per-sample results."""
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], (int, float, bool))]
    agg = {}
    for k in keys:
        vals = [float(r[k]) for r in results]
        agg[k] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "median": round(statistics.median(vals), 4),
            "max": round(max(vals), 4),
            "min": round(min(vals), 4),
        }
    return agg


def compute_failure_rates(results: list[dict]) -> dict:
    """Compute binary failure rates."""
    n = len(results)
    if n == 0:
        return {}
    return {
        "truncated_rate": round(sum(1 for r in results if r["is_short"]) / n, 4),
        "incomplete_rate": round(sum(1 for r in results if r["incomplete_sentence"]) / n, 4),
        "word_repeat_rate": round(sum(1 for r in results if r["max_consec_repeat"] >= 3) / n, 4),
        "severe_ngram_repeat_rate": round(sum(1 for r in results if r.get("rep_4", 0) > 0.3) / n, 4),
        "sentence_repeat_rate": round(sum(1 for r in results if r["max_sent_repeat_count"] >= 3) / n, 4),
        "degenerate_pattern_rate": round(sum(1 for r in results if r["has_degenerate_pattern"]) / n, 4),
        "low_diversity_rate": round(sum(1 for r in results if r["type_token_ratio"] < 0.4) / n, 4),
        "severe_failure_rate": round(sum(1 for r in results if r["severity_score"] > 40) / n, 4),
    }


###############################################################################
# 4. Main
###############################################################################

def main():
    base_dir = Path("/home/ubuntu/efs/RMDM")
    result_dir = base_dir / "eval_results" / "sde_gen_formal"

    # Load prompts
    with open(base_dir / "eval_data" / "sde_prompts_200.json") as f:
        prompts = json.load(f)

    # Build category map
    id2cat = {p["id"]: p["category"] for p in prompts}
    id2prompt = {p["id"]: p["prompt"] for p in prompts}
    categories = sorted(set(id2cat.values()))

    # Method files
    methods = {
        "b1_sde": "prompted_b1_sde_nfe64_gen256.json",
        "original_remask": "prompted_original_remask_nfe64_gen256.json",
        "b1_remask": "prompted_b1_remask_nfe64_gen256.json",
        "xdlm_remask": "prompted_xdlm_remask_nfe64_gen256.json",
        "mdm_cpt_remask": "prompted_mdm_cpt_remask_nfe64_gen256.json",
    }

    all_results = {}
    per_sample_results = {}

    for method_name, filename in methods.items():
        filepath = result_dir / filename
        with open(filepath) as f:
            data = json.load(f)
        texts = data["texts"]
        assert len(texts) == 200, f"{method_name} has {len(texts)} texts, expected 200"

        # Analyze each text
        sample_results = []
        for i, text in enumerate(texts):
            metrics = analyze_text(text)
            metrics["sample_id"] = i
            metrics["category"] = id2cat.get(i, "unknown")
            sample_results.append(metrics)

        per_sample_results[method_name] = sample_results

        # Overall aggregation
        overall_agg = aggregate_metrics(sample_results)
        overall_failures = compute_failure_rates(sample_results)

        # Per-category aggregation
        per_cat = {}
        for cat in categories:
            cat_results = [r for r in sample_results if r["category"] == cat]
            if cat_results:
                per_cat[cat] = {
                    "n": len(cat_results),
                    "metrics": aggregate_metrics(cat_results),
                    "failure_rates": compute_failure_rates(cat_results),
                }

        all_results[method_name] = {
            "n": len(texts),
            "overall_metrics": overall_agg,
            "overall_failure_rates": overall_failures,
            "per_category": per_cat,
        }

    # =========================================================================
    # Print report
    # =========================================================================

    print("=" * 100)
    print("FAILURE MODE ANALYSIS — 5 Methods × 200 Samples")
    print("=" * 100)

    # --- Table 1: Overall Failure Rates ---
    print("\n### Table 1: Overall Failure Rates (fraction of 200 samples)")
    header = f"{'Metric':<35}" + "".join(f"{m:<18}" for m in methods)
    print(header)
    print("-" * len(header))

    failure_keys = [
        "truncated_rate", "incomplete_rate", "word_repeat_rate",
        "severe_ngram_repeat_rate", "sentence_repeat_rate",
        "degenerate_pattern_rate", "low_diversity_rate", "severe_failure_rate"
    ]
    for fk in failure_keys:
        row = f"{fk:<35}"
        for m in methods:
            val = all_results[m]["overall_failure_rates"][fk]
            row += f"{val:<18.4f}"
        print(row)

    # --- Table 2: Key Metric Means ---
    print("\n### Table 2: Key Metric Means (± std)")
    key_metrics = [
        "word_count", "max_consec_repeat", "consec_repeat_ratio",
        "rep_3", "rep_4", "rep_5",
        "max_ngram_freq_4",
        "exact_sent_repeat_ratio", "near_sent_repeat_ratio",
        "comma_spam_ratio", "dot_spam_ratio", "longest_punct_run",
        "type_token_ratio", "hapax_ratio", "new_word_rate_second_half",
        "content_word_ratio",
        "severity_score",
    ]
    header = f"{'Metric':<30}" + "".join(f"{m:<20}" for m in methods)
    print(header)
    print("-" * len(header))
    for mk in key_metrics:
        row = f"{mk:<30}"
        for m in methods:
            agg = all_results[m]["overall_metrics"].get(mk, {})
            mean = agg.get("mean", 0)
            std = agg.get("std", 0)
            row += f"{mean:>7.4f} ± {std:<8.4f}  "
        print(row)

    # --- Table 3: Per-Category Severity Scores ---
    print("\n### Table 3: Mean Severity Score by Category")
    header = f"{'Category':<20}" + "".join(f"{m:<18}" for m in methods)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<20}"
        for m in methods:
            cat_data = all_results[m]["per_category"].get(cat, {})
            if cat_data:
                sev = cat_data["metrics"].get("severity_score", {}).get("mean", 0)
                row += f"{sev:<18.2f}"
            else:
                row += f"{'N/A':<18}"
        print(row)

    # --- Table 4: Per-Category Failure Rates ---
    print("\n### Table 4: Severe Failure Rate by Category")
    header = f"{'Category':<20}" + "".join(f"{m:<18}" for m in methods)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<20}"
        for m in methods:
            cat_data = all_results[m]["per_category"].get(cat, {})
            if cat_data:
                rate = cat_data["failure_rates"].get("severe_failure_rate", 0)
                row += f"{rate:<18.4f}"
            else:
                row += f"{'N/A':<18}"
        print(row)

    # --- Table 5: Degenerate Pattern Rate by Category ---
    print("\n### Table 5: Degenerate Pattern Rate by Category")
    header = f"{'Category':<20}" + "".join(f"{m:<18}" for m in methods)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<20}"
        for m in methods:
            cat_data = all_results[m]["per_category"].get(cat, {})
            if cat_data:
                rate = cat_data["failure_rates"].get("degenerate_pattern_rate", 0)
                row += f"{rate:<18.4f}"
            else:
                row += f"{'N/A':<18}"
        print(row)

    # --- Table 6: Per-Category Key Metrics Detail ---
    for cat in categories:
        print(f"\n### Category: {cat}")
        detail_metrics = [
            "word_count", "type_token_ratio", "rep_4", "consec_repeat_ratio",
            "exact_sent_repeat_ratio", "comma_spam_ratio", "dot_spam_ratio",
            "new_word_rate_second_half", "severity_score"
        ]
        header = f"{'Metric':<30}" + "".join(f"{m:<20}" for m in methods)
        print(header)
        print("-" * len(header))
        for mk in detail_metrics:
            row = f"{mk:<30}"
            for m in methods:
                cat_data = all_results[m]["per_category"].get(cat, {})
                if cat_data:
                    agg = cat_data["metrics"].get(mk, {})
                    mean = agg.get("mean", 0)
                    row += f"{mean:<20.4f}"
                else:
                    row += f"{'N/A':<20}"
            print(row)

    # =========================================================================
    # Save full results
    # =========================================================================
    output = {
        "description": "Failure mode analysis for masked diffusion LM generation quality",
        "metric_definitions": {
            "M1_truncation": {
                "word_count": "Number of words in generated text",
                "char_count": "Number of characters",
                "is_short": "True if word_count < 20 (severely truncated)",
                "incomplete_sentence": "True if text doesn't end with sentence-ending punctuation",
            },
            "M2_word_repetition": {
                "max_consec_repeat": "Length of longest consecutive identical word run",
                "consec_repeat_ratio": "Fraction of words in consecutive repeat runs (length >= 2)",
                "num_consec_runs": "Number of distinct consecutive repeat runs",
            },
            "M3_ngram_repetition": {
                "rep_n": "1 - (unique n-grams / total n-grams), for n in {2,3,4,5}. 0 = no repetition, 1 = fully repeated",
                "max_ngram_freq_n": "Frequency of the most common n-gram",
            },
            "M4_sentence_repetition": {
                "exact_sent_repeat_ratio": "Fraction of sentences that are exact duplicates of a previous sentence",
                "near_sent_repeat_ratio": "Fraction of sentences with Jaccard similarity > 0.8 to a previous sentence",
                "max_sent_repeat_count": "Maximum number of times any single sentence appears",
            },
            "M5_pattern_degeneration": {
                "comma_spam_ratio": "Fraction of characters that are commas",
                "dot_spam_ratio": "Fraction of characters that are periods",
                "newline_spam_ratio": "Fraction of characters that are newlines",
                "punctuation_ratio": "Fraction of characters that are punctuation",
                "longest_punct_run": "Longest consecutive non-alphanumeric character run",
                "has_degenerate_pattern": "True if any spam ratio > 0.15 or longest_punct_run > 20",
            },
            "M6_information_density": {
                "type_token_ratio": "Unique words / total words (lexical diversity)",
                "hapax_ratio": "Words appearing exactly once / total words",
                "new_word_rate_second_half": "Fraction of words in 2nd half not seen in 1st half",
                "content_word_ratio": "Fraction of words that are NOT stopwords",
            },
            "M7_composite_severity": {
                "severity_score": "Composite 0-100 score (higher = worse degradation)",
                "components": "truncation(0.10) + word_repeat(0.15) + ngram_repeat(0.20) + sent_repeat(0.15) + pattern_degen(0.20) + info_density(0.20)",
            },
        },
        "results_by_method": all_results,
    }

    out_path = result_dir / "failure_mode_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to: {out_path}")

    # =========================================================================
    # Worst examples per method
    # =========================================================================
    print("\n" + "=" * 100)
    print("TOP-3 WORST SAMPLES PER METHOD (by severity_score)")
    print("=" * 100)
    for method_name in methods:
        samples = per_sample_results[method_name]
        worst = sorted(samples, key=lambda x: -x["severity_score"])[:3]
        print(f"\n--- {method_name} ---")
        for w in worst:
            print(f"  #{w['sample_id']} ({w['category']}): severity={w['severity_score']:.1f}, "
                  f"rep4={w.get('rep_4',0):.3f}, TTR={w['type_token_ratio']:.3f}, "
                  f"maxConsec={w['max_consec_repeat']}, sentRepeat={w['exact_sent_repeat_ratio']:.3f}, "
                  f"degenPattern={w['has_degenerate_pattern']}")


if __name__ == "__main__":
    main()
