"""LLM-as-judge pairwise evaluation for summarization eval outputs.

Uses Azure OpenAI (GPT-5.4) to pairwise-compare two summaries against a
reference on four axes: factuality, coverage, fluency, conciseness, plus an
overall preference. Positions (A/B) are randomized per sample to remove bias.

Input:  two merged per-config JSONs with per-sample "generated" + "reference".
Output: JSON with per-sample judgements + aggregate win rates, written to
        eval_results/summarization/analysis/judge__{dataset}_nfe{nfe}__{A}_vs_{B}.json

Usage (requires .env.local to be sourced):
    set -a; source .env.local; set +a
    python dsl_llada/judge_summarization.py \
        --dataset xsum --nfe 64 \
        --a b1_sde --b original_remask_eosInf_b32 \
        --n 100 --concurrency 8
"""
import argparse
import asyncio
import json
import os
import random
import re
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(_root, "eval_results", "summarization")
OUTDIR = os.path.join(DIR, "analysis")


SYSTEM_PROMPT = """\
You are a strict expert evaluator of summarization quality. You compare two
candidate summaries (A and B) against a reference summary of a source document.

Score each candidate on four axes, each from 1 (terrible) to 5 (excellent):
  - factuality: does the summary avoid hallucinations and contradictions vs the reference?
  - coverage: does it cover the key points that the reference covers?
  - fluency: is it grammatical, coherent, and well-formed English?
  - conciseness: is it appropriately concise (neither truncated nor bloated)?

Then choose an OVERALL preference: "A", "B", or "tie".

Respond with ONLY a single-line JSON object (no prose, no markdown):
{"a": {"factuality": int, "coverage": int, "fluency": int, "conciseness": int},
 "b": {"factuality": int, "coverage": int, "fluency": int, "conciseness": int},
 "overall": "A"|"B"|"tie",
 "rationale": "<=1 short sentence"}"""


def make_user_prompt(reference, cand_a, cand_b):
    return (
        f"REFERENCE SUMMARY:\n{reference}\n\n"
        f"=== Candidate A ===\n{cand_a}\n\n"
        f"=== Candidate B ===\n{cand_b}\n\n"
        "Score both candidates and pick the overall preference. "
        "Return the single-line JSON now."
    )


def load_samples(dataset, method_tag, nfe):
    path = os.path.join(DIR, f"{dataset}_{method_tag}_nfe{nfe}.json")
    d = json.load(open(path))
    return {s["id"]: s for s in d["samples"]}


def parse_response(text):
    if not text:
        return None
    # find first {...} JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "overall" not in obj or obj["overall"] not in ("A", "B", "tie"):
        return None
    return obj


async def call_judge_once(client, deployment, sys_prompt, user_prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=400,
                model=deployment,
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [judge] giving up after {max_retries}: {e}")
                return None, None
            await asyncio.sleep(2 ** attempt)
    return None, None


async def judge_one(client, deployment, sid, ref, gen_a, gen_b, swap):
    """Returns (record dict, usage) where record stores FLIPPED-back 'a'/'b'
    so 'a' always refers to original config A and 'b' to original config B."""
    cand_a, cand_b = (gen_b, gen_a) if swap else (gen_a, gen_b)
    user = make_user_prompt(ref, cand_a, cand_b)
    raw, usage = await call_judge_once(client, deployment, SYSTEM_PROMPT, user)
    parsed = parse_response(raw) if raw else None
    rec = {
        "id": sid,
        "swap": swap,
        "raw": raw,
        "parsed": parsed,
    }
    if parsed:
        # un-swap so a/b keys refer to original A/B
        a_key, b_key = ("b", "a") if swap else ("a", "b")
        rec["a_scores"] = parsed[a_key]
        rec["b_scores"] = parsed[b_key]
        overall = parsed["overall"]
        if swap and overall in ("A", "B"):
            overall = "B" if overall == "A" else "A"
        rec["overall"] = overall
        rec["rationale"] = parsed.get("rationale", "")
    return rec, usage


async def run(dataset, nfe, tag_a, tag_b, n, concurrency, seed=42, resume=True):
    from openai import AsyncAzureOpenAI
    os.makedirs(OUTDIR, exist_ok=True)

    idx_a = load_samples(dataset, tag_a, nfe)
    idx_b = load_samples(dataset, tag_b, nfe)
    common = sorted(set(idx_a) & set(idx_b))
    rng = random.Random(seed)
    rng.shuffle(common)
    common = common[:n]
    print(f"[judge] {dataset}/nfe={nfe}  A={tag_a}  B={tag_b}  n={len(common)}  concurrency={concurrency}")

    out_file = os.path.join(OUTDIR, f"judge__{dataset}_nfe{nfe}__{tag_a}_vs_{tag_b}.json")
    existing = {}
    if resume and os.path.exists(out_file):
        d = json.load(open(out_file))
        existing = {r["id"]: r for r in d.get("records", []) if r.get("parsed")}
        print(f"  resuming: {len(existing)} already judged")

    client = AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    sem = asyncio.Semaphore(concurrency)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    lock = asyncio.Lock()
    records = list(existing.values())
    t0 = time.time()
    done = 0
    done_lock = asyncio.Lock()

    async def worker(sid):
        nonlocal done
        if sid in existing:
            return
        ref = idx_a[sid]["reference"]
        swap = bool(rng.getrandbits(1))  # fresh rng per seed; fine that it drifts
        async with sem:
            rec, usage = await judge_one(
                client, deployment, sid, ref, idx_a[sid]["generated"], idx_b[sid]["generated"], swap
            )
        async with lock:
            records.append(rec)
            if usage is not None:
                total_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                total_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        async with done_lock:
            done += 1
            if done % 20 == 0 or done == len(common) - len(existing):
                dt = time.time() - t0
                print(f"  progress: {done}/{len(common) - len(existing)}  ({dt:.0f}s)")

    rng2 = random.Random(seed + 1)
    todo = [sid for sid in common if sid not in existing]
    tasks = [asyncio.create_task(worker(sid)) for sid in todo]
    if tasks:
        await asyncio.gather(*tasks)

    # aggregate over successfully parsed records
    parsed = [r for r in records if r.get("parsed")]
    n_ok = len(parsed)
    overall_counts = {"A": 0, "B": 0, "tie": 0}
    axes_a = {k: [] for k in ("factuality", "coverage", "fluency", "conciseness")}
    axes_b = {k: [] for k in ("factuality", "coverage", "fluency", "conciseness")}
    for r in parsed:
        overall_counts[r["overall"]] += 1
        for k in axes_a:
            axes_a[k].append(r["a_scores"].get(k, 0))
            axes_b[k].append(r["b_scores"].get(k, 0))
    agg = {
        "n_parsed": n_ok,
        "n_total": len(common),
        "overall_counts": overall_counts,
        "overall_pct": {k: round(v / max(n_ok, 1) * 100, 1)
                        for k, v in overall_counts.items()},
        "a_means": {k: round(sum(v) / max(len(v), 1), 2) for k, v in axes_a.items()},
        "b_means": {k: round(sum(v) / max(len(v), 1), 2) for k, v in axes_b.items()},
        "usage": total_usage,
        "wall_s": round(time.time() - t0, 1),
    }
    out = {
        "dataset": dataset,
        "nfe": nfe,
        "tag_a": tag_a,
        "tag_b": tag_b,
        "aggregate": agg,
        "records": records,
    }
    with open(out_file, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[judge] done: parsed {n_ok}/{len(common)}  "
          f"A={agg['overall_pct']['A']}%  B={agg['overall_pct']['B']}%  tie={agg['overall_pct']['tie']}%")
    print(f"  usage: prompt={total_usage['prompt_tokens']}  completion={total_usage['completion_tokens']}")
    print(f"  -> {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--nfe", type=int, required=True)
    p.add_argument("--a", dest="tag_a", required=True)
    p.add_argument("--b", dest="tag_b", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_resume", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args.dataset, args.nfe, args.tag_a, args.tag_b,
                    args.n, args.concurrency, args.seed,
                    resume=not args.no_resume))


if __name__ == "__main__":
    main()
