"""Compute BERTScore (roberta-large) for summarization result JSONs.

Reads each eval_results/summarization/*.json with per-sample records, appends
BERTScore P/R/F1 per sample AND aggregate. Skips already-scored files.

Usage:
    python dsl_llada/eval/compute_bertscore_summarization.py \
        --files eval_results/summarization/xsum_b1_sde_nfe64.json
    python dsl_llada/eval/compute_bertscore_summarization.py --pattern "xsum_*_nfe64.json"
    python dsl_llada/eval/compute_bertscore_summarization.py --all --gpu 0
"""
import argparse
import glob
import json
import os

import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(_root, "eval_results", "summarization")


def score_file(path, scorer_state, force=False):
    data = json.load(open(path))
    if "bertscore_f1" in data and not force:
        print(f"  [skip] {os.path.basename(path)} already has BERTScore")
        return
    samples = data.get("samples")
    if not samples:
        print(f"  [skip] {os.path.basename(path)} no samples")
        return

    cands = [s["generated"] for s in samples]
    refs = [s["reference"] for s in samples]
    # bert_score wants non-empty strings
    idx_valid = [i for i, c in enumerate(cands) if c.strip()]
    cand_v = [cands[i] for i in idx_valid]
    ref_v = [refs[i] for i in idx_valid]
    if not cand_v:
        print(f"  [skip] {os.path.basename(path)} no valid candidates")
        return

    from bert_score import score as _bs
    P, R, F = _bs(
        cand_v, ref_v,
        lang="en", model_type=scorer_state["model_type"],
        num_layers=scorer_state.get("num_layers"),
        batch_size=scorer_state["batch_size"],
        device=scorer_state["device"],
        rescale_with_baseline=True,
        verbose=False,
    )
    p_arr = [0.0] * len(samples)
    r_arr = [0.0] * len(samples)
    f_arr = [0.0] * len(samples)
    for k, i in enumerate(idx_valid):
        p_arr[i] = float(P[k])
        r_arr[i] = float(R[k])
        f_arr[i] = float(F[k])

    for s, pv, rv, fv in zip(samples, p_arr, r_arr, f_arr):
        s["bertscore_p"] = round(pv * 100, 2)
        s["bertscore_r"] = round(rv * 100, 2)
        s["bertscore_f1"] = round(fv * 100, 2)

    # aggregate over VALID samples
    valid_f = [f_arr[i] for i in idx_valid]
    valid_p = [p_arr[i] for i in idx_valid]
    valid_r = [r_arr[i] for i in idx_valid]
    data["bertscore_p"] = round(sum(valid_p) / len(valid_p) * 100, 2)
    data["bertscore_r"] = round(sum(valid_r) / len(valid_r) * 100, 2)
    data["bertscore_f1"] = round(sum(valid_f) / len(valid_f) * 100, 2)
    data["bertscore_model"] = scorer_state["model_type"]
    data["bertscore_n"] = len(idx_valid)

    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  {os.path.basename(path)}: P={data['bertscore_p']}  "
          f"R={data['bertscore_r']}  F1={data['bertscore_f1']}  (n={len(idx_valid)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--pattern", default=None,
                   help="glob pattern within eval_results/summarization/")
    p.add_argument("--all", action="store_true",
                   help="score every merged (non-shard) file")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_type", default="roberta-large")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.all:
        paths = sorted(
            f for f in glob.glob(os.path.join(DIR, "*.json"))
            if "_shard" not in os.path.basename(f)
        )
    elif args.pattern:
        paths = sorted(glob.glob(os.path.join(DIR, args.pattern)))
    else:
        assert args.files, "--files / --pattern / --all required"
        paths = args.files

    if not paths:
        print("no files to score"); return
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    state = {
        "device": device,
        "model_type": args.model_type,
        "batch_size": args.batch_size,
    }
    print(f"Scoring {len(paths)} files with {args.model_type} on {device}")
    for path in paths:
        try:
            score_file(path, state, force=args.force)
        except Exception as e:
            print(f"  [error] {path}: {e}")


if __name__ == "__main__":
    main()
