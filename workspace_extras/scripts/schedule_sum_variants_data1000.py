#!/usr/bin/env python3
"""Dynamic GPU scheduler for 1000-sample summarization variant shards.

The older shell launcher starts 8 shards at once and then waits for the slowest
GPU, which leaves early-finished GPUs idle. This scheduler launches individual
shards whenever a GPU is idle, skips completed shard/merged outputs, and merges
a config once all 8 shards exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval_results" / "summarization"
LOG_ROOT = ROOT / "logs" / "sum_variant_data1000_dynamic"

DATA_FILES = {
    "xsum": "eval_data/xsum_1000.json",
    "cnn_dailymail": "eval_data/cnn_dailymail_1000.json",
    "pubmed": "eval_data/pubmed_1000.json",
    "arxiv": "eval_data/arxiv_1000.json",
}
DATASETS = ["xsum", "cnn_dailymail", "pubmed", "arxiv"]
NFES = [8, 16, 32, 64]
NUM_SHARDS = 8

BETA_TAG = "hf_beta1_sde_beta1_snrScaled_noNoiseScale_data1000"
RMDM_TAG = "original_rmdm_data1000"


def complete_json(path: Path, n_total: int = 1000) -> bool:
    if not path.exists():
        return False
    try:
        with path.open() as f:
            d = json.load(f)
        return d.get("n_samples_total") == n_total and d.get("n_samples_here") == n_total
    except Exception:
        return False


def valid_shard(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open() as f:
            d = json.load(f)
        return d.get("n_samples_total") == 1000 and d.get("n_samples_here", 0) > 0
    except Exception:
        return False


def gpu_memory_used() -> dict[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used = {}
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        used[int(idx)] = int(mem)
    return used


def config_queue():
    configs = []
    for ds in DATASETS:
        for nfe in NFES:
            configs.append(
                {
                    "dataset": ds,
                    "nfe": nfe,
                    "tag": BETA_TAG,
                    "args": [
                        "--model_key",
                        "hf_beta1",
                        "--method",
                        "sde",
                        "--out_tag",
                        "beta1_snrScaled_noNoiseScale_data1000",
                        "--sde_beta_infer",
                        "1.0",
                        "--sde_snr_min",
                        "0.01984375",
                        "--sde_sensitive_low",
                        "13.890625",
                        "--sde_sensitive_high",
                        "146.84375",
                        "--sde_snr_max",
                        "198.4375",
                    ],
                }
            )
    for ds in DATASETS:
        for nfe in NFES:
            configs.append(
                {
                    "dataset": ds,
                    "nfe": nfe,
                    "tag": RMDM_TAG,
                    "args": [
                        "--model_key",
                        "original",
                        "--method",
                        "rmdm",
                        "--out_tag",
                        "data1000",
                    ],
                }
            )
    return configs


def merged_path(cfg) -> Path:
    return OUT_DIR / f"{cfg['dataset']}_{cfg['tag']}_nfe{cfg['nfe']}.json"


def shard_path(cfg, shard_id: int) -> Path:
    return OUT_DIR / (
        f"{cfg['dataset']}_{cfg['tag']}_nfe{cfg['nfe']}_"
        f"shard{shard_id}of{NUM_SHARDS}.json"
    )


def maybe_merge(cfg) -> bool:
    mp = merged_path(cfg)
    if complete_json(mp):
        return True
    if not all(valid_shard(shard_path(cfg, i)) for i in range(NUM_SHARDS)):
        return False
    cmd = [
        "python",
        "dsl_llada/merge_summarization_shards.py",
        "--dataset",
        cfg["dataset"],
        "--method_tag",
        cfg["tag"],
        "--nfe",
        str(cfg["nfe"]),
    ]
    print("MERGE", cfg["dataset"], cfg["tag"], cfg["nfe"], flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return complete_json(mp)


def next_missing_shard(configs, running_keys: set[tuple[str, int, int]]):
    for cfg_idx, cfg in enumerate(configs):
        if maybe_merge(cfg):
            continue
        for shard_id in range(NUM_SHARDS):
            key = (cfg["tag"] + "/" + cfg["dataset"], cfg["nfe"], shard_id)
            if key in running_keys:
                continue
            if not valid_shard(shard_path(cfg, shard_id)):
                return cfg_idx, cfg, shard_id, key
    return None


def launch(cfg, shard_id: int, gpu: int):
    logdir = LOG_ROOT / cfg["tag"] / cfg["dataset"] / f"nfe{cfg['nfe']}"
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"shard{shard_id}.log"
    cmd = [
        "python",
        "dsl_llada/eval_summarization.py",
        "--dataset",
        cfg["dataset"],
        "--data_file",
        DATA_FILES[cfg["dataset"]],
        "--nfe",
        str(cfg["nfe"]),
        "--gpu",
        "0",
        "--limit",
        "1000",
        "--shard_id",
        str(shard_id),
        "--num_shards",
        str(NUM_SHARDS),
        *cfg["args"],
    ]
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/data2/ylong030/huggingface")
    env.setdefault("TRANSFORMERS_CACHE", "/data2/ylong030/huggingface/hub")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(
        f"LAUNCH gpu={gpu} {cfg['dataset']} {cfg['tag']} "
        f"nfe={cfg['nfe']} shard={shard_id}",
        flush=True,
    )
    fh = log_path.open("w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh


def main():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    configs = config_queue()
    running = {}

    while True:
        done = all(maybe_merge(cfg) for cfg in configs)
        if done and not running:
            print("ALL_DONE", flush=True)
            return

        finished = []
        for key, item in running.items():
            proc, fh, cfg, shard_id, gpu = item
            rc = proc.poll()
            if rc is not None:
                fh.close()
                print(
                    f"DONE rc={rc} gpu={gpu} {cfg['dataset']} {cfg['tag']} "
                    f"nfe={cfg['nfe']} shard={shard_id}",
                    flush=True,
                )
                if rc != 0:
                    print("Shard failed; leaving scheduler running for inspection.", flush=True)
                    raise SystemExit(rc)
                finished.append(key)
        for key in finished:
            running.pop(key, None)

        running_gpus = {item[4] for item in running.values()}
        try:
            mem = gpu_memory_used()
        except Exception as exc:
            print(f"GPU query failed: {exc}", flush=True)
            time.sleep(30)
            continue

        idle = [gpu for gpu, used in sorted(mem.items()) if used < 1024 and gpu not in running_gpus]
        running_keys = set(running)
        for gpu in idle:
            nxt = next_missing_shard(configs, running_keys)
            if nxt is None:
                break
            _, cfg, shard_id, key = nxt
            proc, fh = launch(cfg, shard_id, gpu)
            running[key] = (proc, fh, cfg, shard_id, gpu)
            running_keys.add(key)

        time.sleep(20)


if __name__ == "__main__":
    main()
