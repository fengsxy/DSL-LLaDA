#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset
from hydra import initialize, compose

# --- Make "dsl" importable when running as a plain script ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dsl.dsl import build_dsl_from_cfg
from dsl.metrics import step_ce_loss, nll_roar
from dsl.utils import pick_device, prepare_batch
from dataloader import get_tokenizer
from dsl.snrs import build_snr_path


class SyntheticTokenDataset(Dataset):
    """
    CPU-side synthetic tokens; lets you include DataLoader + H2D overhead
    when --e2e is on.
    """
    def __init__(self, length: int, vocab_size: int, num_batches: int, batch_size: int):
        self.L = int(length)
        self.V = int(vocab_size)
        self.N = int(num_batches * batch_size)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return torch.randint(0, self.V, (self.L,), dtype=torch.long)


def main():
    p = argparse.ArgumentParser("Throughput sweep w/ grad accumulation (mirrors train.py)")
    p.add_argument("--config_name", default="config", help="Top-level Hydra config name")
    p.add_argument("--sizes", type=str,
                   default="8,16,24,32,40,48,64,80,96,112,128,144,160,176,192,208,224,240,256,384,512",
                   help="Comma-separated micro-batch sizes to try")
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps (micro-batches per optimizer step)")
    p.add_argument("--steps", type=int, default=10, help="# optimizer steps measured per size")
    p.add_argument("--warmup", type=int, default=5, help="# warmup optimizer steps per size")
    p.add_argument("--mode", choices=["fwd", "fwbw"], default="fwbw", help="Forward only or forward+backward")
    p.add_argument("--e2e", action="store_true", help="Include DataLoader + H2D copies in timing")
    p.add_argument("--mix_lambda", type=float, default=0.0, help="0=CE only, 1=ROAR only (match train loop)")
    p.add_argument("--clip", type=float, default=1.0, help="Grad clip norm (match train loop)")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    accum = max(1, int(args.accum))

    # Hydra config: path relative to this script (../configs)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config_name)

    # Disable wandb for this micro-benchmark
    if hasattr(cfg, "logging") and hasattr(cfg.logging, "enabled"):
        cfg.logging.enabled = False

    device = pick_device(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Data/tokenizer like train.py
    tok = get_tokenizer(cfg)
    V = len(tok)
    pad_id = getattr(tok, "pad_token_id", None)

    # Model
    model = build_dsl_from_cfg(cfg, V).to(device)
    model.eval() if args.mode == "fwd" else model.train()

    # SNR path (mirror train.py)
    snr_path = build_snr_path(cfg.snrpath, device=device)

    # Optimizer + scheduler if doing backward
    optim = None
    scheduler = None
    if args.mode == "fwbw":
        from dsl.optimizers import build_optimizer, build_scheduler
        optim = build_optimizer(cfg.optim, model.parameters())
        scheduler = build_scheduler(cfg.optim.scheduler, optim)

    # Pull shapes from config
    L = int(cfg.data.block_size)

    print(f"\nSweep: mode={args.mode}, steps={args.steps}, warmup={args.warmup}, e2e={args.e2e}, accum={accum}")
    print(f"Seq len L={L}, Vocab V={V}, device={device}")
    print("-" * 116)
    print(f"{'MB':>6}  {'ACCUM':>6}  {'EFF_B':>7}  {'TOK/S':>12}  {'TIME/STEP(ms)':>14}  {'MEM(GB)':>8}  {'MIX':>5}  {'CLIP':>5}  RESULT")
    print("-" * 116)

    # Use inference mode for forward-only to avoid autograd storage
    grad_ctx = (torch.inference_mode if args.mode == "fwd" else nullcontext)

    for B in sizes:
        eff_B = B * accum  # effective batch per optimizer step
        try:
            # Prepare inputs generator for e2e path
            if args.e2e:
                ds = SyntheticTokenDataset(L, V, num_batches=(args.warmup + args.steps) * accum + 8, batch_size=B)
                dl = DataLoader(
                    ds,
                    batch_size=B,
                    shuffle=False,
                    num_workers=0,       # friendlier on Mac
                    pin_memory=False,    # CUDA-centric; avoid on MPS/CPU
                    drop_last=True,
                )
                dl_iter = iter(dl)
                ids_cpu = next(dl_iter)
                batch = {"input_ids": ids_cpu}
                input_ids, valid_tokens = prepare_batch(batch, pad_id, device)
            else:
                ids = torch.randint(0, V, (B, L), device=device, dtype=torch.long)
                batch = {"input_ids": ids}
                input_ids, valid_tokens = prepare_batch(batch, pad_id, device)

            # one-time warm-up graph build
            with grad_ctx():
                snr = snr_path.sample(B, L)
                z = model.noisy_embedding(input_ids, snr)
                _ = model(z)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            # ---------- WARMUP (not timed) ----------
            for _ in range(args.warmup):
                if args.mode == "fwbw":
                    optim.zero_grad(set_to_none=True)

                for _m in range(accum):
                    if args.e2e:
                        try:
                            ids_cpu = next(dl_iter)
                        except StopIteration:
                            dl_iter = iter(dl)
                            ids_cpu = next(dl_iter)
                        batch = {"input_ids": ids_cpu}
                        input_ids, valid_tokens = prepare_batch(batch, pad_id, device)
                    else:
                        ids = torch.randint(0, V, (B, L), device=device, dtype=torch.long)
                        batch = {"input_ids": ids}
                        input_ids, valid_tokens = prepare_batch(batch, pad_id, device)

                    with grad_ctx():
                        if args.mode == "fwbw":
                            ce_coeff = 1.0 - args.mix_lambda
                            if ce_coeff != 0.0:
                                snr = snr_path.sample(B, L)
                                z = model.noisy_embedding(input_ids, snr)
                                logits = model(z)
                                ce_loss = step_ce_loss(logits, input_ids, valid_tokens)
                                (ce_coeff * ce_loss / accum).backward()
                            if args.mix_lambda != 0.0:
                                roar_loss = nll_roar(model, input_ids, valid_tokens, n_rep=1).mean()
                                (args.mix_lambda * roar_loss / accum).backward()
                        else:
                            ce_coeff = 1.0 - args.mix_lambda
                            if ce_coeff != 0.0:
                                snr = snr_path.sample(B, L)
                                z = model.noisy_embedding(input_ids, snr)
                                logits = model(z)
                                ce_loss = step_ce_loss(logits, input_ids, valid_tokens)
                            if args.mix_lambda != 0.0:
                                roar_loss = nll_roar(model, input_ids, valid_tokens, n_rep=1).mean()

                if args.mode == "fwbw":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optim.step()
                    if scheduler is not None:
                        scheduler.step()

            # ---------- MEASURE ----------
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()

            for _ in range(args.steps):
                if args.mode == "fwbw":
                    optim.zero_grad(set_to_none=True)

                for _m in range(accum):
                    if args.e2e:
                        try:
                            ids_cpu = next(dl_iter)
                        except StopIteration:
                            dl_iter = iter(dl)
                            ids_cpu = next(dl_iter)
                        batch = {"input_ids": ids_cpu}
                        input_ids, valid_tokens = prepare_batch(batch, pad_id, device)
                    else:
                        ids = torch.randint(0, V, (B, L), device=device, dtype=torch.long)
                        batch = {"input_ids": ids}
                        input_ids, valid_tokens = prepare_batch(batch, pad_id, device)

                    with grad_ctx():
                        if args.mode == "fwbw":
                            ce_coeff = 1.0 - args.mix_lambda
                            if ce_coeff != 0.0:
                                snr = snr_path.sample(B, L)
                                z = model.noisy_embedding(input_ids, snr)
                                logits = model(z)
                                ce_loss = step_ce_loss(logits, input_ids, valid_tokens)
                                (ce_coeff * ce_loss / accum).backward()
                            if args.mix_lambda != 0.0:
                                roar_loss = nll_roar(model, input_ids, valid_tokens, n_rep=1).mean()
                                (args.mix_lambda * roar_loss / accum).backward()
                        else:
                            ce_coeff = 1.0 - args.mix_lambda
                            if ce_coeff != 0.0:
                                snr = snr_path.sample(B, L)
                                z = model.noisy_embedding(input_ids, snr)
                                logits = model(z)
                                ce_loss = step_ce_loss(logits, input_ids, valid_tokens)
                            if args.mix_lambda != 0.0:
                                roar_loss = nll_roar(model, input_ids, valid_tokens, n_rep=1).mean()

                if args.mode == "fwbw":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optim.step()
                    if scheduler is not None:
                        scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0

            # Effective work = tokens per optimizer step * steps
            toks = eff_B * L * args.steps
            toks_per_s = toks / max(dt, 1e-9)
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                # No easy per-process memory on CPU/MPS; report 0.0
                mem_gb = 0.0

            print(f"{B:6d}  {accum:6d}  {eff_B:7d}  {toks_per_s:12.0f}  {1e3*dt/args.steps:14.2f}  {mem_gb:8.2f}  {args.mix_lambda:5.2f}  {args.clip:5.2f}  ok")

        except torch.cuda.OutOfMemoryError:
            print(f"{B:6d}  {accum:6d}  {eff_B:7d}  {'-':>12}  {'-':>14}  {'OOM':>8}  {args.mix_lambda:5.2f}  {args.clip:5.2f}  OOM")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{B:6d}  {accum:6d}  {eff_B:7d}  {'-':>12}  {'-':>14}  {'ERR':>8}  {args.mix_lambda:5.2f}  {args.clip:5.2f}  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()