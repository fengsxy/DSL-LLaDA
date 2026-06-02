import os
import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # "medium" is default; "high" can be faster on Ampere
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # helps with oom

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

# Package imports
from dsl.dsl import build_dsl_from_cfg
from dataloader import get_tokenizer, get_dataloaders
from dsl.optimizers import build_optimizer
from dsl.optimizers import build_scheduler
from dsl.utils import save_checkpoint, load_checkpoint, pick_device, prepare_batch, load_weights_from_pretrained
from dsl.metrics import step_ce_loss, compute_metrics, nll_roar
from dsl.snrs import build_snr_path


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 0: Device and seed
    device = pick_device(cfg)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = False  # True  # Speed vs reproducibility
    torch.backends.cudnn.benchmark = True  # False

    # 0c: Logger
    logger = None
    if getattr(cfg.logging, "enabled", False):
        logger = wandb.init(project=cfg.logging.project, entity=cfg.logging.entity,
                            config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True, throw_on_missing=False))

    # 1) Data
    tokenizer = get_tokenizer(cfg)
    V = len(tokenizer)
    train_loader, val_loader = get_dataloaders(cfg, tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    print(f"[data] vocab_size={V}  pad_id={pad_id}")

    # 2) Model
    model = build_dsl_from_cfg(cfg, vocab_size=V)
    ema_model = build_dsl_from_cfg(cfg, vocab_size=V)
    if cfg.train.init_pretrained:
        model = load_weights_from_pretrained(model, model_name=cfg.train.pretrained_model)
    model = model.to(device)
    ema_model = ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    # 2b) SNR path
    snr_path = build_snr_path(cfg.snrpath, device=device)

    # 3) Optimizer
    optim = build_optimizer(cfg.optim, model.parameters())
    scheduler = build_scheduler(cfg.optim.scheduler, optim)

    # Optionally resume from a checkpoint
    global_step = 0
    if cfg.train.resume:
        ckpt_path = os.path.expanduser(cfg.train.resume)
        if os.path.exists(ckpt_path):
            print(f"[ckpt] Resuming from {ckpt_path}")
            state = load_checkpoint(ckpt_path, model, optim, scheduler=scheduler, map_location=device)
            if isinstance(state, dict):
                global_step = int(state.get("step", 0))
        else:
            print(f"[ckpt] resume path not found: {ckpt_path}")

    # 4) Training loop
    model.train()
    log_every = int(cfg.logging.log_every)
    max_steps = int(cfg.train.max_steps)

    ckpt_dir = os.path.expanduser(os.path.expandvars(cfg.train.ckpt.dir))
    save_every = cfg.train.ckpt.save_every
    save_last = cfg.train.ckpt.save_last
    save_best = cfg.train.ckpt.save_best

    best_val = float("inf")
    running_loss = 0.0
    running_tokens = 0
    t0 = time.time()

    accum_steps = int(getattr(cfg.train, "grad_accum_steps", 1))
    train_iter = iter(train_loader)

    while global_step < max_steps:
        optim.zero_grad(set_to_none=True)

        step_token_sum = 0       # tokens this optimizer step (mask-aware)
        step_loss_sum  = 0.0     # sum over micro-batches of (loss.item() * ntok)
        mix_lambda = float(getattr(cfg.train, "mix_lambda", 0.0))  # in [0, 1]; 0=CE only, 1=ROAR only

        for _ in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids, valid_tokens = prepare_batch(batch, pad_id, device)
            ntok = int(valid_tokens.sum().item())

            # SNR sampling (scalar per-sample)
            snr = snr_path.sample(input_ids.size(0), input_ids.size(1))

            # ----- Forward pass 1: CE on noisy path -----
            ce_coeff = 1.0 - mix_lambda
            if ce_coeff != 0.0:
                z = model.noisy_embedding(input_ids, snr)
                logits = model(z)
                ce_loss = step_ce_loss(logits, input_ids, valid_tokens)  # mean CE over non-pad tokens
                (ce_coeff * ce_loss / accum_steps).backward()  # free graph right after
            else:
                ce_loss = torch.tensor(0.0, device=device)

            # ----- pass 2: ROAR on masked/unmasked path -----
            if mix_lambda != 0.0:
                roar_loss = nll_roar(model, input_ids, valid_tokens, n_rep=1).mean()
                (mix_lambda * roar_loss / accum_steps).backward()
            else:
                roar_loss = torch.tensor(0.0, device=device)

            # Track training loss
            current_loss = ce_loss.item() + roar_loss.item()
            step_loss_sum  += current_loss * ntok
            step_token_sum += ntok

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if scheduler:
            scheduler.step()

        with torch.no_grad():  # EMA update
            msd = model.state_dict()
            for k, v in ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * cfg.optim.ema_decay + msd[k] * (1.0 - cfg.optim.ema_decay))

        # Bookkeeping for logging window
        running_loss   += step_loss_sum
        running_tokens += step_token_sum
        global_step += 1

        # Periodic checkpoint save
        if save_every and (global_step % save_every == 0):
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(ckpt_dir, f"step_{global_step}.pt"),
                model, optim, step=global_step, scheduler=scheduler,
            )

        # Logging
        if global_step % log_every == 0:
            dt = max(1e-9, time.time() - t0)
            running_loss = running_loss / max(1, running_tokens)           # token-weighted CE
            tok_per_s = running_tokens / dt                      # mask-aware throughput

            # NLL on the last micro-batch's inputs (kept small by default)
            with torch.no_grad():
                logm = compute_metrics(model, input_ids, valid_tokens, snr_path)

            if logger is not None:
                payload = {
                    "train/ce": logm['ce'],
                    "train/nll_best": logm["nll"],
                    "train/nll_max_snr": logm["nll_max_snr"],
                    "train/nll_roar": logm["nll_roar"],
                    "train/nll_diff": logm["nll_diff"],
                    "train/nll_recon": logm["nll_recon"],
                    "train/loss": running_loss,
                    "train/tok_per_s": tok_per_s,
                    "train/lr": optim.param_groups[0]["lr"]
                }
                wandb.log(payload, step=global_step)

            print(f"[train] step={global_step:4d} loss={running_loss:.4f}")

            running_loss = 0.0
            running_tokens = 0
            t0 = time.time()

        # Validation
        val_every = int(getattr(cfg.train.val, "every", 0))
        if val_every and global_step > 0 and (global_step % val_every == 0):
            model.eval()
            with torch.no_grad():
                val_ce_sum = 0.0
                cum_tok = 0
                nll_sum = nll_diff_sum = nll_recon_sum = 0.0
                roar_sum = 0.0
                nll_max_snr_sum = 0.0

                max_batches = getattr(cfg.train.val, "max_batches", None)
                for i, vb in enumerate(val_loader):
                    v_input_ids, valid_tokens = prepare_batch(vb, pad_id, device)

                    # Full-batch metrics (use compute_metrics for CE + NLL)
                    logm_v = compute_metrics(model, v_input_ids, valid_tokens, snr_path)

                    # loss accumulation (token-weighted)
                    val_ce_sum    += logm_v["ce"]        * logm_v["ntok"]
                    nll_sum       += logm_v["nll"]       * logm_v["ntok"]
                    nll_diff_sum  += logm_v["nll_diff"]  * logm_v["ntok"]
                    nll_recon_sum += logm_v["nll_recon"] * logm_v["ntok"]
                    roar_sum      += logm_v["nll_roar"]  * logm_v["ntok"]
                    nll_max_snr_sum += logm_v["nll_max_snr"] * logm_v["ntok"]

                    cum_tok += logm_v["ntok"]

                    if max_batches is not None and (i + 1) >= int(max_batches):
                        break

            val_ce             = val_ce_sum    / max(1, cum_tok)
            mean_val_nll       = nll_sum       / max(1, cum_tok)
            mean_val_nll_diff  = nll_diff_sum  / max(1, cum_tok)
            mean_val_nll_recon = nll_recon_sum / max(1, cum_tok)
            mean_val_roar      = roar_sum      / max(1, cum_tok)
            mean_val_nll_max_snr = nll_max_snr_sum / max(1, cum_tok)
            print(f"[valid] loss={val_ce:.4f}  nll={mean_val_nll:.4f}  roar={mean_val_roar:.4f}  nll_max_snr={mean_val_nll_max_snr:.4f}")

            if logger is not None:
                wandb.log(
                    {
                        "eval/ce": val_ce,
                        "eval/nll_best": mean_val_nll,
                        "eval/nll_max_snr": mean_val_nll_max_snr,
                        "eval/nll_roar": mean_val_roar,
                        "eval/nll_diff": mean_val_nll_diff,
                        "eval/nll_recon": mean_val_nll_recon,
                        "eval/loss": (1 - mix_lambda) * val_ce + mix_lambda * mean_val_roar,
                    },
                    step=global_step,
                )
            model.train()

            if save_best and val_ce < best_val:
                best_val = val_ce
                os.makedirs(ckpt_dir, exist_ok=True)
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    model, optim, step=global_step, scheduler=scheduler,
                    extra={"val_loss": float(val_ce)},
                )
                save_checkpoint(
                    os.path.join(ckpt_dir, "ema_best.pt"), ema_model, None, step=global_step,
                    extra={"val_loss": float(val_ce)},
                )

        if global_step >= max_steps:
            break

    if save_last:
        os.makedirs(ckpt_dir, exist_ok=True)
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optim, step=global_step)
        save_checkpoint(os.path.join(ckpt_dir, "ema_last.pt"), ema_model, None, step=global_step)

    if logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()