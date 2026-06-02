from __future__ import annotations
from typing import Iterable, Mapping
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def build_optimizer(optim_cfg: Mapping, params: Iterable) -> Optimizer:
    """
    Build an optimizer from a Hydra config group under `optim`.
    Supports: adamw, sgd
    """
    name = str(optim_cfg.get("name", "")).lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        betas = optim_cfg.get("betas", [0.9, 0.999])
        eps = float(optim_cfg.get("eps", 1e-8))
        # Ensure tuple for torch
        betas = (float(betas[0]), float(betas[1]))
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    elif name == "sgd":
        momentum = float(optim_cfg.get("momentum", 0.0))
        nesterov = bool(optim_cfg.get("nesterov", False))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer name '{name}'. Expected one of: 'adamw', 'sgd'.")


def build_scheduler(cfg, optimizer):
    name = getattr(cfg, "name", "none")
    if name == "none": return None
    if name == "cosine_warmup":
        ws, tmax, min_lr = cfg.warmup_steps, cfg.t_max, cfg.min_lr
        base_lr = optimizer.param_groups[0]["lr"]
        def lr_lambda(step):
            if step < ws: return (step + 1) / max(1, ws)
            # cosine from 1.0 -> min_lr/base_lr
            progress = min(1.0, (step - ws) / max(1, tmax - ws))
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            scale = min_lr / base_lr + (1 - min_lr / base_lr) * cosine
            return scale
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unknown scheduler: {name}")