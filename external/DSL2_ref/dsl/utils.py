from __future__ import annotations
import os
import math
import torch
from torch.nn import functional as F
from typing import Any, Dict, Optional
import tempfile


def load_weights_from_pretrained(dsl_model, model_name='kuleshov-group/mdlm-owt'):
    print(f"Attempting to load weights from {model_name}")
    if os.path.exists(model_name) or model_name.endswith(('.pt', '.pth', '.ckpt')):
        print(f"[load_weights_from_pretrained] Detected checkpoint file: {model_name}")
        ckpt = torch.load(model_name, map_location='cpu')['model']
        state_dict = ckpt.get('state_dict', ckpt)
        missing, unexpected = dsl_model.load_state_dict(state_dict, strict=False)
        print("Loaded from checkpoint.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        return dsl_model

    import transformers
    from transformers import AutoModelForMaskedLM

    # Check we use same tokenizer! And put mask in right place
    # mdlm-owt uses GPT2 tokenizer (50257 tokens) and adds <mask> token after last token in gpt2 (=> 50258 tokens)
    trained_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

    trained_sd = trained_model.state_dict()
    dsl_sd = dsl_model.state_dict()

    # Filter only keys that match
    matched_sd = {k: v for k, v in trained_sd.items() if k in dsl_sd and v.shape == dsl_sd[k].shape}

    for name, param in matched_sd.items():
        if not torch.is_tensor(param):
            continue  # skip buffers or metadata

        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        max_abs = param.abs().max().item()

        if has_nan or has_inf or max_abs > 1e3:  # Adjust threshold as needed
            print(f"[!] Issue in '{name}': "
                  f"{'NaN' if has_nan else ''} "
                  f"{'Inf' if has_inf else ''} "
                  f"Max abs: {max_abs:.2e}")

    unused_keys = [k for k, v in trained_sd.items() if k not in matched_sd]
    print("Unused keys from pretrained model: ", unused_keys)

    # Load weights (non-strict to allow missing keys like vocab_embed)
    missing, unexpected = dsl_model.load_state_dict(matched_sd, strict=False)
    print("Keys in DSL not found in pretrained:", missing)
    print("Unexpected keys:", unexpected)

    print("Matching the output layer separately, due to size mismatch")
    # MDLM allows predicting mask. We cut that as it isn't allowed in our framework.
    dsl_sd['backbone.output_layer.linear.weight'].copy_(trained_sd['backbone.output_layer.linear.weight'][:-1])
    dsl_sd['backbone.output_layer.linear.bias'].copy_(trained_sd['backbone.output_layer.linear.bias'][:-1])

    print("Matching the embeddings separately...")
    trained_embed = trained_sd['backbone.vocab_embed.embedding']
    dsl_embed = dsl_sd['convert.backbone_embedding.weight']
    assert trained_embed.T.shape == dsl_embed.shape, f"{trained_embed.T.shape}, {dsl_embed.shape}"
    dsl_embed.copy_(trained_embed.T)  # Copy to the dsl embedding, in place.
    dsl_model.convert.logit_bias.data = torch.zeros_like(dsl_model.convert.logit_bias)
    mask_token_id = -1  # last token for mdlm-owt

    # Could initialize the token embedding with a random projection (OPTIONAL, no demonstrated difference)
    # Vp, dp = dsl_model.embed.weight_v.shape
    # V, d = trained_embed.shape
    # compressed_embed = trained_embed[:Vp]
    # R = F.normalize(torch.randn(dp, d), dim=1)
    # compressed_embed = compressed_embed @ R.T
    # compressed_embed = F.normalize(compressed_embed, dim=1)
    # with torch.no_grad():
    #     dsl_model.embed.weight_v.copy_(compressed_embed)

    vocab_size = dsl_embed.shape[1]
    bias_mag = math.log(vocab_size / (1./0.99 - 1))  # Target
    dsl_model.convert.logit_bias.data[mask_token_id] = bias_mag  # Set a bias so we give mask token for z=0.
    return dsl_model


# CHECKPOINTS
def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step) if step is not None else None,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload["rng"] = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if extra:
        payload["extra"] = extra

    dirpath = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + "."
    fd, tmppath = tempfile.mkstemp(dir=dirpath, prefix=prefix, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmppath)
        os.replace(tmppath, path)  # atomic on POSIX when same dir/filesystem
    finally:
        # best-effort cleanup if something went wrong before replace
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except OSError:
                pass

def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def pick_device(cfg: DictConfig) -> torch.device:
    if getattr(cfg, "device", None):  # allow manual override
        return torch.device(cfg.device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_batch(batch, pad_id, device):
    input_ids = batch["input_ids"].long().to(device, non_blocking=True)
    if pad_id is not None:
        valid_tokens = (input_ids != pad_id)
    else:
        valid_tokens = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, valid_tokens