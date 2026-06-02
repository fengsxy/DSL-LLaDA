# dsl_llada/train/llada_cpt_dsl.py
"""
Extended training script: adds DSL continuous-noise training to dllm_Trainer.

Key design: DSL components (noise_embed, converter) are attached directly to
the LLaDA model as submodules — no wrapper. The model stays a PreTrainedModel
so HF Trainer features (gradient checkpointing, save/load, LoRA) work normally
and DeepSpeed parameter partitioning is unaffected.

Usage (same as LLaDA-XDLM/llada_cpt.py, but with --training_method dsl):
    deepspeed dsl_llada/train/llada_cpt_dsl.py \
        --deepspeed dsl_llada/configs/ds_config.json \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --training_method dsl \
        [... other llada_cpt.py flags ...]
"""
import os
import sys

# Resolve paths relative to this file so the script works from any CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'LLaDA-XDLM'))  # for llada_cpt
sys.path.insert(0, _REPO_ROOT)                               # for dsl_llada package

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import TrainerCallback
import deepspeed

# Compatibility shim: newer transformers (>4.38) removed send_example_telemetry
import transformers.utils as _tu
if not hasattr(_tu, 'send_example_telemetry'):
    _tu.send_example_telemetry = lambda *a, **kw: None

def _gather_scalar(param):
    """Safely read a scalar/1D nn.Parameter under FSDP or ZeRO-3 (may be partitioned to 0 elements)."""
    if param.numel() == 0:
        # Try FSDP summed_data approach (FSDP flattens+shards, but small params often on all ranks)
        # Fall back to DeepSpeed GatheredParameters, then to broadcast
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # Under FSDP, use _full_param to get unsharded version
            if hasattr(param, '_full_param'):
                return param._full_param.detach().clone()
        except (ImportError, AttributeError):
            pass
        try:
            with deepspeed.zero.GatheredParameters(param):
                return param.detach().clone()
        except Exception:
            pass
        # Last resort: all-reduce to get the value from the owning rank
        full = torch.zeros(1, device=param.device)
        dist.all_reduce(full, op=dist.ReduceOp.SUM)
        return full
    return param


# Import llada_cpt as a module so we can monkey-patch its globals
import llada_cpt
from llada_cpt import *  # noqa: F401, F403  (re-exports main, build_model, dllm_Trainer, …)

from dsl_llada.core.dsl_modules import (
    attach_dsl_modules, sample_mixed_snr, noisy_embedding, embed_health_metrics,
    nll_integral_quick,
)


class SanitizeNaNCallback(TrainerCallback):
    """Placeholder — kept for backward compat."""
    pass


class dllm_Trainer_DSL(dllm_Trainer):  # noqa: F405
    """Extends dllm_Trainer with DSL continuous-noise training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-SNR rolling accumulators for stable monitoring
        self._snr_accum = {}  # bucket_key → {'sum': float, 'n': int}
        self._snr_accum_step = 0  # last reset step
        # Eval buffer: accumulate multiple batches for stable eval
        self._eval_buffer = []  # list of input_ids tensors
        self._eval_buffer_max = int(os.environ.get('DSL_EVAL_BUFFER_SIZE', '8'))
        # Separate optimizer for ff_out (managed outside DeepSpeed)
        self._ffout_optimizer = None
        self._ffout_fp32_weight = None
        self._ffout_fp32_bias = None
        self._ffout_grad_accum_count = 0

    def _init_ffout_optimizer(self):
        """Create a separate fp32 AdamW optimizer for ff_out.

        ff_out is FROZEN for DeepSpeed (excluded from ZeRO param groups) to avoid
        partition boundary NaN. Instead we maintain fp32 copies and a separate optimizer.
        """
        model = self.model
        ff_out = model.model.transformer.ff_out

        # Create fp32 copies on this rank's GPU
        device = torch.device(f'cuda:{self.args.local_rank}') if self.args.local_rank >= 0 else ff_out.weight.device
        self._ffout_fp32_weight = ff_out.weight.data.float().clone().to(device).requires_grad_(True)
        params = [self._ffout_fp32_weight]
        if ff_out.bias is not None:
            self._ffout_fp32_bias = ff_out.bias.data.float().clone().to(device).requires_grad_(True)
            params.append(self._ffout_fp32_bias)

        lr = self.args.learning_rate
        wd = self.args.weight_decay
        self._ffout_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        self._ffout_grad_accum_count = 0

        if self.args.local_rank in (-1, 0):
            print(f"[DSL] ff_out separate optimizer: lr={lr}, wd={wd}, "
                  f"weight shape={ff_out.weight.shape}", flush=True)

    def _ffout_step(self, grad_accum_steps):
        """Accumulate ff_out gradients and step when accumulation is complete."""
        if self._ffout_optimizer is None:
            return
        self._ffout_grad_accum_count += 1
        if self._ffout_grad_accum_count >= grad_accum_steps:
            # Average gradients
            if self._ffout_fp32_weight.grad is not None:
                self._ffout_fp32_weight.grad.div_(grad_accum_steps)
            if self._ffout_fp32_bias is not None and self._ffout_fp32_bias.grad is not None:
                self._ffout_fp32_bias.grad.div_(grad_accum_steps)
            # Clip
            torch.nn.utils.clip_grad_norm_(
                [self._ffout_fp32_weight] + ([self._ffout_fp32_bias] if self._ffout_fp32_bias else []),
                self.args.max_grad_norm)
            # All-reduce gradients across ranks
            if dist.is_initialized():
                dist.all_reduce(self._ffout_fp32_weight.grad, op=dist.ReduceOp.AVG)
                if self._ffout_fp32_bias is not None and self._ffout_fp32_bias.grad is not None:
                    dist.all_reduce(self._ffout_fp32_bias.grad, op=dist.ReduceOp.AVG)
            self._ffout_optimizer.step()
            self._ffout_optimizer.zero_grad()
            # Copy fp32 → bf16 model weights
            ff_out = self.model.model.transformer.ff_out
            ff_out.weight.data.copy_(self._ffout_fp32_weight.data.to(ff_out.weight.dtype))
            if ff_out.bias is not None and self._ffout_fp32_bias is not None:
                ff_out.bias.data.copy_(self._ffout_fp32_bias.data.to(ff_out.bias.dtype))
            self._ffout_grad_accum_count = 0

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        # Step ff_out's separate optimizer after gradient accumulation
        if self._ffout_optimizer is not None:
            self._ffout_step(self.args.gradient_accumulation_steps)
        return loss

    def train(self, *args, **kwargs):
        # Init ff_out optimizer if needed (after DeepSpeed setup)
        if os.environ.get('DSL_FREEZE_FFOUT', '1') == '0':
            self._init_ffout_optimizer()
        return super().train(*args, **kwargs)

    @torch.no_grad()
    def _eval_fixed_snr(self, model):
        """Evaluate at fixed SNR points + clean CE + infilling + NLL bound.

        Uses accumulated eval buffer (multiple batches) for stable metrics.
        Logs to wandb.
        """
        if not self._eval_buffer:
            return

        step = self.state.global_step
        device = self._eval_buffer[0].device
        n_samples = len(self._eval_buffer)
        wb = {}

        # Accumulate metrics over all buffered samples
        fixed_snrs = [1.0, 5.0, 10.0, 20.0, 50.0]
        snr_ce_acc = {s: {'ce': 0., 'acc': 0.} for s in fixed_snrs}
        ce_clean_sum, acc_clean_sum = 0., 0.
        fill_acc_sum = 0.
        nll_metrics_sum = {'nll_bpt': 0., 'nll_diff_bpt': 0., 'nll_recon_bpt': 0., 'best_snr': 0.}
        has_nll = hasattr(model, 'noise_embed') and os.environ.get('DSL_EVAL_NLL', '0') == '1'

        for input_ids in self._eval_buffer:
            B, L = input_ids.shape

            # 1. Fixed-SNR CE + accuracy
            for snr_val in fixed_snrs:
                snr_t = torch.full((B, L), snr_val, device=device)
                z = noisy_embedding(model.noise_embed, input_ids, snr_t)
                h = model.converter(z)
                out = model(input_ids=input_ids, inputs_embeds=h)
                ce = F.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)).float(),
                    input_ids.view(-1), reduction='mean').item()
                preds = out.logits.argmax(dim=-1)
                acc = (preds == input_ids).float().mean().item()
                snr_ce_acc[snr_val]['ce'] += ce
                snr_ce_acc[snr_val]['acc'] += acc

            # 2. Clean CE
            out_clean = model(input_ids=input_ids)
            ce_clean_sum += F.cross_entropy(
                out_clean.logits.view(-1, out_clean.logits.size(-1)).float(),
                input_ids.view(-1), reduction='mean').item()
            acc_clean_sum += (out_clean.logits.argmax(dim=-1) == input_ids).float().mean().item()

            # 3. Infilling
            mask_id = 126336
            n_mask = L // 2
            masked = input_ids.clone()
            perm = torch.randperm(L, device=device)[:n_mask]
            masked[:, perm] = mask_id
            out_fill = model(masked)
            fill_acc_sum += (out_fill.logits.argmax(dim=-1)[:, perm] == input_ids[:, perm]).float().mean().item()

            # 4. NLL integral bound
            if has_nll:
                nll_m = nll_integral_quick(model, input_ids, int_steps=20, snr_max=30.0)
                for k in nll_metrics_sum:
                    nll_metrics_sum[k] += nll_m[k]

        # Average over samples
        for snr_val in fixed_snrs:
            wb[f'eval/ce_snr{int(snr_val)}'] = snr_ce_acc[snr_val]['ce'] / n_samples
            wb[f'eval/acc_snr{int(snr_val)}'] = snr_ce_acc[snr_val]['acc'] / n_samples
        ce_clean = ce_clean_sum / n_samples
        wb['eval/ce_clean'] = ce_clean
        wb['eval/clean_acc'] = acc_clean_sum / n_samples
        fill_acc = fill_acc_sum / n_samples
        wb['eval/infill_acc_50pct'] = fill_acc
        if has_nll:
            for k in nll_metrics_sum:
                wb[f'eval/{k}'] = nll_metrics_sum[k] / n_samples

        # Print summary
        snr_str = " ".join([f"SNR{int(s)}={wb[f'eval/acc_snr{int(s)}']:.3f}" for s in fixed_snrs])
        print(f"  [Eval step={step}, n={n_samples}] clean_CE={ce_clean:.3f}, infill@50%={fill_acc:.3f}",
              flush=True)
        print(f"    Fixed-SNR acc: {snr_str}", flush=True)
        if has_nll and 'eval/nll_bpt' in wb:
            print(f"    NLL bound: {wb['eval/nll_bpt']:.3f} bpt "
                  f"(diff={wb['eval/nll_diff_bpt']:.3f}, recon={wb['eval/nll_recon_bpt']:.3f}, "
                  f"best_snr={wb['eval/best_snr']:.1f})", flush=True)

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(wb, step=step)
        except ImportError:
            pass

    def compute_loss_dsl(self, model, input_ids, prompt_length):
        """
        DSL forward: noise_embed → noisy_embedding → converter → LLaDA backbone → CE loss.
        Interface matches compute_loss_mdm / compute_loss_xdm: returns (loss_sum, logits).
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Sample SNRs using snr_max (read from env to avoid FSDP/ZeRO-3 gather issues on small params)
        snr_max_val = float(os.environ.get('DSL_SNR_MAX', '100.0'))
        snr_max_ln_val = min(snr_max_val * 0.4, 40.0)  # LogNormal clamp ≤ 40% of ROAR max
        snrs = sample_mixed_snr(B, L, device, snr_max=snr_max_val, snr_max_ln=snr_max_ln_val)

        # ZeRO-3: converter & noise_embed are called outside model.forward(),
        # so we must explicitly gather their partitioned parameters.
        dsl_params = list(model.noise_embed.parameters()) + list(model.converter.parameters())
        with deepspeed.zero.GatheredParameters(dsl_params, modifier_rank=None):
            z_noisy = noisy_embedding(model.noise_embed, input_ids, snrs)
            h = model.converter(z_noisy)  # (B, L, d_backbone)

        # Pass to LLaDA backbone (inputs_embeds bypasses wte)
        # ff_out is managed separately from DeepSpeed to avoid ZeRO-2 partition NaN.
        # We use Identity() trick to get hidden states, then compute logits in fp32.
        if self._ffout_optimizer is not None:
            # ff_out is frozen for DeepSpeed but trained via separate optimizer
            saved_ff_out = model.model.transformer.ff_out
            model.model.transformer.ff_out = torch.nn.Identity()
            out = model(input_ids=input_ids, inputs_embeds=h)
            model.model.transformer.ff_out = saved_ff_out
            hidden = out.logits.contiguous()
            # Use our fp32 copies (with grad tracking) for logits
            logits = F.linear(hidden.float(), self._ffout_fp32_weight,
                              self._ffout_fp32_bias if self._ffout_fp32_bias is not None else None)
        else:
            out = model(input_ids=input_ids, inputs_embeds=h)
            logits = out.logits.contiguous().float()  # (B, L, V)

        # CE loss on all tokens (float32 for 126k-dim log_softmax precision)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # already fp32
            input_ids.view(-1),
            reduction='none',
        ).view(input_ids.shape)

        if prompt_length is not None:
            prompt_mask = (
                torch.arange(L, device=device).unsqueeze(0) < prompt_length.view(-1, 1)
            )
            loss[prompt_mask] = 0.0

        # Per-SNR accumulation (every step on rank-0, cheap)
        if self.args.local_rank in (-1, 0):
            with torch.no_grad():
                flat_snr = snrs.view(-1)
                flat_loss = loss.detach().view(-1)
                buckets = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
                for lo, hi in buckets:
                    key = f'{lo}_{hi}'
                    bmask = (flat_snr >= lo) & (flat_snr < hi)
                    n = bmask.sum().item()
                    if n > 0:
                        s = flat_loss[bmask].sum().item()
                        if key not in self._snr_accum:
                            self._snr_accum[key] = {'sum': 0.0, 'n': 0}
                        self._snr_accum[key]['sum'] += s
                        self._snr_accum[key]['n'] += n
                # ROAR vs continuous
                snr_std = snrs.std(dim=1)
                is_roar = snr_std > 0.1
                flat_roar = is_roar.unsqueeze(1).expand(B, L).reshape(-1)
                for tag, fmask in [('roar', flat_roar), ('cont', ~flat_roar)]:
                    n = fmask.sum().item()
                    if n > 0:
                        s = flat_loss[fmask].sum().item()
                        if tag not in self._snr_accum:
                            self._snr_accum[tag] = {'sum': 0.0, 'n': 0}
                        self._snr_accum[tag]['sum'] += s
                        self._snr_accum[tag]['n'] += n

        # DSL diagnostics (rank-0 only, every 10 steps) — print rolling averages
        if self.args.local_rank in (-1, 0) and self.state.global_step % 10 == 0:
            with torch.no_grad():
                # Top-1 accuracy
                preds = logits.argmax(dim=-1)
                acc = (preds == input_ids).float().mean().item()
                # Converter quality: cosine sim between h and wte(input_ids)
                wte_h = model.model.transformer.wte(input_ids)  # (B, L, d)
                cos = F.cosine_similarity(h.float(), wte_h.float(), dim=-1).mean().item()
                # Converter params
                # Read converter params safely (may be sharded under FSDP/ZeRO-3)
                try:
                    beta_val = model.converter.beta.data.flatten()[0].item() if model.converter.beta.numel() > 0 else 5.0
                    mask_bias = model.converter.logit_bias.data[-1].item() if model.converter.logit_bias.numel() > 0 else 11.75
                    snr_max_log = model.log_snr_max.data.flatten()[0].item() if hasattr(model, 'log_snr_max') and model.log_snr_max.numel() > 0 else math.log(100)
                except Exception:
                    beta_val, mask_bias, snr_max_log = 5.0, 11.75, math.log(100)
                wb_metrics = {
                    'dsl/acc': acc,
                    'dsl/cos_h_wte': cos,
                    'dsl/beta': beta_val,
                    'dsl/mask_bias': mask_bias,
                    'dsl/snr_max': math.exp(snr_max_log),
                    'dsl/log_snr_max': snr_max_log,
                }

                # Print rolling per-SNR averages (accumulated over ~10 steps)
                buckets = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
                parts = []
                for lo, hi in buckets:
                    key = f'{lo}_{hi}'
                    if key in self._snr_accum and self._snr_accum[key]['n'] > 0:
                        avg = self._snr_accum[key]['sum'] / self._snr_accum[key]['n']
                        n = self._snr_accum[key]['n']
                        parts.append(f"[{lo},{hi})={avg:.2f}({n})")
                        wb_metrics[f'dsl/loss_snr_{lo}_{hi}'] = avg

                roar_avg = (self._snr_accum['roar']['sum'] / self._snr_accum['roar']['n']
                            if 'roar' in self._snr_accum and self._snr_accum['roar']['n'] > 0 else 0)
                cont_avg = (self._snr_accum['cont']['sum'] / self._snr_accum['cont']['n']
                            if 'cont' in self._snr_accum and self._snr_accum['cont']['n'] > 0 else 0)
                wb_metrics['dsl/roar_loss'] = roar_avg
                wb_metrics['dsl/cont_loss'] = cont_avg

                bucket_str = " ".join(parts)
                accum_steps = self.state.global_step - self._snr_accum_step
                print(f"  [DSL step={self.state.global_step}] "
                      f"acc={acc:.4f}, cos(h,wte)={cos:.4f}, "
                      f"beta={beta_val:.4f}, mask_bias={mask_bias:.2f}, "
                      f"snr_max={math.exp(snr_max_log):.1f}",
                      flush=True)
                print(f"    SNR buckets (rolling {accum_steps} steps): {bucket_str}",
                      flush=True)
                print(f"    ROAR_loss={roar_avg:.3f} Cont_loss={cont_avg:.3f}",
                      flush=True)

                # Reset accumulators
                self._snr_accum = {}
                self._snr_accum_step = self.state.global_step

                # Log to wandb
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(wb_metrics, step=self.state.global_step)
                except ImportError:
                    pass

        # Accumulate eval buffer (rank-0 only, detach to avoid memory leak)
        # Skip when ff_out optimizer is active (saves ~2GB VRAM)
        if self.args.local_rank in (-1, 0) and self._ffout_optimizer is None:
            if len(self._eval_buffer) < self._eval_buffer_max:
                self._eval_buffer.append(input_ids.detach().clone())

        # Detailed eval every 50 steps (rank-0 only): uses buffered samples
        # Skip eval when ff_out separate optimizer is active (fp32 copies use ~2GB extra VRAM)
        if (self.state.global_step % 50 == 0
                and self.state.global_step > 0
                and self.args.local_rank in (-1, 0)
                and self._ffout_optimizer is None):
            self._eval_fixed_snr(model)
            self._eval_buffer = []  # reset buffer after eval

        # Log embedding health every 100 steps (rank-0 only)
        if (self.state.global_step % 100 == 0
                and hasattr(model, 'noise_embed')
                and self.args.local_rank in (-1, 0)):
            health = embed_health_metrics(model)
            parts = [f"{k}={v:.4f}" for k, v in health.items()]
            print(f"  [EmbedHealth step={self.state.global_step}] {', '.join(parts)}",
                  flush=True)
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(health, step=self.state.global_step)
            except ImportError:
                pass

        return loss.sum(), logits

    def compute_loss_core(self, model, input_ids, prompt_length):
        if self.reverse_process_training_method == 'dsl':
            return self.compute_loss_dsl(model, input_ids, prompt_length)
        return super().compute_loss_core(model, input_ids, prompt_length)


def _build_model_dsl(model_args):
    """
    Load LLaDA with the original build_model, then attach DSL modules
    (noise_embed, converter) when training_method == 'dsl'.
    """
    model, tokenizer, config = llada_cpt._orig_build_model(model_args)
    if model_args.training_method == 'dsl':
        # ALWAYS freeze ff_out for DeepSpeed to avoid ZeRO-2 partition boundary NaN.
        # When DSL_FREEZE_FFOUT=0, ff_out is trained via a separate fp32 optimizer
        # managed by dllm_Trainer_DSL (bypasses DeepSpeed entirely for ff_out).
        noise_init = os.environ.get('DSL_NOISE_INIT', 'random')
        attach_dsl_modules(model, freeze_ff_out=True, noise_init=noise_init)

        want_train_ffout = os.environ.get('DSL_FREEZE_FFOUT', '1') == '0'
        if want_train_ffout and int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print("[DSL] ff_out will be trained via separate fp32 optimizer (outside DeepSpeed)",
                  flush=True)

        # Optional: scale converter gradients for higher effective lr
        converter_lr_scale = float(os.environ.get('DSL_CONVERTER_LR_SCALE', '1'))
        if converter_lr_scale != 1.0:
            for name, p in model.converter.named_parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad, s=converter_lr_scale: grad * s)
            if int(os.environ.get('LOCAL_RANK', '0')) == 0:
                print(f"[DSL] Converter grad scale={converter_lr_scale}",
                      flush=True)
    return model, tokenizer, config


if __name__ == '__main__':
    # Patch llada_cpt's module globals so that main() picks up our overrides.
    llada_cpt._orig_build_model = llada_cpt.build_model
    llada_cpt.build_model = _build_model_dsl
    llada_cpt.dllm_Trainer = dllm_Trainer_DSL
    main()  # noqa: F405
