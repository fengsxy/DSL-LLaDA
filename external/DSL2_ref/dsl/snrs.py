"""Utilities for specifying SNR paths.
Desiderata:
- during training we need to sample
- During inference (sampling and NLL), we need a specific discrete path
- There should be a config for paths... then we can name different paths and hypers. Certain distributions + hypers. Generated paths.
- We'd like to consider fixed learned paths - generated using a trained model and then saved.
Maybe we could have a script that generates paths and saves them, with some registry and metadata,
then an SNRPath class that can load them and sample from them with some registry.
"""
from typing import Any, Mapping
from omegaconf import OmegaConf, DictConfig

import math
import torch
import torch.distributions as D

from dsl.metrics import make_mask_tensor

# ---------------------------------------------------------------------------
# Base API
# ---------------------------------------------------------------------------

class SNRPath:
    """
    Abstract interface for SNR paths.

    Two primary operations:
    - sample(batch_size): stochastic per-sample SNRs for training.
    - discrete_path(K): a deterministic K-point SNR grid for inference / NLL.
    The base class only tracks dtype and device, and does not implement any clamping or bounds.
    """

    name: str = "base"

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.dtype = dtype
        self.device = torch.device(device)

    # ---- Public API ----
    def sample(self, batch_size: int, num_vars: int) -> torch.Tensor:
        """Draw per-sample, possibly per variable SNRs (shape: [B, T])."""
        raise NotImplementedError

    def discrete_path(self, K: int) -> torch.Tensor:
        """Return a deterministic length-K SNR grid (shape: [K])."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Linear path
# ---------------------------------------------------------------------------

class LinearSNRPath(SNRPath):
    """Uniform SNR in [snr_min, snr_max] for training; linear grid for inference.

    Requires explicit snr_min and snr_max; no clamping.
    """

    name: str = "linear"

    def __init__(
        self,
        snr_min: float,
        snr_max: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__(dtype=dtype, device=device)
        self.snr_min = torch.tensor(snr_min, dtype=dtype, device=self.device)
        self.snr_max = torch.tensor(snr_max, dtype=dtype, device=self.device)
        self._dist = D.Uniform(self.snr_min, self.snr_max)

    def sample(self, batch_size: int, num_vars: int) -> torch.Tensor:
        return self._dist.sample((batch_size,))

    def discrete_path(self, K: int) -> torch.Tensor:
        return torch.linspace(self.snr_min.item(), self.snr_max.item(), steps=K, dtype=self.dtype, device=self.device)

# ---------------------------------------------------------------------------
# Log-linear path
# ---------------------------------------------------------------------------

class LogLinearSNRPath(SNRPath):
    """Uniform in log(SNR) for training; log-linear grid for inference.

    Requires explicit snr_min and snr_max; no clamping.
    """
    name: str = "loglinear"

    def __init__(
        self,
        snr_min: float,
        snr_max: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__(dtype=dtype, device=device)
        self.snr_min = torch.tensor(snr_min, dtype=dtype, device=self.device)
        self.snr_max = torch.tensor(snr_max, dtype=dtype, device=self.device)
        log_min = torch.log(self.snr_min)
        log_max = torch.log(self.snr_max)
        self._dist = D.Uniform(log_min, log_max)

    def sample(self, batch_size: int, num_vars: int) -> torch.Tensor:
        # Uniform in log-space, then exponentiate
        samples = self._dist.sample((batch_size,))
        return samples.exp()

    def discrete_path(self, K: int) -> torch.Tensor:
        # Logspace between log10(snr_min) and log10(snr_max)
        return torch.logspace(
            math.log10(self.snr_min.item()),
            math.log10(self.snr_max.item()),
            steps=K,
            dtype=self.dtype,
            device=self.device,
        )


# ---------------------------------------------------------------------------
# Log-normal path
# ---------------------------------------------------------------------------

class LogNormalSNRPath(SNRPath):
    """LogNormal(mu, sigma) for training; quantile grid for inference.
    No reference to snr_min or snr_max; distribution is always on self.device.
    """

    name: str = "lognormal"

    def __init__(
        self,
        mu: float = 1.4,
        sigma: float = 0.55,
        snr_max: float = 40.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        **kwargs
    ):
        super().__init__(dtype=dtype, device=device)
        self.mu = torch.tensor(mu, device=self.device, dtype=dtype)
        self.sigma = torch.tensor(sigma, device=self.device, dtype=dtype)
        self.snr_min = torch.tensor(0., device=self.device, dtype=dtype)
        self.snr_max = torch.tensor(snr_max, device=self.device, dtype=dtype)
        self._dist = D.LogNormal(self.mu, self.sigma)

    def sample(self, batch_size: int, num_vars: int) -> torch.Tensor:
        return self._dist.sample((batch_size,))

    def discrete_path(self, K: int) -> torch.Tensor:
        # Create a K-point grid in quantile space between the CDF at 1e-4 and the CDF at snr_max.
        # This yields SNR values approximately spanning [1e-4, snr_max] (inclusive at the endpoints).
        eps = torch.finfo(self.dtype).eps
        # Map value bounds to quantiles, and clamp away from 0/1 to avoid infs in icdf
        q_lo = torch.clamp(self._dist.cdf(self.snr_min), min=0., max=1.0 - eps)
        q_hi = torch.clamp(self._dist.cdf(self.snr_max), min=q_lo.item() + eps, max=1.0 - eps)
        q = torch.linspace(q_lo.item(), q_hi.item(), steps=K, device=self.device, dtype=self.dtype)
        return self._dist.icdf(q)


class ROARSNRPath(SNRPath):
    """Random Order Autoregressive (ROAR) style per-variable SNR path.

    For each sequence of length T:
      - Choose a random number of unmasked tokens k in [0, T).
      - Sample a random subset of size k to receive high SNR (snr_max).
      - Remaining tokens get zero SNR.
    """

    name: str = "roar"

    def __init__(self, snr_max: float = 100.0, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        super().__init__(dtype=dtype, device=device)
        self.snr_max = snr_max

    def sample(self, batch_size: int, num_vars: int) -> torch.Tensor:
        """Draw per-token SNRs for a batch of sequences, using ROAR paths.

        Args:
            batch_size: int
            num_var: int - usually the sequence length T
        Returns:
            snrs: (B, T) tensor with values in {0, snr_max}.
        """
        B, T = batch_size, num_vars
        # pick random number of unmasked tokens per sequence
        unmask_sizes = torch.randint(0, T, (B,), device=self.device)
        mask = make_mask_tensor(unmask_sizes, T)  # [B, T], True=masked, False=unmasked
        snrs = torch.zeros(B, T, device=self.device, dtype=self.dtype)
        snrs[~mask] = self.snr_max
        return snrs

    def discrete_path(self, K: int) -> torch.Tensor:
        # We could implement this, basically increase SNR for one variable at a time in random order
        raise NotImplementedError("ROARSNRPath is not implemented.")


class MixedSNRPath(SNRPath):
    """Half batch from ROAR, half from LogNormal."""

    name: str = "mixed"

    def __init__(
        self,
        # ROAR params
        snr_max: float = 100.0,
        # LogNormal params
        mu: float = 1.4,
        sigma: float = 0.55,
        snr_max_ln: float = 40.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__(dtype=dtype, device=device)
        self.roar = ROARSNRPath(snr_max=snr_max, dtype=dtype, device=device)
        self.logn = LogNormalSNRPath(mu=mu, sigma=sigma,
                                     snr_max=snr_max_ln,
                                     dtype=dtype, device=device)

    def sample(self, batch_size: int, num_vars: int) -> torch.Tensor:
        # Split batch
        frac = batch_size // 4
        roar_snrs = self.roar.sample(frac, num_vars)
        ln_snrs   = self.logn.sample(batch_size - frac, num_vars).unsqueeze(1).expand(-1, num_vars)
        return torch.cat([roar_snrs, ln_snrs], dim=0)

    def discrete_path(self, K: int) -> torch.Tensor:
        # Could return concatenated grids, but simplest is lognormal’s
        return self.logn.discrete_path(K)

# ---------------------------------------------------------------------------
# Saved (fixed) path
# ---------------------------------------------------------------------------

class SavedSNRPath(SNRPath):
    """Use a fixed SNR path loaded from disk.

    Expected file formats:
    - a 1D tensor saved via torch.save(...)
    - or a dict with key 'snrs' / 'snr_path' / 'path' containing a 1D tensor
    """

    name: str = "saved"

    def __init__(
        self,
        snrs: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        if not isinstance(snrs, torch.Tensor):
            raise TypeError("SavedSNRPath expects a torch.Tensor for 'snrs'.")
        super().__init__(dtype=dtype, device=device)
        self._snrs = snrs.detach().flatten().to(device=self.device, dtype=dtype)

    def sample(self, batch_size: int, num_vars: int) -> torch.Tensor:
        idx = torch.randint(0, self._snrs.numel(), (batch_size,), device=self.device)
        return self._snrs.index_select(0, idx)

    def discrete_path(self, K: int) -> torch.Tensor:
        N = self._snrs.numel()
        if K == N:
            return self._snrs
        x_src = torch.linspace(0.0, 1.0, steps=N, device=self.device)
        x_tgt = torch.linspace(0.0, 1.0, steps=K, device=self.device)
        return torch.interp(x_tgt, x_src, self._snrs)


# ---------------------------------------------------------------------------
# Simple builder (config → SNRPath)
# ---------------------------------------------------------------------------

def build_snr_path(snr_cfg: Mapping[str, Any] | Any,
                   device='cpu', dtype=torch.float32) -> SNRPath:
    """Build an SNRPath from config.

    Expected structure:
    - Either pass the snrpath dict directly, e.g. {"name":"lognormal", "mu":1.4, "sigma":0.55, ...}
    - Or pass a Hydra/omegaconf object with attribute `.snrpath` holding the same fields.

    Known names: {"linear", "lognormal", "saved"}.
    """
    if isinstance(snr_cfg, DictConfig):
        snr_cfg = OmegaConf.to_container(snr_cfg, resolve=True)  # -> dict
    assert isinstance(snr_cfg, dict), TypeError("snr_cfg must be DictConfig or dict")
    name = snr_cfg['name']
    snr_min = snr_cfg.get('snr_min', 0.)
    if name == "linear":
        snr_min = snr_min
        snr_max = snr_cfg['snr_max']
        if snr_min is None or snr_max is None:
            raise ValueError("linear SNRPath requires both 'snr_min' and 'snr_max' in config.")
        return LinearSNRPath(
            snr_min=snr_min,
            snr_max=snr_max,
            dtype=dtype,
            device=device,
        )
    elif name == "loglinear":
        snr_min = snr_cfg['snr_min']
        snr_max = snr_cfg['snr_max']
        if snr_min is None or snr_max is None:
            raise ValueError("loglinear SNRPath requires both 'snr_min' and 'snr_max' in config.")
        return LogLinearSNRPath(
            snr_min=snr_min,
            snr_max=snr_max,
            dtype=dtype,
            device=device,
        )
    elif name == "lognormal":
        return LogNormalSNRPath(
            mu=float(snr_cfg['mu']),
            sigma=float(snr_cfg['sigma']),
            snr_min=snr_min,
            snr_max=float(snr_cfg['snr_max']),
            dtype=dtype,
            device=device,
        )
    elif name == 'roar':
        return ROARSNRPath(
            snr_max=float(snr_cfg.get('snr_max', 100.0)),
            dtype=dtype,
            device=device,
        )
    elif name == "mixed":
        return MixedSNRPath(
            snr_max=float(snr_cfg.get('snr_max', 100.0)),
            mu=float(snr_cfg.get('mu', 1.4)),
            sigma=float(snr_cfg.get('sigma', 0.55)),
            snr_max_ln=float(snr_cfg.get('snr_max_ln', 40.0)),
            dtype=dtype,
            device=device,
        )
    elif name == "saved":
        path = snr_cfg['path']
        if path is None:
            raise ValueError("saved SNRPath requires 'path' (or 'file') to a torch-saved tensor or dict.")
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            snrs = obj
        elif isinstance(obj, Mapping):
            for key in ("snrs", "snr_path", "path"):
                if key in obj:
                    val = obj[key]
                    if isinstance(val, torch.Tensor):
                        snrs = val
                        break
            else:
                raise ValueError(f"Could not find 'snrs' tensor in saved file: {path}")
        else:
            raise TypeError(f"Unsupported saved SNR format in {path} (expect Tensor or dict).")
        return SavedSNRPath(snrs=snrs, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown snrpath name: {name!r}. Expected one of: linear, lognormal, saved.")

def build_discrete_path_from_kw(k_steps, device='cpu', **kwargs):
    """Helper to build a discrete SNR path from keyword args."""
    snr_path = build_snr_path(kwargs, device=device)
    return snr_path.discrete_path(k_steps)
