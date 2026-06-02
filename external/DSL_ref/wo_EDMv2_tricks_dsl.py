"""Main class for Discrete Stochastic Localization
It requires a denoiser (some defined in models) that takes in embeddings and denoises to log p(x_i|z),
a (denoised) distribution over tokens.
The DSL class wraps the denoiser with an embedding layer, gets x hat from p(x_i|z), and provides methods for sampling.
We also implement an "Empirical" denoiser subclass, which uses the empirical distribution of the data.
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.log_normal import LogNormal
import math
import lightning as L
from torch.nn.utils import weight_norm
import wandb
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp

# Internal imports
import utils
import dataloader
import models.dit
import models.ema

# MDML imports
import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torchmetrics
import transformers
from torch import Tensor

LOG2 = math.log(2)


# NOTE: Gumbel-max sampling
def _sample_categorical(categorical_probs):
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(
       * x.shape,
       * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
            bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
            Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class SoftmaxConvert(nn.Module):
    def __init__(self, d, vocab_size, embed):
        """
        Convert the noisy embedding (dimension of input, z) to an embedding for the transformer (with dimension d).
        Intuition is that we want something closer to discrete diffusion. We look at how close noisy embedding
        is to nearby tokens, and we do a softmax over those, to get an output embedding (in transformer dim)
        :param d: Output dimension is the embedding size for the transformer
        :param vocab_size: Vocabulary size
        """
        super(SoftmaxConvert, self).__init__()
        self.embed = embed  # Noisy token embedding space
        self.embedding = nn.Linear(vocab_size, d, bias=False)  # Trainable weight matrix of size (vocab_size, d)
        self.beta = nn.Parameter(1. / torch.sqrt(torch.tensor(d)))

    def forward(self, z):
        """
        :param z: Input tensor of shape (batch, seq_length, noisy input embedding dimension)
        :return: Output tensor of shape (batch, seq_length, d)
        """
        softmaxed = F.softmax(self.beta * z @ self.embed.weight.T, dim=-1)  # Softmax over the last dimension (vocab_size)
        output = self.embedding(softmaxed)  # Linear transformation to output dimension d
        return output


class DSL(L.LightningModule):
    """Base class for Discrete Stochastic Localization.
    The denoising model must have a forward function that outputs x_hat(z). Optionally, it can have logits(z).
    that takes a batch of sequence embeddings and returns denoised sequences.
    """
    def __init__(
            self, 
            config, 
            tokenizer: transformers.PreTrainedTokenizer, 
            embed=None):
        super().__init__()
        self.save_hyperparameters()  # Use by lightning for saving/loading
        self.config = config

        # Denoiser
        if self.config.backbone == 'dit':
            self.backbone = models.dit.DIT(config)
        else:
            raise ValueError(
                f'Unknown backbone: {self.config.backbone}'
            )
        
        self.tokenizer = tokenizer
        self.vocab_size = config.data.vocab_size
        self.hidden_size = config.model.dim_embed # token embedding size, not the transformer hidden size

        # Token emebdding
        # BUG: possible issue with the following way of implementing unit norm
        if embed is None:
            self.embed = nn.Embedding(self.vocab_size, self.hidden_size) # the initialazition of nn.Embed should be a xxx random matrix, so full-rank matric, when token embedding is learned during training, low-rank structure should appear
        else:
            self.embed = embed
        self.embed = weight_norm(self.embed, name='weight', dim=0)
        with torch.no_grad():  # reparametrize to direction/norm. Set unit norm and disable grad on norm.
            self.embed.weight_g.fill_(1.)
            self.embed.weight_g.requires_grad_(False)
        self.convert = SoftmaxConvert(config.model.dim_h, self.vocab_size, self.embed)
        if hasattr(self.backbone, 'embed') and self.backbone.embed is None:
            self.backbone.embed = self.embed  # May be used for weight tying
        
        # Training and sampling noise levels
        self.t_max = config.training.t_max

        # Metrics related
        # metrics are automatically reset at end of epoch
        # NOTE: just automatically record the metrics
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(),
            'bpd': BPD(),
            'ppl': Perplexity(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        
        self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path
        # NOTE: they evaluate Gen-PPL each val epoch?
        # generative perplexity
        self.gen_ppl_metric = Perplexity()
        self.eval_model_tokenizer = transformers.AutoTokenizer.\
            from_pretrained(self.gen_ppl_eval_model_name_or_path)
        if self.eval_model_tokenizer.pad_token is None:
            self.eval_model_tokenizer.pad_token =\
                self.eval_model_tokenizer.eos_token
            self.eval_model_tokenizer.pad_token_id =\
                self.eval_model_tokenizer.eos_token_id

        # EMA training
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters()),
                decay=self.config.training.ema)
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        # NOTE: something here are not very clear here, check it out!
        self.sampling_eps = self.config.training.sampling_eps # NOTE: what is this???
        self.time_conditioning = self.config.time_conditioning # NOTE: what is this???
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None


    @property
    def norm_weight(self):
        """Get normalized embedding weight matrix."""
        return self.embed(torch.arange(self.vocab_size, device=self.device))

   
    # Will call this hook when restart from the checkpoint. 
    def on_load_checkpoint(self, checkpoint):
        """When restart from checkpoint, resume EMA and also number of epochs and batches have been done."""
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
                'current']['completed']

    # Will call this hook when saving the checkpoint.
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
                'completed'] = checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                        'optimizer']['step']['total'][
                            'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
                'completed'] = checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                        'optimizer']['step']['current'][
                            'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
                '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                        'optimizer']['step']['total']['completed']

        # NOTE: check training dataloader's data engine, saving its random_state, 
        # reason why do this: 这样的意义是：采样器的随机状态对于重现性和确保在恢复训练时数据采样的一致性非常重要，
        # 尤其是当数据需要随机打乱顺序时，通过保存采样器状态可以保证在恢复后继续保持相同的随机序列
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.\
                train_dataloader.sampler.state_dict()
            checkpoint['sampler'][
                'random_state'] = sampler_state_dict.get(
                    'random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    # Will call this hook before training start, 
    def on_train_start(self):
        """To initialize ema and dataloader for later restart."""
        # print(f'before training start...')
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        # NOTE: checking whether using distributed training, and select different training dataloader sampler 数据采样器不同
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        
        # NOTE: loop all dataloader, reset data sampler and resume the training according to last training status. 
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)

            if (distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': (self.fast_forward_batches
                                * self.config.loader.batch_size)})
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False))
                    # persistent_workers=True))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters()))

    def forward(self, z):
        """reconstruct clean batch of sequences of token embeddings from noisy ones.
        params z: tensor of shape (batch, seq_len), noisy sentences"""
        # diffusion forward needs float32
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.logits(z)
        p = logits.softmax(dim=-1)
        x_hat = torch.matmul(p.float(), self.norm_weight)
        return x_hat

    def logits(self, z):
        """Logits for a particular z (embedding or noisy embedding of tokens).
        Implements a backup heuristic that works with Empirical denoiser.
        :params z: embedding or noisy embedding for tokens
        :return: logits of z, of shape (batch, seq_length, #token)"""
        if self.backbone is None:
            return torch.matmul(z, self.norm_weight.T)
        else:
            z_convert = self.convert(z)
            mags = z.norm(dim=-1)
            # NOTE: no need bf16 control here, since the backbone already has this.
            return self.backbone(z_convert, mags)
            # z_convert_norm = z_convert.norm(dim=-1, keepdim=True)
            # z_convert_norm = torch.clamp_min(z_convert_norm, 1e-12)
            # normalized_z_convert = z_convert / z_convert_norm
            # return self.backbone(normalized_z_convert, mags)

    # SCHEDULE
    def uniform(self, t_max, n_steps):
        """Uniform in time, or SNR"""
        ts = torch.linspace(0., t_max, n_steps, device=self.device)
        return ts
    
    # SAMPLING ROUTINES
    def sample(self, s, t):
        """Given sequence token ids, sample embedded sequences with noise.

        Parameters:
        - s: tensor of token ids, shape (batch, seq_length)
        - t: scalar, (batch,), or (batch, seq_length) tensor, representing noise levels.

        Returns:
        - Noisy embeddings, shape (batch, seq_length, d)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
        if t.dim() == 0:  # scalar
            t = t.view(1, 1, 1)  # Make it compatible for broadcasting
        elif t.dim() == 1:  # (batch,)
            t = t.view(-1, 1, 1)  # Match the (batch, seq_length, d) shape
        elif t.dim() == 2:  # (batch, seq_length)
            t = t.view(t.size(0), t.size(1), 1)  # Match the (batch, seq_length, d) shape

        z = self.embed(s)
        return t * z + torch.sqrt(t) * torch.randn_like(z)
    
    @torch.no_grad()
    def simulate(self, batch_size, seq_length, ts, a=1., history=False):
        """Simulate the model, dz = x hat dt + dW. Can select "a" to get equivalent SDE family, or ODE.
        dz = (a x_hat +(1-a) z/t) dt + (2 a - 1) dW, a=1 is standard, a=1/2 gives ODE.
        (only a >=1/2 works in derivation)
        """
        # z = torch.zeros((batch_size, seq_length, self.hidden_size), device=self.device)  # initial condition
        small_const = 1e-4
        z = torch.randn((batch_size, seq_length, self.hidden_size), device=self.device) * small_const
        if history:
            zs = torch.zeros((batch_size, seq_length, self.hidden_size, len(ts)), device=self.device)
        for i, dt in enumerate(ts[1:] - ts[:-1]):
            z = z + a * self(z) * dt + (2*a-1) * math.sqrt(dt) * torch.randn_like(z)
            if history:
                zs[..., i+1] = z
        if history:
            return zs
        return z
   
    def curvature(self, z):
        """
        使用梯度估计速度场变化的曲率。
        z: 形状为(B, T, d)的张量
        返回: 形状为(B, T, 1)的张量
            每个输出[b,t] = norm_{d dim} f(z) · ∇_{z} f_{t,d}(z)
        """
        B, T, D = z.shape
        z_flat = z.reshape(B, -1)  # (B, -1)
        
        z_flat = z_flat.detach()
        z_flat.requires_grad_(True)
        f_flat = self(z_flat.reshape(B, T, D)).reshape(B, -1)

        # 计算所有f输出的jvp (f(z) · ∇ f_i)
        _, out_flat = jvp(lambda z_: self(z_.reshape(B, T, D)),
                        (z_flat,), (f_flat,))
        
        return out_flat.reshape(B, T, -1).norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def simulate_per_token(self, batch_size, seq_length, n_iter=50, verbose=False, history=False):
        """
        使用基于曲率的自适应步长进行per-token SDE模拟。
        每个token的动力学步长由当前曲率决定："曲率"估计了速度场(x_hat)在x_hat方向步进时的变化速度。
        如果变化快(对于一个token)，我们使用较小的SNR步长。
        
        参数:
            batch_size: 批次大小
            seq_length: 序列长度
            ts: 时间步长序列
            a: SDE族参数(默认为1)，为保持与simulate接口一致而保留
            history: 是否返回历史状态
        
        返回:
            生成的嵌入序列
        """
        # 初始化参数
        gamma_min = torch.tensor(0.01, device=self.device)
        gamma_max = torch.tensor(self.t_max, device=self.device) 
        
        # 每个token都有不同的gamma值
        gamma = gamma_min * torch.ones((batch_size, seq_length, 1), device=self.device)
        delta = torch.tensor(0.01, device=self.device)  # step size, before conditioning
        delta_max = gamma_max / n_iter  # 最大步长保持在训练范围内
        
        # z的初始条件
        small_const = 0. # 1e-4
        z = torch.randn((batch_size, seq_length, self.hidden_size), 
                    device=self.device) * small_const
        
        if history:
            zs = torch.zeros((batch_size, seq_length, self.hidden_size, n_iter+1), device=self.device)
            zs[..., 0] = z
        
        # verbose = False  # 设置为True以获取更多调试信息
        
        for i in range(n_iter):
            # # 计算当前x_hat用于曲率估计
            # x_hat = self(z)
            # 计算曲率 - 速度场变化的速率
            kappa = self.curvature(z)
            # 基于曲率的自适应步长 - 曲率高的token使用较小步长
            h = torch.minimum(delta / (kappa + 1e-6), delta_max)
            # 更新z - 注意每个token位置都有不同的步长h
            z += h * self(z) + torch.sqrt(h) * torch.randn_like(z)
            gamma += h
            
            if verbose and i % 10 == 0:
                print(f'Iteration {i}: h min/max: {h.min().item():.3f}, {h.max().item():.3f}')
                print(f"gamma: {gamma[0,:5,0].cpu()}")  # 显示不同token的gamma路径
                print(f'** {self.tokenizer.decode(self.logits(z)[0].argmax(dim=-1))}')
            
            if history:
                zs[..., i+1] = z
        
        if history:
            return zs
        return z


    # ESTIMATION ROUTINES
    @torch.no_grad()
    def mse(self, s, ts):
        """MSE for recovering embeddings of x from noisy embeddings of x, for given SNR values."""
        mse_x = torch.zeros((s.shape[0], len(ts)), device=s.device)
        x_embed = self.embed(s)
        for i, t in enumerate(ts):
            z = self.sample(s, t) # only sample noisy tokens with noise level t
            x_hat = self(z)
            error = (x_embed - x_hat).flatten(start_dim=1)
            mse_x[:, i] = torch.einsum('ij,ij->i', error, error)
        return mse_x  # MSE of epsilon estimate, per sample

    @torch.no_grad()
    def nll_x(self, x, ts):
        """-log p(x) for a single sample, x"""
        mses = self.mse(x, ts)
        # print(f"ts: {ts}")
        # print(f"MSEs: {mses.mean(dim=0)}")
        return 0.5 * torch.trapz(mses, ts) / math.log(2) / x.shape[1]  # Interpretable as entropy (bits per token)

    @torch.no_grad()
    def debug_nll_x(self, x, ts):
        """计算改进的NLL估计，包括[10,∞)区间的贡献"""
        # 计算数值积分[0,10]部分
        mses = self.mse(x, ts)
        integral_numerical = torch.trapz(mses, ts)
        
        # 估计[10,∞)部分的积分贡献
        t_max = ts[-1]
        mse_at_t_max = mses[:, -1]
        
        # 选择一种衰减假设：这里使用多项式衰减1/t²
        c = mse_at_t_max * t_max**2
        integral_tail = c / t_max
        
        # 合并两部分积分
        total_integral = integral_numerical + integral_tail
        
        # 计算NLL
        nll_bits = 0.5 * total_integral / math.log(2) / x.shape[1]
        
        # 输出中间结果以便调试
        # print(f"数值积分[0,{t_max}]: {integral_numerical.mean().item():.2f}")
        # print(f"尾部积分[{t_max},∞): {integral_tail.mean().item():.2f}")
        # print(f"总积分: {total_integral.mean().item():.2f}")
        # print(f"NLL (bits per token): {nll_bits.mean().item():.4f}")
        
        return nll_bits

    @torch.no_grad()
    def nll(self, x, ts):
        """Estimate of negative log likelihood for a batch, - E_x [log p(x)], the data distribution."""
        return self.nll_x(x, ts).mean()  # bits per token
        # return self.debug_nll_x(x, ts).mean()  # bits per token

    def plaid_style_nll(self, x, ts):
        """Use plaid reconstruction style for tail bound, the rest are the same as self.nll()"""
        batch_size = x.shape[0]
        # print("x.shape[1] = " + str(x.shape[1]))
        # use 1/8 for reconstruction loss -- an upper bound for the tail [SNR_max, inf)
        reconst_bs = batch_size // 16
        reconst_bs += int(np.random.binomial(1, (batch_size % 16) / 16.))
        avg_reconst_bs = batch_size / 16.
        # reconst_bs = batch_size
        # avg_reconst_bs = batch_size

        # # 1. Reconstruction loss for high SNR region
        z_low = self.sample(x[:reconst_bs], torch.full((reconst_bs,), self.t_max, device=x.device))
        logits_low = self.logits(z_low).transpose(1, 2)
        reconst_loss = nn.functional.cross_entropy(logits_low, x[:reconst_bs], reduction='none').mean(dim=1)
        nll_reconst = reconst_loss.sum() / (avg_reconst_bs * math.log(2))
        # print("nll reconst = " + str(nll_reconst))
        # return nll_reconst, nll_reconst, 0.

        # 2. Diffusion loss, normal NLL calculation here
        mses_diffusion = self.mse(x[reconst_bs:], ts)
        diffusion_integral = torch.trapz(mses_diffusion, ts)
        nll_diffusion = 0.5 * diffusion_integral.sum() / ((batch_size - reconst_bs) * math.log(2) * x[reconst_bs:].shape[1])
        # print("nll diffusion " + str(nll_diffusion))
        
        nll = nll_reconst + nll_diffusion
        return nll, nll_reconst, nll_diffusion


    def plaid_style_nll_float64(self, x, ts):
        """Use plaid reconstruction style for tail bound, the rest are the same as self.nll()"""
        batch_size = x.shape[0]
        # use 1/8 for reconstruction loss -- an upper bound for the tail [SNR_max, inf)
        # NOTE: do I really need a split of data to calculate different parts of the NLL?
        reconst_bs = batch_size // 16
        reconst_bs += int(np.random.binomial(1, (batch_size % 16) / 16.))

        # # 1. Reconstruction loss for high SNR region
        z_low = self.sample(x, torch.full((batch_size,), self.t_max, device=x.device))
        logits_low = self.logits(z_low).transpose(1, 2).double() # NLL calculation needs float64

        # calculate the CE loss in float64
        reconst_loss = nn.functional.cross_entropy(logits_low, x, reduction='none').double().mean(dim=1)
        nll_reconst = reconst_loss.sum().double() / (batch_size * math.log(2))
        # print("nll reconst = " + str(nll_reconst))

        # 2. Diffusion loss, normal NLL calculation here
        mses_diffusion = self.mse(x, ts).double()
        diffusion_integral = torch.trapz(mses_diffusion, ts.double())
        # nll_diffusion = 0.5 * diffusion_integral.sum().double() / (batch_size * math.log(2) * x.shape[1])
        nll_diffusion = 0.5 * diffusion_integral.sum().double() / (batch_size * math.log(2) * x.shape[1])
        # print("nll diffusion " + str(nll_diffusion))
        
        nll = nll_reconst + nll_diffusion
        return nll, nll_reconst, nll_diffusion


    def perplexity(self, x, ts):
        """Estimate of perplexity for a batch, PPL = 2 ** NLL."""
        return 2 ** (self.nll(x, ts))
    
    def logistic_integrate(self, npoints, loc, scale, clip=4.):
        """Return sample points and weights for integration, using
        a truncated logistic distribution as the base, and importance weights.
        
        Parameters:
        - npoints: number of sample points
        - loc: location parameter for logistic distribution
        - scale: scale parameter for logistic distribution
        - clip: truncation parameter in logit space
        
        Returns:
        - t_samples: sampled time points
        - weights: importance weights for each sample
        """
        # Move parameters to device
        loc = torch.tensor(loc, device=self.device)
        scale = torch.tensor(scale, device=self.device)
        clip = torch.tensor(clip, device=self.device)
        
        # IID samples from uniform, use inverse CDF to transform to target distribution
        ps = torch.rand(npoints, device=self.device)
        # Scale quantiles to the truncated range [sigmoid(-clip), sigmoid(clip)]
        ps = torch.sigmoid(-clip) + (torch.sigmoid(clip) - torch.sigmoid(-clip)) * ps
        # Using quantile function (inverse CDF) for logistic distribution
        t_samples = loc + scale * torch.logit(ps)
        
        # Calculate importance weights
        weights = scale * torch.tanh(clip / 2) / (torch.sigmoid((t_samples - loc)/scale) * torch.sigmoid(-(t_samples - loc)/scale))
        
        return t_samples, weights

    @torch.no_grad()
    def nll_x_importance_sampling(self, x, num_samples=100):
        """
        -log p(x) for a single sample x using importance sampling with truncated logistic distribution
        
        Parameters:
        - x: input tensor of token ids
        - num_samples: number of samples for Monte Carlo estimation
        
        Returns:
        - Estimated negative log likelihood (bits per token)
        """
        # Choose logistic distribution parameters based on the MSE curve
        # These parameters should be tuned to match your MMSE curve
        logsnr_loc = 5.0  # Center of the logistic distribution (t value where MSE is in middle range)
        logsnr_scale = 2.0  # Scale of the logistic distribution
        clip = 4.0  # Truncation parameter (how many scale units from loc to truncate)
        
        # Get samples and weights using logistic integration
        t_samples, importance_weights = self.logistic_integrate(num_samples, logsnr_loc, logsnr_scale, clip)
        
        # Make sure t_samples is within valid range for model
        t_samples = torch.clamp(t_samples, min=1e-6, max=self.t_max)
        
        # Compute MSEs at sampled time points
        mse_samples = self.mse(x, t_samples)
        
        # Weight the MSE values by importance weights
        weighted_mses = mse_samples * importance_weights.unsqueeze(0)
        
        # Average the weighted samples to estimate the integral
        integral_estimate = weighted_mses.mean(dim=1)
        
        # Return in bits per token
        return 0.5 * integral_estimate / math.log(2) / x.shape[1]

    def on_train_epoch_start(self):
        """xxx.train() sets these submodules as training mode, then to do e.g. dropout, batch-normalization"""
        # print(f'on train epoch start...')
        self.backbone.train()
        # NOTE: no need to add encoder embedding here, since it doesn't separate training and eval tiem behavior

    # TRAINING (Pytorch lightning)
    def process_batch(self, batch):
        if isinstance(batch, dict):
            s = batch.get("input_ids", None)
            if s is None:
                raise ValueError("Batch dictionary does not contain 'input_ids'.")
        else:
            # this is for the empirical experiment here
            s = batch
        return s.to(self.device)

    '''
    def training_step(self, batch, batch_idx, skip_log=False):
        s = self.process_batch(batch)
        
        # # Uniform sample t
        # t = torch.rand(len(s), device=self.device) * self.t_max
        # Log-normal sample t
        mu, sigma = 1.4, 0.55 # 0-2区间：~9.92%（原参数仅0.06%）, 8-10区间：~6.02%（与原参数相近）, 2-4区间：~39.07%, 4-6区间：~26.74%
        # mu, sigma = 2.0, 0.55 # [0, 1]0.13, [1, 2]3.36%, [2, 3]8.90%, [3, 4]13.00%, [4, 5]14.28%, [5, 6]13.38%, [6, 7]11.40%, [7, 8]9.18%, [8, 9]7.16%, [9, 10]5.45%
        dist = LogNormal(mu, sigma)
        t = dist.sample((len(s),)).to(self.device)
        t = torch.clamp(t, 0, self.t_max)

        z = self.sample(s, t)  # TODO: different timestep each seq item? Low discrepancy sampler
        logits = self.logits(z).transpose(1, 2)  # batch, vocab, seq - arranged for CE
     
        # 计算每个样本的损失，但不要立即减少
        per_sample_loss = nn.functional.cross_entropy(logits, s, reduction='none')  # 形状: [batch_size, seq_len]
        per_sample_loss = per_sample_loss.mean(dim=1)  # 对每个样本取平均，形状: [batch_size]
        
        # 使用原始损失，不应用重加权
        loss = per_sample_loss.mean()  # 所有样本的平均损失
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # 每100批次分析一次不同噪声水平的损失分布
        if self.config.training.val_mse_plot and batch_idx % 100 == 0:  
            num_bins = 9
            # 创建噪声级别的分桶
            loss_bins = [[] for _ in range(num_bins)]
            
            # 将每个样本按噪声水平分组
            for i, ti in enumerate(t):
                bin_idx = min(int((ti / self.t_max) * num_bins), num_bins - 1)
                loss_bins[bin_idx].append(per_sample_loss[i].item())
            
            # 记录每个分桶的平均损失
            for i, losses in enumerate(loss_bins):
                if losses:  # 确保分桶不为空
                    bin_avg_loss = sum(losses) / len(losses)
                    self.log(f"train/loss_t_{i}", bin_avg_loss, sync_dist=True)
                    
            # 可选: 记录损失方差最大和最小的分桶
            bin_avgs = [sum(b)/len(b) if b else 0 for b in loss_bins]
            valid_avgs = [avg for avg in bin_avgs if avg > 0]
            if valid_avgs:
                min_loss = min(valid_avgs)
                max_loss = max(valid_avgs)
                self.log("train/loss_t_min", min_loss, sync_dist=True)
                self.log("train/loss_t_max", max_loss, sync_dist=True)
                self.log("train/loss_t_ratio", max_loss/min_loss if min_loss > 0 else 0, sync_dist=True)

        if self.config.training.val_mse_plot and self.global_step % 200 == 0:
            # 只用一小部分样本来计算，例如当前批次的前8个样本
            small_s = s[:min(8, s.shape[0])]
            n_steps = 50  # 使用较少的步数来减少计算量
            ts = self.uniform(self.t_max, n_steps=n_steps)
            
            # 计算 MSE 曲线
            mses = self.mse(small_s, ts).mean(dim=0)
            
            # 使用 wandb 记录图表（如果使用 wandb）
            if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                wandb_logger = self.logger.experiment
                fig_mse = utils.plot_mse(ts.cpu().numpy(), mses.cpu().numpy())
                wandb_logger.log({"train_mses": wandb.Image(fig_mse)})
                plt.close(fig_mse)  # 添加这一行来关闭图表

        self.log("train/loss", loss, sync_dist=True)
        return loss
    '''

    def training_step(self, batch, batch_idx, skip_log=False):
        s = self.process_batch(batch)

        batch_size = s.shape[0]
        # # Use 1/16 for reconstruction loss -- for an upper bound for the tail [SNR_max, inf)
        # reconst_bs = batch_size // 16
        # reconst_bs += int(np.random.binomial(1, (batch_size % 16) / 16.))
        # avg_reconst_bs = batch_size / 16.
        # # Calculate reconstruction loss for high SNR region
        # z_low = self.sample(s[:reconst_bs], torch.full((reconst_bs,), self.t_max, device=s.device))
        # logits_low = self.logits(z_low).transpose(1, 2)
        # reconst_loss = nn.functional.cross_entropy(logits_low, s[:reconst_bs], reduction='none').mean(dim=1)
        # weighted_reconst_loss = (self.config.training.reconst_weight * reconst_loss).sum() / avg_reconst_bs
        # No reconst loss
        reconst_bs = 0
        weighted_reconst_loss = 0.0
        
        # 获取当前梯度累积步数
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        # 判断当前是否为累积的最后一步
        is_last_accum_step = ((batch_idx + 1) % accumulate_grad_batches == 0) or (batch_idx + 1 == self.trainer.num_training_batches)
        
        # Sample t for diffusion samples (all remaining samples in batch)
        diff_batch_size = batch_size
    
        # # Uniform sample t
        # t = torch.rand(len(s), device=self.device) * self.t_max
        # Log-normal sample t
        # mu, sigma = 2., 0.5
        mu, sigma = 1.4, 0.55
        mu = torch.tensor(mu, device=self.device)
        sigma = torch.tensor(sigma, device=self.device)
        dist = LogNormal(mu, sigma)
        t = dist.sample((diff_batch_size,))
        t = torch.clamp(t, 0, self.t_max)

        # Calcualte diffusion loss, make sure only use the rest of the batch for diffusion_loss calculation -- s[reconst_bs:]
        # 非最后一步梯度累积，使用no_sync()跳过梯度同步
        if not is_last_accum_step and accumulate_grad_batches > 1 and hasattr(self, 'trainer'):
            # 使用DDP的no_sync()上下文管理器跳过不必要的gradient同步
            # 首先获取实际的DDP模型
            ddp_model = None
            if hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'model'):
                ddp_model = self.trainer.strategy.model
            
            if ddp_model is not None and hasattr(ddp_model, 'no_sync'):
                with ddp_model.no_sync():
                    z = self.sample(s[reconst_bs:], t)
                    logits = self.logits(z).transpose(1, 2)
                    per_sample_loss = nn.functional.cross_entropy(logits, s[reconst_bs:], reduction='none').mean(dim=1)
                    diffusion_loss = per_sample_loss.sum() / diff_batch_size
            else:
                # 如果模型没有no_sync方法（非DDP情况），正常处理
                z = self.sample(s[reconst_bs:], t)
                logits = self.logits(z).transpose(1, 2)
                per_sample_loss = nn.functional.cross_entropy(logits, s[reconst_bs:], reduction='none').mean(dim=1)
                diffusion_loss = per_sample_loss.sum() / diff_batch_size
        else:
            # 最后一步梯度累积，正常执行带同步的backward
            z = self.sample(s[reconst_bs:], t)
            logits = self.logits(z).transpose(1, 2)
            per_sample_loss = nn.functional.cross_entropy(logits, s[reconst_bs:], reduction='none').mean(dim=1)
            diffusion_loss = per_sample_loss.sum() / diff_batch_size
        
        # Total loss
        loss = diffusion_loss + weighted_reconst_loss

        # 记录日志逻辑保持不变
        if batch_idx % 100 == 0:  
            self.log("train/loss", loss, sync_dist=True)
            
            t_uniform = self.uniform(self.t_max, n_steps=100)
            # train_nll = self.nll(s, t_uniform)
            train_nll, nll_reconst, nll_diffusion = self.plaid_style_nll_float64(s, t_uniform)
            # print('train nll t_uniform: {}'.format(t_uniform))
            self.log("train/nll", train_nll, sync_dist=True)
            self.log("train/nll_reconstruct", nll_reconst, sync_dist=True)

            if self.config.training.val_mse_plot:
                num_bins = 9
                loss_bins = [[] for _ in range(num_bins)]
                
                for i, ti in enumerate(t):
                    bin_idx = min(int((ti / self.t_max) * num_bins), num_bins - 1)
                    loss_bins[bin_idx].append(per_sample_loss[i].item())
                
                for i, losses in enumerate(loss_bins):
                    if losses:
                        bin_avg_loss = sum(losses) / len(losses)
                        self.log(f"train/loss_t_{i}", bin_avg_loss, sync_dist=True)
                        
                bin_avgs = [sum(b)/len(b) if b else 0 for b in loss_bins]
                valid_avgs = [avg for avg in bin_avgs if avg > 0]
                if valid_avgs:
                    min_loss = min(valid_avgs)
                    max_loss = max(valid_avgs)
                    self.log("train/loss_t_min", min_loss, sync_dist=True)
                    self.log("train/loss_t_max", max_loss, sync_dist=True)
                    self.log("train/loss_t_ratio", max_loss/min_loss if min_loss > 0 else 0, sync_dist=True)

                small_s = s[:min(8, s.shape[0])]
                n_steps = 50
                ts = self.uniform(self.t_max, n_steps=n_steps)
                mses = self.mse(small_s, ts).mean(dim=0)
                if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                    wandb_logger = self.logger.experiment
                    fig_mse = utils.plot_mse(ts.cpu().numpy(), mses.cpu().numpy())
                    wandb_logger.log({"train_mses": wandb.Image(fig_mse)})
                    plt.close(fig_mse)
        return loss

    def on_validation_epoch_start(self):
        # print('on validation epoch start...')
        if self.ema:
            self.ema.store(itertools.chain(self.backbone.parameters()))
            self.ema.copy_to(itertools.chain(self.backbone.parameters()))
        self.backbone.eval()
        # NOTE: every valid epoch, the following values will be reset to zero, check it before validation epoch
        # NOTE: Yunshu, in DSL, we don't calcuate the nll in the following way, 
        # so need to check if it's reasonable to still uncomment the following code
        # assert self.valid_metrics.nll.mean_value == 0
        # assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx, n_steps=100):
        self.log("number steps", n_steps, sync_dist=True)
        # Log your scalar metrics with the Lightning built-in log
        val_loss = self.training_step(batch, batch_idx, skip_log=True)
        self.log("val/loss", val_loss, sync_dist=True)
        
        s = self.process_batch(batch)
        ts = self.uniform(self.t_max, n_steps=n_steps) # Uniformly sample t
        # mu, sigma = 1.4, 0.55
        # dist = LogNormal(mu, sigma)
        # t = dist.sample((len(s),)).to(self.device)
        # ts = torch.clamp(t, 0, self.t_max)

        # plaid_val_nll, nll_reconst, nll_diffusion = self.plaid_style_nll(s, ts)
        plaid_val_nll, nll_reconst, nll_diffusion = self.plaid_style_nll_float64(s, ts)
        # print(f'plaid style val_nll = {plaid_val_nll}')
        # print(f'nll_reconst = {nll_reconst}')
        # print(f'nll_diffusion = {nll_diffusion}')
        # val_nll = self.nll(s, ts)
        # print(f'val_nll = {val_nll}')
        self.log("val/nll", plaid_val_nll, sync_dist=True)
        self.log("val/nll_diffusion", nll_diffusion, sync_dist=True)
        self.log("val/nll_reconstruct", nll_reconst, sync_dist=True)

        val_ppl = 2 ** plaid_val_nll
        self.log("val/ppl", val_ppl, sync_dist=True)

        if self.config.training.val_mse_plot and batch_idx == 0:
            # Calculate MSE curve
            mses = self.mse(s, ts).mean(dim=0)
            wandb_logger = self.logger.experiment
            fig_mse = utils.plot_mse(ts.cpu().numpy(), mses.cpu().numpy())
            wandb_logger.log({"mses": wandb.Image(fig_mse)})
            plt.close(fig_mse)  # 添加这一行来关闭图表
            # Plot and log correlation heatmap
            correlation_mat = self.norm_weight.data @ self.norm_weight.data.T
            correlation_mat = correlation_mat.float()
            fig_corr_mat = utils.plot_correlation_mat(
                correlation_mat.detach().cpu().numpy(),
                title="vocab cosine similarity"
            )
            wandb_logger.log({"correlations": wandb.Image(fig_corr_mat)})
            plt.close(fig_corr_mat)  # 添加这一行来关闭图表

        # Return the val_loss so it is properly aggregated in Lightning
        return val_loss

    def on_validation_epoch_end(self):
        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
             and self.config.eval.generate_samples):
            print(f'on validation epoch end...')
            samples, text_samples = None, None
            for _ in range(
                self.config.sampling.num_sample_batches):

                # NOTE: change to my own sampler function! 
                samples = self._sample()
                
                # Decode the samples to be re-tokenized by eval model 
                text_samples = self.tokenizer.batch_decode(samples)
                # print(f'text samples = {text_samples}')
                if self.config.eval.compute_generative_perplexity:
                    print(f'gen ppl .....')
                    self.compute_generative_perplexity(text_samples) # TODO: implement the function
            if self.trainer.global_rank == 0 and hasattr(
                self.trainer.logger, 'log_table'):
                # Log the last generated samples
                text_samples = text_samples[ : self.config.sampling.num_sample_log]
                self.trainer.logger.log_table(
                    key=f'samples@global_step{self.global_step}',
                    columns=['Generated Samples'],
                    data=[[s] for s in text_samples])
            if self.config.eval.compute_generative_perplexity:
                self.log('val/gen_ppl',
                         self.gen_ppl_metric,
                         on_epoch=True,
                         on_step=False,
                         sync_dist=True)
        if self.ema: 
            self.ema.restore(itertools.chain(self.backbone.parameters()))

    def test_step(self, batch, batch_idx, n_steps=100):
        s = self.process_batch(batch)
        if s.shape[0] == 0:
            print("WARNING!!! EMPTY BATCH!!!")
            return  

        self.log("test/loss", self.training_step(batch, batch_idx, skip_log=True), sync_dist=True)
        ts = self.uniform(self.t_max, n_steps=n_steps)
        # self.log("test/nll", self.nll(s, ts), sync_dist=True)
        self.log("test/nll", self.plaid_style_nll(s, ts)[0], sync_dist=True)

    # def test_step(self, batch, batch_idx, n_steps=100):
    #     self.log("test/loss", self.training_step(batch, batch_idx, skip_log=True), sync_dist=True)
    #     s = self.process_batch(batch)
    #     # Use importance sampling for NLL estimation
    #     self.log("test/nll", self.nll_x_importance_sampling(s, num_samples=n_steps).mean(), sync_dist=True)

    def configure_optimizers(self):
        # TODO (yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                   self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]


    @torch.no_grad()
    def eval_retokenize(self, text_samples, max_length):
        """Retokenizes samples for the eval model.
        Args:
            text_samples: List of sentences generated by the model.
            NOTE: Yunshu, is this token itself???? I think it's not IDs
        Returns:
            samples: Samples re-tokenized for the eval model
            attn_mask: Attention mask for the eval model
            eval_context_size: Size of the context for the eval model
        """
        if 'llama2' in self.gen_ppl_eval_model_name_or_path:
            tokenizer_kwargs = {
                'text_samples': text_samples,
                'return_tensors': 'pt',
                'return_token_type_ids': False,
                'return_attention_mask': True,
                'truncation': True,
                'padding': True,
                'max_length': max_length,
            }
            eval_context_size = 4096
        else:
            tokenizer_kwargs = {
                'return_tensors': 'pt',
                'return_token_type_ids': False,
                'return_attention_mask': True,
                'truncation': True,
                'padding': True,
                'max_length': max_length,
            }
            eval_context_size = 1024
        samples = self.eval_model_tokenizer(
            text_samples, ** tokenizer_kwargs)
        attn_mask = samples['attention_mask']
        samples = samples['input_ids']
        if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
            attn_mask = attn_mask.to(self.device)
            samples = samples.to(self.device)
        
        return samples, attn_mask, eval_context_size


    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5):
        logger = utils.get_logger("DSL.sampling._sample")
        
        # 使用本地GPU
        device = self.device
        logger.info(f"Using device for sampling: {device}")
        # 批次大小仅针对本地GPU
        batch_size_per_gpu = self.config.loader.eval_batch_size
        
        # 设置采样步数
        if num_steps is None:
            num_steps = self.config.sampling.steps
        
        # 确保使用'uniform_t'方法
        ts = self.uniform(t_max=self.t_max, n_steps=num_steps).to(device)
        logger.info(f"Time steps tensor on device: {ts.device}")
        
        # 禁用分布式同步 - 关键修改
        with torch.cuda.device(device):
            if self.config.sampling.sampling_method == 'per_sentence':
                z = self.simulate(batch_size=batch_size_per_gpu, seq_length=self.config.model.length, ts=ts)
            elif self.config.sampling.sampling_method == 'per_token':
                z = self.simulate_per_token(batch_size=batch_size_per_gpu, seq_length=self.config.model.length, n_iter=num_steps)
            logger.info(f"Simulated z tensor on device: {z.device} with shape {z.shape}")
            # 获取logits并执行argmax
            logits = self.logits(z) #.double() # float()
            logger.info(f"Logits tensor on device: {logits.device} with shape {logits.shape}")
            s = logits.argmax(dim=-1)
            logger.info(f"Final samples tensor on device: {s.device} with shape {s.shape}")
        
        return s


    # @torch.no_grad()
    # def compute_generative_perplexity(
    #     self,
    #     text_samples: typing.List[str],
    #     retokenize: bool = True,
    #     max_length: typing.Optional[int] = None) -> None:
    #     """Compute the generative perplexity of the model.
    #     Args:
    #         text_samples: List of sentences generated by the model.
    #     Returns:
    #         Perplexity of the generated text under a different
    #         pre-trained AR model (e.g., GPT2).
    #     """
    #     print(f'Computing generative perplexity....')
    #     os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    #     eval_model = transformers.AutoModelForCausalLM.from_pretrained(
    #         self.gen_ppl_eval_model_name_or_path).eval()
    #     if max_length is None:
    #         max_length = self.config.model.length
    #     if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
    #         eval_model = eval_model.to(self.device)
    #     # Re-tokenize using eval model's tokenizer
    #     if retokenize:
    #         (samples, attn_mask,
    #          eval_context_size) = self.eval_retokenize(
    #             text_samples, max_length=max_length)
    #     else:
    #         samples = text_samples
    #         attn_mask = torch.ones(samples.shape).to(self.device)
    #         eval_context_size = samples.shape[-1]
    #     batch_size = min(self.config.eval.perplexity_batch_size,
    #                      samples.shape[0])
    #     num_batches = samples.shape[0] // batch_size
    #     for i in range(num_batches):
    #         _samples = torch.split(
    #             samples[i * batch_size: (i + 1) * batch_size],
    #             eval_context_size,
    #             dim=-1)
    #         _attn_mask = torch.split(
    #             attn_mask[i * batch_size: (i + 1) * batch_size],
    #             eval_context_size,
    #             dim=-1)
    #         for (sample_chunk, attn_mask_chunk) in zip(_samples, _attn_mask):
    #             logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
    #             logits = logits.transpose(-1, -2)
    #             nlls = F.cross_entropy(logits[..., :-1],
    #                                    sample_chunk[..., 1:], 
    #                                    reduction='none')
    #             first_eos = (sample_chunk == self.eval_model_tokenizer\
    #                          .eos_token_id).cumsum(-1) == 1
    #             token_mask = (sample_chunk != self.eval_model_tokenizer.eos_token_id)
    #             self.gen_ppl_metric.update(nlls, first_eos[..., 1:] + token_mask[..., 1:])


    @torch.no_grad()
    def compute_generative_perplexity(
        self,
        text_samples: typing.List[str],
        retokenize: bool = True,
        max_length: typing.Optional[int] = None) -> None:
        """Compute the generative perplexity of the model with proper distributed support."""
        print(f'Computing generative perplexity....')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # 分布式环境检查
        is_distributed = torch.distributed.is_initialized()
        if is_distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            # 只在主进程上加载模型或在每个进程上加载但明确指定设备
            if rank == 0:
                print(f"Process {rank}/{world_size} loading evaluation model...")
        else:
            rank = 0
            world_size = 1
        
        # 添加同步点
        if is_distributed:
            torch.distributed.barrier()
        
        # 加载模型 - 可以考虑只在rank=0的进程上加载
        # 或者确保每个进程加载模型到正确的设备上
        eval_model = None
        if rank == 0 or not is_distributed:
            eval_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.gen_ppl_eval_model_name_or_path).eval()
            # 明确指定设备
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            eval_model = eval_model.to(device)
        
        if max_length is None:
            max_length = self.config.model.length
        
        # 重新分配文本样本，每个进程只处理自己的部分
        if is_distributed:
            # 计算每个进程应处理的样本数量
            samples_per_process = max(1, len(text_samples) // world_size)
            start_idx = rank * samples_per_process
            end_idx = start_idx + samples_per_process if rank < world_size - 1 else len(text_samples)
            process_text_samples = text_samples[start_idx:end_idx] if start_idx < len(text_samples) else []
        else:
            process_text_samples = text_samples
        
        # 重新标记化
        local_metric = Perplexity().to(self.device)
        
        if process_text_samples:  # 确保有样本要处理
            if retokenize:
                (samples, attn_mask,
                    eval_context_size) = self.eval_retokenize(
                    process_text_samples, max_length=max_length)
            else:
                samples = process_text_samples
                attn_mask = torch.ones(samples.shape).to(self.device)
                eval_context_size = samples.shape[-1]
            
            # 批处理计算
            if hasattr(samples, 'shape') and samples.shape[0] > 0:
                batch_size = min(self.config.eval.perplexity_batch_size, samples.shape[0])
                num_batches = max(1, samples.shape[0] // batch_size)
                
                for i in range(num_batches):
                    if batch_size > 0:  # 确保batch_size不为0
                        _samples = torch.split(
                            samples[i * batch_size: (i + 1) * batch_size],
                            eval_context_size,
                            dim=-1)
                        _attn_mask = torch.split(
                            attn_mask[i * batch_size: (i + 1) * batch_size],
                            eval_context_size,
                            dim=-1)
                        
                        for (sample_chunk, attn_mask_chunk) in zip(_samples, _attn_mask):
                            if eval_model is not None:  # 确保模型已加载
                                logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
                                logits = logits.transpose(-1, -2)
                                nlls = F.cross_entropy(logits[..., :-1],
                                                    sample_chunk[..., 1:], 
                                                    reduction='none')
                                first_eos = (sample_chunk == self.eval_model_tokenizer\
                                            .eos_token_id).cumsum(-1) == 1
                                token_mask = (sample_chunk != self.eval_model_tokenizer.eos_token_id)
                                local_metric.update(nlls, first_eos[..., 1:] + token_mask[..., 1:])
        
        # 在分布式环境中收集所有进程的结果
        if is_distributed:
            # 收集所有进程的mean_value和weight
            mean_value = local_metric.mean_value if local_metric.mean_value is not None else torch.tensor(0.0, device=self.device)
            weight = local_metric.weight if local_metric.weight is not None else torch.tensor(0.0, device=self.device)
            
            mean_values = torch.tensor([mean_value], device=self.device)
            weights = torch.tensor([weight], device=self.device)
            
            # 使用all_reduce来聚合结果
            torch.distributed.all_reduce(mean_values, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(weights, op=torch.distributed.ReduceOp.SUM)
            
            # 更新全局度量
            self.gen_ppl_metric.mean_value = mean_values[0]
            self.gen_ppl_metric.weight = weights[0]
        else:
            # 非分布式环境，直接使用局部度量
            self.gen_ppl_metric.mean_value = local_metric.mean_value
            self.gen_ppl_metric.weight = local_metric.weight
        
        # 最终同步点
        if is_distributed:
            torch.distributed.barrier()
            
        print(f"Process {rank} completed compute_generative_perplexity")


    # def restore_model_and_sample(self, num_steps, eps=1e-5):
    #     """Generate samples from the model."""
    #     if self.ema:
    #         self.ema.store(itertools.chain(self.backbone.parameters()))
    #         self.ema.copy_to(itertools.chain(self.backbone.parameters()))
    #     self.backbone.eval()
    #     samples = self._sample(num_steps=num_steps, eps=eps)
    #     if self.ema:
    #         self.ema.restore(itertools.chain(self.backbone.parameters()))
    #     self.backbone.train()
    #     return samples


    @torch.no_grad()
    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model, ensuring all components are on CUDA."""
        logger = utils.get_logger("DSL.sampling")
        
        # 强制将模型和所有组件移动到CUDA
        if torch.cuda.is_available():
            device = self.device
            logger.info(f"Restoring model on device: {device}")
        else:
            raise RuntimeError("CUDA is required for sampling with Flash Attention and Triton kernels")
        
        # 将整个模型移动到CUDA
        self.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # 确保所有子模块都在CUDA上
        for module in self.modules():
            if hasattr(module, 'weight'):
                module.to(device)
        
        # 特别检查关键组件
        if hasattr(self, 'backbone'):
            self.backbone = self.backbone.to(device)
            # 检查backbone的所有子模块
            for module in self.backbone.modules():
                if hasattr(module, 'weight'):
                    module.to(device)
        
        # 确保embedding相关组件在正确设备上
        self.embed = self.embed.to(device)
        self.convert = self.convert.to(device)
        self.convert.beta = self.convert.beta.to(device)
        self.convert.embedding = self.convert.embedding.to(device)
        
        # Apply EMA if available
        if self.ema:
            logger.info("Applying EMA parameters for sampling")
            self.ema.store(itertools.chain(self.backbone.parameters()))
            self.ema.copy_to(itertools.chain(self.backbone.parameters()))
            # 确保EMA后参数也在CUDA上
            for p in self.backbone.parameters():
                p.data = p.data.to(device)
        
        # Set to evaluation mode
        self.backbone.eval()
        
        # 验证设备位置
        logger.info(f"Model device verification - "
                f"Model: {next(self.parameters()).device}, "
                f"Backbone: {next(self.backbone.parameters()).device}, "
                f"Embed: {self.embed.weight.device}, "
                f"Convert.beta: {self.convert.beta.device}")
        
        # Sample with proper error handling
        try:
            logger.info(f"Sampling with {num_steps} steps")
            # 确保_sample过程中的所有张量都在CUDA上
            ts = self.uniform(t_max=10., n_steps=100).to(device)
            
            samples = self._sample(num_steps=num_steps, eps=eps)
            logger.info(f"Generated samples with shape: {samples.shape}")
        except Exception as e:
            logger.error(f"Error during sampling: {str(e)}")
            logger.error(f"Last device locations - "
                    f"Model: {next(self.parameters()).device}, "
                    f"Backbone: {next(self.backbone.parameters()).device}, "
                    f"Embed: {self.embed.weight.device}, "
                    f"Convert.beta: {self.convert.beta.device}")
            raise
        
        # Restore original parameters if using EMA
        if self.ema:
            self.ema.restore(itertools.chain(self.backbone.parameters()))
        
        # Return to training mode
        self.backbone.train()
        
        return samples