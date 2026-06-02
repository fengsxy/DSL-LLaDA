#!/bin/bash

# 指定可见的 GPU：只暴露卡号 2 和 3
# export CUDA_VISIBLE_DEVICES=2,3
export NCCL_TIMEOUT=4800000 

# 基本设置
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 禁用IB（因为没有安装 libibverbs-dev）
export NCCL_IB_DISABLE=1

## NCCL通信设置
export NCCL_P2P_DISABLE=0    # 启用P2P通信
export NCCL_MAX_NRINGS=4     # 从8降到4
export NCCL_MAX_NCHANNELS=2  # 从4降到2
export NCCL_MIN_NCHANNELS=1
export NCCL_BUFFSIZE=2097152  # 减小缓冲区大小
export NCCL_SOCKET_IFNAME=lo  # 本地回环接口
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 资源管理设置（仍然使用两个GPU）
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 每个GPU的连接数限制为1
export NCCL_ASYNC_ERROR_HANDLING=1    # 启用异步错误处理

# 不再override loader.batch_size 和 loader.eval_batch_size
python -u -m main \
  model=small \
  data=openwebtext-split \
  wandb.name=dsl-owt-small \
  eval.compute_generative_perplexity=True \
  loader.global_batch_size=512 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model.length=1024 \
  optim.lr=3e-4 \
  loader.num_workers=8 \
  model.dim_embed=16 \
  checkpointing.resume_ckpt_path=/home/ywu380/DSL_mdml/outputs/openwebtext-train/2025.04.28/103404/checkpoints/last.ckpt > train_owt.log
  # callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 > train_owt_from_scratch.log