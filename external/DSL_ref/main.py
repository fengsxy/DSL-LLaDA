import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
# import diffusion
import wo_EDMv2_tricks_dsl
import utils

# register some new configs
omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


# Modified according to DSL Need
def _load_from_checkpoint(config, tokenizer):
    """
    Load model from checkpoint, correctly setting up for device handling.
    Lightning's Trainer will handle device placement automatically when using test().
    """
    logger = utils.get_logger(__name__)
    
    if 'hf' in config.backbone:
        logger.info('Loading model from HuggingFace...')
        # model = diffusion.DSL(config, tokenizer=tokenizer)
        model = wo_EDMv2_tricks_dsl.DSL(config, tokenizer=tokenizer)
    else:
        logger.info(f'Loading model from checkpoint: {config.eval.checkpoint_path}')
        # model = diffusion.DSL.load_from_checkpoint(
        model = wo_EDMv2_tricks_dsl.DSL.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer,
            config=config,
            strict=True  # Ensure all keys match
        )
    
    # Explicitly ensure critical attributes are set
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        model.tokenizer = tokenizer
    
    # Let's check if we loaded the model correctly
    logger.info(f'Model loaded with backbone type: {config.backbone}')
    logger.info(f'Model vocab size: {model.vocab_size}')
    
    # Lightning's Trainer will handle device placement automatically
    # But we'll explicitly set the model to evaluation mode
    model.eval()
    
    return model


###### Not checking the following two functions in detail ###### 
@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True,
    save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open('{}/config_tree.txt'.format(config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)


# def generate_samples(config, logger, tokenizer):
#     logger.info('Generating samples.')
#     model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    
#     if config.eval.disable_ema:
#         logger.info('Disabling EMA.')
#         model.ema = None
    
#     # Reset metrics
#     model.gen_ppl_metric.reset()
    
#     # Get validation dataloader
#     _, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    
#     # Setup DDP strategy and trainer for testing
#     logger.info('Setting up Lightning Trainer for testing.')
#     ddp = L.pytorch.strategies.DDPStrategy(process_group_backend="gloo")
    
#     # Configure devices
#     devices = config.trainer.devices if hasattr(config.trainer, 'devices') else 1
    
#     trainer = L.Trainer(
#         strategy=ddp,
#         limit_test_batches=10,  # Only test on 10 batches
#         max_epochs=1,  # Not used for testing, but required
#         accelerator="gpu",
#         devices=devices
#     )
    
#     # Run test step with Lightning
#     logger.info('Running test evaluation on validation set.')
#     test_results = trainer.test(model, dataloaders=valid_ds)
#     logger.info(f'Test results: {test_results}')
    
#     # 重要：确保在生成样本之前模型在CUDA上
#     logger.info('Moving model to CUDA for sampling')
#     # 强制将模型移动到CUDA设备
#     model = model.to('cuda')
    
#     # 确保模型的所有子组件也在CUDA上
#     for module in model.modules():
#         if hasattr(module, 'weight') or hasattr(module, 'bias'):
#             module.to('cuda')
    
#     # 特别处理backbone
#     if hasattr(model, 'backbone'):
#         model.backbone = model.backbone.to('cuda')
    
#     # 特别处理embed和convert
#     model.embed = model.embed.to('cuda')
#     model.convert = model.convert.to('cuda')
#     model.convert.beta = model.convert.beta.to('cuda')
#     model.convert.embedding = model.convert.embedding.to('cuda')
    
#     # Now generate samples
#     logger.info('Generating text samples.')
#     text_samples = []
#     for i in range(config.sampling.num_sample_batches):
#         logger.info(f'Generating batch {i+1}/{config.sampling.num_sample_batches}')
#         samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
#         batch_text_samples = model.tokenizer.batch_decode(samples)
#         text_samples.extend(batch_text_samples)
        
#         # Calculate generative perplexity
#         model.compute_generative_perplexity(batch_text_samples)
    
#     # Show generated samples
#     for idx, sample in enumerate(text_samples[:3]):  # Print first 3 samples
#         # logger.info(f'Sample {idx+1}: {sample[:150]}...')  # Show first 150 chars
#         logger.info(f'Sample {idx+1}: {sample}')
    
#     # Log generative perplexity
#     gen_ppl = model.gen_ppl_metric.compute()
#     logger.info(f'Generative perplexity: {gen_ppl.item():.4f}')
    
#     return text_samples

import torch.nn as nn
def generate_samples(config, logger, tokenizer):
    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    
    # Reset metrics
    model.gen_ppl_metric.reset()
    
    # Get validation dataloader
    _, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    
    # Setup DDP strategy and trainer for testing
    logger.info('Setting up Lightning Trainer for testing.')
    ddp = L.pytorch.strategies.DDPStrategy(process_group_backend="gloo")
    
    # Configure devices
    devices = config.trainer.devices if hasattr(config.trainer, 'devices') else 1
    
    trainer = L.Trainer(
        strategy=ddp,
        limit_test_batches=10,  # Only test on 10 batches
        max_epochs=1,  # Not used for testing, but required
        accelerator="gpu",
        devices=devices
    )
    
    # 确定设备
    if torch.cuda.is_available():
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cuda:0')
        logger.info(f'Using device: {device}')
    else:
        logger.warning('CUDA not available. Using CPU instead.')
        device = torch.device('cpu')
    
    # 首先将整个模型移动到设备
    model = model.to(device)
    
    # 直接处理问题组件 - embed
    logger.info("明确处理embed组件...")
    
    # 确保embed及其权重在正确设备上
    model.embed = model.embed.to(device)
    # 特别处理weight_norm版本的embed
    if hasattr(model.embed, 'weight_v'):
        model.embed.weight_v.data = model.embed.weight_v.data.to(device)
    if hasattr(model.embed, 'weight_g'):
        model.embed.weight_g.data = model.embed.weight_g.data.to(device)
    
    # 解除并重新应用weight_norm，确保所有内部张量都在正确设备上
    if hasattr(model.embed, 'weight_v') and hasattr(model.embed, 'weight_g'):
        logger.info("重新应用weight_norm到embed...")
        # 获取原始尺寸
        out_features, in_features = model.embed.weight.shape
        
        # 创建新的Embedding层在正确设备上
        new_embed = nn.Embedding(out_features, in_features, device=device)
        
        # 复制权重
        with torch.no_grad():
            new_embed.weight.copy_(model.embed.weight.to(device))
        
        # 应用weight_norm
        model.embed = torch.nn.utils.weight_norm(new_embed, name='weight', dim=0)
        
        # 确保weight_g为1且不需要梯度
        with torch.no_grad():
            model.embed.weight_g.fill_(1.)
            model.embed.weight_g.requires_grad_(False)
    
    # 确保convert及其组件在正确设备上
    model.convert = model.convert.to(device)
    model.convert.beta.data = model.convert.beta.data.to(device)
    model.convert.embedding.weight.data = model.convert.embedding.weight.data.to(device)
    
    # 更新convert中的embed引用，确保它指向正确设备上的模型.embed
    model.convert.embed = model.embed
    
    # 打印设备验证
    logger.info(f"处理后设备验证 - Model: {next(model.parameters()).device}, "
               f"Embed: {model.embed.weight.device}, "
               f"Convert.beta: {model.convert.beta.device}, "
               f"Convert.embed: {model.convert.embed.weight.device}")
    
    # Now generate samples
    logger.info('Generating text samples.')
    text_samples = []
    for i in range(config.sampling.num_sample_batches):
        logger.info(f'Generating batch {i+1}/{config.sampling.num_sample_batches}')
        
        try:
            with torch.cuda.device(device):
                # 修改restore_model_and_sample方法以确保embed在正确设备上
                def patched_restore_model_and_sample(orig_method):
                    def wrapper(num_steps, eps=1e-5):
                        # 确保embed在正确设备上
                        model.embed = model.embed.to(device)
                        model.convert.embed = model.embed
                        # 调用原始方法
                        return orig_method(num_steps, eps)
                    return wrapper
                
                # 临时替换方法
                original_method = model.restore_model_and_sample
                model.restore_model_and_sample = patched_restore_model_and_sample(original_method)
                
                # 生成样本
                samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
                
                # 恢复原始方法
                model.restore_model_and_sample = original_method
                
                batch_text_samples = model.tokenizer.batch_decode(samples)
                text_samples.extend(batch_text_samples)
            
            # Calculate generative perplexity
            model.compute_generative_perplexity(batch_text_samples)
            
        except Exception as e:
            logger.error(f"生成样本时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 我们已经尝试了最佳修复，如果仍然失败，则退出循环
            break
    
    # Show generated samples
    if text_samples:
        for idx, sample in enumerate(text_samples[:3]):  # Print first 3 samples
            logger.info(f'Sample {idx+1}: {sample}')
    else:
        logger.warning("没有成功生成任何样本")
    
    # Log generative perplexity
    try:
        gen_ppl = model.gen_ppl_metric.compute()
        logger.info(f'Generative perplexity: {gen_ppl.item():.4f}')
    except Exception as e:
        logger.error(f"计算生成困惑度时出错: {str(e)}")
    
    return text_samples


def _ppl_eval(config, logger, tokenizer):
    logger.info('Starting Zero Shot Eval.')
    logger.info(f'Evaluating checkpoint = {config.eval.checkpoint_path}')
    logger.info(F'SNR MAX = {config.training.t_max}')

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
    print(f'CHECKING.....')
    print(torch.cuda.device_count())
    print(f'global_batch_size = {config.loader.global_batch_size}, devices = {config.trainer.devices}, accumulate_grad_batches = {config.trainer.accumulate_grad_batches}')
    # batch_size: ${div_up:${.global_batch_size}, ${eval:${trainer.devices} * ${trainer.num_nodes}}}
    print(f'loader.batch_size = {config.loader.batch_size}, trainer.num_nodes = {config.trainer.num_nodes}')
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    
    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    # model = diffusion.DSL(config, tokenizer)
    model = wo_EDMv2_tricks_dsl.DSL(config, tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer, 
        default_root_dir=os.getcwd(), 
        callbacks=callbacks, 
        strategy=hydra.utils.instantiate(config.strategy), # this needs specify which parallelization to be used
        logger=wandb_logger
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)



@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    print("loader.num_workers =", config.loader.num_workers)

    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == 'sample_eval':
        generate_samples(config, logger, tokenizer)
    elif config.mode == 'ppl_eval':
        _ppl_eval(config, logger, tokenizer)
    if config.mode == 'train':
        _train(config, logger, tokenizer)


if __name__ == '__main__':
    main()