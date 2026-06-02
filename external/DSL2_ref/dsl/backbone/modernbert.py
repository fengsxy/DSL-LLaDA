# dsl/backbone/modernbert.py  (or append to bert.py if you prefer)
import torch
import torch.nn as nn
try:
    from transformers import ModernBertConfig, ModernBertModel
except Exception as e:
    raise ImportError(
        "ModernBERT backbone requires `transformers>=4.48`. "
        "pip install -U transformers"
    ) from e

class ModernBERT(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        bb = cfg.backbone
        d_model = int(bb.dim_h)
        mcfg = ModernBertConfig(
            vocab_size=int(vocab_size),                # unused if we pass inputs_embeds, harmless to set
            hidden_size=d_model,
            num_attention_heads=int(bb.n_heads),
            num_hidden_layers=int(bb.n_blocks),
            intermediate_size=int(getattr(bb, "intermediate_size", 4 * d_model)),
            max_position_embeddings=int(getattr(getattr(cfg, "data", {}), "block_size", 8192)),
            attention_dropout=float(getattr(bb, "dropout", 0.0)),
            hidden_dropout=float(getattr(bb, "dropout", 0.0)),
            local_attention=int(getattr(bb, "local_attention", 128)),
            pad_token_id=0  # we don't use the embedding layer, so irrelevant, but needed to init
        )

        # Force Hugging Face to use FlashAttention 2 kernels
        setattr(mcfg, "attn_implementation", "flash_attention_2")
        setattr(mcfg, "_attn_implementation", "flash_attention_2")  # older transformers use the underscored name
        # Construct the model directly in bf16 so FA2 sees the dtype

        # NOTE: ModernBertModel **constructor** does NOT accept `torch_dtype=`; only `from_pretrained` does.
        # Set dtype on the config, then instantiate, then move/cast the module.
        mcfg.torch_dtype = torch.bfloat16
        self.encoder = ModernBertModel(mcfg)
        self.encoder.to(device="cuda", dtype=torch.bfloat16)
        # Keep the config in sync post-construction as well (some HF code reads model.config later)
        self.encoder.config.attn_implementation = "flash_attention_2"
        setattr(self.encoder.config, "_attn_implementation", "flash_attention_2")

        self.encoder.resize_token_embeddings(1)  # we don't use the embedding layer
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device='cuda', dtype=torch.bfloat16)

    def forward(self, h, _mags):
        # h: (B,T,d_model) from convert; mags unused here
        # Hardcode CUDA + BF16: move/cast inputs, run encoder, project
        h = h.to(device='cuda', dtype=torch.bfloat16)
        outputs = self.encoder(inputs_embeds=h)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits
