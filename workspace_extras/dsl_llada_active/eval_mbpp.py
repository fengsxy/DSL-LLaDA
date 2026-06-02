#!/usr/bin/env python3
"""MBPP evaluation for DSL-LLaDA using lm-evaluation-harness.

Wraps LLaDA's eval_llada.py LLaDAEvalHarness class to support
loading DSL checkpoints and testing at different step counts.

Usage:
    # Original LLaDA baseline (256 steps)
    cd /home/ubuntu/efs/RMDM
    HF_ALLOW_CODE_EVAL=1 python dsl_llada/eval_mbpp.py \
        --model llada_dsl --model_args model_path=GSAI-ML/LLaDA-8B-Instruct,steps=256,gen_length=256,block_length=256

    # DSL checkpoint at 32 steps
    HF_ALLOW_CODE_EVAL=1 python dsl_llada/eval_mbpp.py \
        --model llada_dsl --model_args model_path=checkpoints/pertoken_b2_d100_1k/checkpoint-1000,steps=32,gen_length=256,block_length=256

    # Multi-step sweep (run separately for each step count)
    for s in 32 64 128 256; do
        HF_ALLOW_CODE_EVAL=1 python dsl_llada/eval_mbpp.py \
            --model llada_dsl \
            --model_args model_path=checkpoints/pertoken_b2_d100_1k/checkpoint-1000,steps=$s,gen_length=256,block_length=256 \
            --output_path eval_results/mbpp_sm_b2_s${s}.json
    done
"""
import os
import sys
import json
import random
import torch
import numpy as np

# Add LLaDA to path for generate function
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_root, "LLaDA"))

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from generate import generate

MASK_ID = 126336


def set_seed(seed=1234):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dsl")
class LLaDADSLEvalHarness(LM):
    """LLaDA eval harness with optional DSL checkpoint loading."""

    def __init__(
        self,
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        mask_id=126336,
        max_length=4096,
        batch_size=1,
        mc_num=128,
        is_check_greedy=False,
        cfg=0.0,
        steps=256,
        gen_length=256,
        block_length=256,
        remasking="low_confidence",
        confidence_eos_eot_inf=True,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self._rank = 0
        self._world_size = 1

        # Resolve local checkpoint path
        if not model_path.startswith("/") and "/" in model_path:
            abs_path = os.path.join(_root, model_path)
            if os.path.isdir(abs_path):
                model_path = abs_path

        # Parse lora_path from model_args if provided
        lora_path = kwargs.get("lora_path", "")
        if lora_path and not lora_path.startswith("/") and "/" in lora_path:
            abs_lora = os.path.join(_root, lora_path)
            if os.path.isdir(abs_lora):
                lora_path = abs_lora

        # Load model — local checkpoints are based on LLaDA, use its tokenizer
        is_local = os.path.isdir(model_path) and not model_path.startswith("GSAI")
        tokenizer_path = "GSAI-ML/LLaDA-8B-Instruct" if is_local else model_path

        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )

        # Load LoRA adapter if specified
        if lora_path and os.path.isdir(lora_path):
            from peft import PeftModel
            print(f"  Loading LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()
            print(f"  LoRA merged")

        self.model.eval()
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = float(cfg)
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.remasking = remasking
        # Parse string "True"/"False" from model_args
        if isinstance(confidence_eos_eot_inf, str):
            confidence_eos_eot_inf = confidence_eos_eot_inf.lower() == "true"
        self.confidence_eos_eot_inf = confidence_eos_eot_inf

        print(f"  Model: {model_path}")
        print(f"  Steps: {self.steps}, gen_length: {self.gen_length}, "
              f"block_length: {self.block_length}")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def loglikelihood(self, requests):
        # Not needed for MBPP (generation task)
        raise NotImplementedError("Use generate_until for MBPP")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [
            {"question": req.args[0], "until": req.args[1]["until"]}
            for req in requests
        ]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc=f"MBPP (steps={self.steps})"):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            with torch.no_grad():
                generated = generate(
                    self.model,
                    prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    confidence_eos_eot_inf=self.confidence_eos_eot_inf,
                )

            text = self.tokenizer.decode(
                generated[0][prompt.shape[1]:], skip_special_tokens=False
            )
            for stop_seq in stop_tokens:
                if stop_seq in text:
                    text = text.split(stop_seq)[0]

            # Clean special tokens
            text_ids = self.tokenizer(text)["input_ids"]
            text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
            out.append(text)

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
