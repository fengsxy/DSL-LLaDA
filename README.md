# DSL-LLaDA EMNLP Reproducibility Package

This directory contains the code and sampled data needed to reproduce the
DSL-LLaDA experiments described in the EMNLP submission.

## Demo

Try the live demo:

<http://138.23.28.165:7860>

## Layout

- `dsl_llada/`: release-style DSL-LLaDA implementation used for training and
  evaluation.
- `eval_data/`: sampled evaluation data used by the paper scripts.
- `figures/`: plotting helpers.
- `index.html`: paper homepage for GitHub Pages.
- `tests/`: lightweight checks for the public release layout.

Generated directories such as `checkpoints/`, `eval_results/`, `logs/`, and
`wandb/` are intentionally not included. Paper-only assets and internal
workspace snapshots are kept out of this repository.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For full reproduction, install optional evaluation packages as needed:

```bash
pip install bert-score mauve-text deepspeed
```

## Model Checkpoints

The code does not include model weights. The main DSL-LLaDA checkpoint is
available on Hugging Face:

<https://huggingface.co/liddlefish/DSL-LLaDA-Beta1>

To download it locally, use:

```bash
mkdir -p checkpoints
huggingface-cli download liddlefish/DSL-LLaDA-Beta1 \
  --local-dir checkpoints/DSL-LLaDA-Beta1 \
  --local-dir-use-symlinks False
```

Then use the portable registry template:

```bash
export DSL_LLADA_REGISTRY=dsl_llada/configs/registry_hf_beta1.template.json
```

The baseline LLaDA model is loaded from `GSAI-ML/LLaDA-8B-Instruct` unless the
registry is edited to point to a local copy.

## Quick Smoke Tests

Syntax and lightweight checks:

```bash
python -m py_compile $(find dsl_llada -name '*.py' | sort)
for f in dsl_llada/scripts/*.sh; do bash -n "$f"; done
python dsl_llada/eval/sanity_check.py
```

SDE generation smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python dsl_llada/eval/eval_sde_gen_formal.py \
  --model_key hf_beta1 \
  --method sde \
  --nfe 8 \
  --gen_length 64 \
  --prompts eval_data/sde_prompts_200.json \
  --max_prompts 4 \
  --gpu 0 \
  --tag smoke
```

XSum summarization smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python dsl_llada/eval/eval_summarization.py \
  --dataset xsum \
  --data_file eval_data/xsum_1000.json \
  --method sde \
  --model_key hf_beta1 \
  --nfe 8 \
  --gen_length 128 \
  --limit 4 \
  --gpu 0 \
  --out_tag smoke
```
