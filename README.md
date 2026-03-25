# DSL-LLaDA

Scaling Continuous Noise Training (DSL) to 8B Masked Diffusion Language Models.

Three inference modes on LLaDA-8B with DSL training:
1. **Standard Remasking** — discrete confidence-based unmasking (best for reasoning)
2. **SDE Generation** — continuous Heun diffusion (best for creative writing)
3. **Error Correction** — detect and fix corrupted tokens

## Models

| Model | HuggingFace | Beta | Best For |
|---|---|---|---|
| Beta1 | [liddlefish/DSL-LLaDA-Beta1](https://huggingface.co/liddlefish/DSL-LLaDA-Beta1) | 0.99 | SDE generation (soft converter) |
| Beta2 | [liddlefish/DSL-LLaDA-Beta2](https://huggingface.co/liddlefish/DSL-LLaDA-Beta2) | 1.98 | Reasoning (GSM8K 50.7%) |
| Highpass | [liddlefish/DSL-LLaDA-Highpass](https://huggingface.co/liddlefish/DSL-LLaDA-Highpass) | 1.98 + residual | Best reasoning (GSM8K 52.1%, MATH 20.2%) |

## Quick Start

```bash
pip install torch transformers safetensors
python inference.py --model Beta1 --mode sde --prompt "Write a story about a robot."
python inference.py --model Beta2 --mode standard --prompt "Question: What is 2+3?\nAnswer:"
python inference.py --model Beta2 --mode correct --input "The cat sat on the mat"
```

## Results

### Standard Remasking (gen=256, 64 steps)
| Method | GSM8K (1319) | MATH-500 |
|---|---|---|
| Original LLaDA | 35.6% | 14.4% |
| Beta2 (DSL) | 50.7% | 17.6% |
| Highpass | **52.1%** | **20.2%** |

### Error Correction (100 texts)
| Method | Random Fix@10% | Clean Preserved |
|---|---|---|
| Original | 13.9% | 90.8% |
| Beta2 (DSL) | **64.9%** | **99.3%** |

### SDE Generation
Best config: Beta1 + Heun integrator + norm init + sensitive schedule.
Zero Chinese character artifacts, fluent text generation in 16 steps.

## Interactive Demo

```bash
pip install gradio
python app.py
# Open http://localhost:7860
```

## License

Based on [LLaDA-8B](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and [DSL2](https://arxiv.org/abs/2602.16169).
