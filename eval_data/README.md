# eval_data/ -- Static Evaluation Datasets

Fixed, deterministic datasets for reproducible DSL-LLaDA evaluation.
All random sampling uses `seed=42`.

## Files

### texts_100.json
- **Source**: First 100 texts from `results/paper/eval_texts.json` (WikiText-103)
- **Tokenizer**: `GSAI-ML/LLaDA-8B-Instruct` (trust_remote_code=True)
- **Format**: `[{"id", "text", "tokens", "n_tokens"}, ...]`
- **Use**: Corruption probes, calibration, perplexity evaluation

### sde_prompts_100.json
- **Source**: 5 templates x 20 topics (deterministic, no sampling)
- **Format**: `[{"id", "template", "topic", "prompt"}, ...]`
- **Use**: SDE/ODE generation quality evaluation

### gsm8k_100.json
- **Source**: `openai/gsm8k`, config=`main`, split=`test` (1319 total)
- **Sampling**: `random.seed(42); random.sample(range(1319), 100)`
- **Gold extraction**: regex on `#### X` in answer field
- **Format**: `[{"id", "original_idx", "question", "gold_answer"}, ...]`
- **Use**: Reasoning benchmark (primary)

### math_100.json
- **Source**: `EleutherAI/hendrycks_math`, all 7 configs merged, split=`test` (5000 total)
- **Configs**: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus
- **Sampling**: `random.seed(42); random.sample(range(5000), 100)`
- **Gold extraction**: last `\boxed{...}` in solution (nested-brace aware)
- **Format**: `[{"id", "original_idx", "question", "gold_answer"}, ...]`
- **Use**: Reasoning benchmark (secondary)

### semantic_corruption_manual.json
- **Source**: Hand-crafted (100 cases)
- **Use**: Semantic corruption evaluation

## Regeneration

```bash
source /home/ubuntu/efs/RMDM/.venv/bin/activate
python dsl_llada/prepare_eval_data.py
```

Requires: `transformers`, `datasets` (both in `.venv`).
