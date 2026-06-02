# Discrete Stochastic Localization (DSL)

This repository contains an implementation of **Discrete Stochastic Localization (DSL)**.

> *Discrete Stochastic Localization for Non-autoregressive Generation*

DSL is a new framework for **non-autoregressive generation of discrete data**. 
It unifies masked diffusion, continuous Gaussian diffusion, and autoregressive models as particular SNR paths within the same framework. 
DSL achieves improved probability modeling compared to continuous diffusion baselines, while offering flexible and efficient sampling.

This version is a refactor of previous one. I (GV) did this partially for learning purposes,
to play with the ideas in https://github.com/google-research/tuning_playbook, but also to make it easier for me 
to test some ideas:
- easily iterate through "convert" architecture ideas. A new convert, softmaxconvertbias improves a lot for language modeling
- easier hyper-parameter tuning over SNR paths
- Different NLL estimate implementation - use different SNR maxes and choose best bound
- Add NLL ROAR (Random Order AutoRegressive) for estimation and training. Corresponds to ROAR SNR paths in our framework. The derivation was a bit interesting and tricky, must write down somewhere. 

---

Key Features:
- **SNR-path invariance**: A single denoiser works across arbitrary signal-to-noise ratio (SNR) paths, which may include different SNRs per variable.
- **Flexible sampling**: Supports per-sentence and per-token SNR sampling, with adaptive step sizes for efficiency.
- **Bridging paradigms**: Autoregressive models, masked diffusion, and Gaussian diffusion appear as special cases.
- **State-of-the-art results on Text8**: Competitive Bits Per Character (BPC) and Generative Perplexity (Gen PPL).

---

## Installation and Usage
```bash
git clone https://github.com/gregversteeg/DSL2.git
cd DSL2
pip install -r requirements.txt
```


Train and eval example:
```bash
python train.py
python eval.py \
  eval.ckpt=path/to/checkpoint.pt
```

## Checkpoint

We provide a small checkpoint that starts from **Sahoo’s MDM** and is fine-tuned with **DSL**:

👉 [Download checkpoint](https://www.dropbox.com/scl/fo/vjj1abwumjqwvrelczm3y/AI7rgZxmsESHYhVOYvuj7Ig?rlkey=8sp5f8cxjo0b02g7fj1htjzvl&dl=0)

👉 [Download checkpoint - more tuning](https://www.dropbox.com/scl/fi/e5u4vhn08o1zhkzzqz76p/tune3.pt?rlkey=a2kg5bubyai5k1vv9gsbwmpqv&dl=0)
---

## Results
Using a small subset of OWT validation for now.

**Baseline (MDM pretrained, no fine-tuning):**
- Diffusion NLL: **12.2**
- NLL ROAR: **4.6**

**Fine-tuned on OpenWebText (OWT):**
- Bits per token:
  - Diffusion NLL: **6.33**
  - NLL ROAR: **5.18**

We lost a little ROAR performance, but that seems to be because I used a large LR, 1e-3. 
Longer, slower training should help. 

## Repository Structure

```bash
DSL2/
├── configs          # Hydra configs
├── dataloader.py    # From Sahoo et al.
├── dsl
│   ├── backbone       # diffusion transformers
│   ├── convert.py     # Convert from embedding to "backbone" space
│   ├── dsl.py         # Main class, minimal
│   ├── metrics.py     # New NLL bounds here
│   ├── optimizers.py  # boilerplate
│   ├── samplers.py    # For generation w/ trained model
│   ├── snrs.py        # SNR paths
│   └── utils.py       # boilerplate
├── eval.py
├── README.md  # You are here
├── requirements.txt
├── scripts  # hyper-parameter sweeps, best throughput
└── train.py
```
## To-dos
Missing functionality from Yunshu's version. 
- multi-gpu (I reasoned that param sweeps with small one gpu jobs were more important to start)
- EMA (I believe it can give a good boost - will be nice to add)
- Samplers (curvature, adaptive)
- Hardware optimizations?


## Citation
```bibtex
@inproceedings{dsl2026,
  title={Discrete Stochastic Localization for Non-autoregressive Generation},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR) 2026 (submitted)},
  year={2026}
}
```

## License

## Acknowledgements

