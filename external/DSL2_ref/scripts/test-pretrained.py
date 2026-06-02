from omegaconf import OmegaConf, DictConfig
import IPython
import hydra
from hydra import initialize, compose

import math
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer

from dsl.dsl import build_dsl_from_cfg
from dataloader import get_tokenizer, get_dataloaders
from dsl.utils import prepare_batch, load_weights_from_pretrained
from dsl.metrics import make_mask_tensor
from dsl.snrs import build_snr_path



@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ---- Config and Setup ----

    # You can still override here if needed
    cfg.train.init_pretrained = True
    cfg.train.mix_lambda = 0.5
    cfg.backbone.name = "dit-sahoo"

    device = "cuda"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Get tokenizer and batch of data, generate a mask
    tokenizer = get_tokenizer(cfg)
    V = len(tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    train_loader, _ = get_dataloaders(cfg, tokenizer)
    batch = next(iter(train_loader))
    input_ids, valid_tokens = prepare_batch(batch, pad_id, device)
    B, T = input_ids.shape
    unmask_sizes = torch.randint(0, T, (B,), device=device)
    mask = make_mask_tensor(unmask_sizes, T)  # True = masked


    # ---- Our Model with pretrained weights loaded from Sahoo checkpoint
    model_ours = build_dsl_from_cfg(cfg, vocab_size=V)
    model_ours = load_weights_from_pretrained(model_ours, cfg.train.pretrained_model)
    model_ours = model_ours.to(device)
    model_ours.eval()

    # Make z representing mask/unmask via zero / large SNR
    snr_max = 100 * math.log(V) / math.log(35)
    snrs = snr_max * (~mask)
    z = model_ours.noisy_embedding(input_ids, snrs)

    # For our model, what does convert give?
    with torch.no_grad():
        p_ours = model_ours.convert.get_token_probs(z)
        h_ours = model_ours.convert(z)
        print(f'Minimum probability of most probable token: {p_ours.max(dim=-1).values.min():.3f}')
        logits = model_ours.forward(z)
        one_hot = F.one_hot(p_ours.argmax(dim=-1), num_classes=p_ours.shape[-1]).to(p_ours.dtype)
        h_ours_2 = model_ours.convert.backbone_embedding(one_hot)  # embedding w/ exact one hots

    # ---- HF MDLM Model ----
    hf_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    mdlm_model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/mdlm-owt', trust_remote_code=True).to(device)
    mdlm_model.eval()

    # Convert tokens to GPT2 vocab if needed
    if hf_tokenizer.vocab_size != V:
        print("WARNING: tokenizer vocab mismatch!")

    input_ids_hf = input_ids.clone()
    mask_id = len(hf_tokenizer)  # not defined in gpt2 tokenizer. Sahoo uses new token after as mask
    input_ids_hf[mask] = mask_id

    with torch.no_grad():
        timesteps = torch.zeros(B, device='cuda')
        h_mdlm = mdlm_model.backbone.vocab_embed(input_ids_hf)  # the hidden inputs
        logits_hf = mdlm_model(input_ids=input_ids_hf, timesteps=timesteps)


    # ---- Compare ----
    assert (mdlm_model.backbone.vocab_embed.embedding.T == model_ours.convert.backbone_embedding.weight).all()  # match the weights
    input_match = p_ours.argmax(dim=-1) == input_ids_hf
    print(f'All inputs match? {input_match.all()}')
    print(f'inputs to backbone match?')
    print(f'w/ exact one hots? {F.mse_loss(h_ours_2, h_mdlm):.3f}')
    print(f'w/ approx one hots? {F.mse_loss(h_ours, h_mdlm):.3f}')

    prediction_match = logits.argmax(dim=-1) == logits_hf.argmax(dim=-1)
    print(f'Predictions match? {prediction_match.all()}, Fraction: {prediction_match.float().mean():.3f}')

    IPython.embed()
    print("Test no errors with intermediate SNRs (as opposed to mask/unmask or high/zero SNR)")
    # ----- Assuming that the predictions are the same or similar
    # Test what happens when we apply to *non-masking* SNR noise. Do we get NAN?
    # Ok, we don't get NAN in logits, of course.
    # The issue must be that the logits are putting zero probability on the correct answer, sometimes,
    # in this intermediate SNR regimes, leading to NAN for the diffusion loss.
    #

    snr_path = build_snr_path(cfg.snrpath, device=device)
    for i in range(10):
        batch = next(iter(train_loader))
        input_ids, valid_tokens = prepare_batch(batch, pad_id, device)
        snrs = snr_path.sample(len(input_ids), input_ids.shape[1])
        z = model_ours.noisy_embedding(input_ids, snrs)
        with torch.no_grad():
            p_ours = model_ours.convert.get_token_probs(z)
            h_ours = model_ours.convert(z)
            print(f'Minimum probability of most probable token: {p_ours.max(dim=-1).values.min():.3f}')
            logits = model_ours.forward(z)
            ces = F.cross_entropy(logits.transpose(1, 2).float(), input_ids, reduction="none")

            for name, p in [('logits', logits), ('ces', ces)]:
                has_nan = torch.isnan(p).any().item()
                has_inf = torch.isinf(p).any().item()
                max_abs = p.abs().max().item()

                print(f"[!] Issues in {name} for round {i}?: \n"
                      f"{'NaN' if has_nan else ''}"
                      f"{'Inf' if has_inf else ''}")


    # CHECK for nans in gradients
    model_ours.train()
    z = model_ours.noisy_embedding(input_ids, snrs)
    logits = model_ours.forward(z)
    ces = F.cross_entropy(logits.transpose(1, 2).float(), input_ids, reduction="none")
    loss = (ces * valid_tokens).sum()
    with torch.autograd.detect_anomaly():
        loss.backward()


    IPython.embed()

if __name__ == "__main__":
    main()


# The convertor seems to output something close to the embedded input for mdlm backbone
# But logits are wildly different
# See if backbones (ours and mdlm) give similar predictions on an h.
# difficult... mdlm backbone takes input ids. Our takes the embedded values.

# Ok it seems like inputs match, but the embedded inputs do not match.

#diff = (logits_hf - logits_ours).abs().mean()
#print(f"Mean absolute difference between logits: {diff.item():.4f}")

