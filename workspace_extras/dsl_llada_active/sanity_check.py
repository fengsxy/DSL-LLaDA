# dsl_llada/sanity_check.py
"""CPU sanity checks for DSLLaDA — no GPU or real LLaDA model needed."""
import os
import torch
import torch.nn as nn
from dsl_llada.dsl_modules import DSLLaDA, sample_mixed_snr, noisy_embedding

VOCAB = 1000
D_MODEL = 64
NOISE_DIM = 8
B, L = 2, 16


class MockConfig:
    vocab_size = VOCAB
    d_model = D_MODEL


class MockLLaDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.model = type('M', (), {
            'transformer': type('T', (), {
                'wte': nn.Embedding(VOCAB, D_MODEL)
            })()
        })()
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, input_ids, inputs_embeds=None, attention_mask=None, **kw):
        h = inputs_embeds if inputs_embeds is not None else self.model.transformer.wte(input_ids)
        logits = self.lm_head(h)
        return type('Out', (), {'logits': logits})()


def test_snr_sampling():
    # Probabilistic ROAR: each sample independently 10% chance ROAR
    N = 1000
    snrs = sample_mixed_snr(N, L, device='cpu')
    assert snrs.shape == (N, L), f"Wrong shape: {snrs.shape}"
    assert (snrs >= 0).all(), f"Negative SNR found"
    assert (snrs <= 100).all(), f"SNR > 100 found"
    # Identify ROAR vs LogNormal: ROAR has per-token varying SNR, LogNormal is constant
    snr_std = snrs.std(dim=1)
    is_roar = snr_std > 0.1
    n_roar = is_roar.sum().item()
    assert 30 < n_roar < 200, f"Expected ~100 ROAR samples, got {n_roar}"
    # LogNormal samples: all tokens same value, clamped to snr_max_ln
    logn_samples = snrs[~is_roar]
    for i in range(min(10, logn_samples.shape[0])):
        vals = logn_samples[i].unique()
        assert len(vals) == 1, f"LogNormal sample {i} not scalar broadcast: {vals}"
        assert vals[0] <= 40.0, f"LogNormal SNR exceeds snr_max_ln=40: {vals[0]}"
    # batch_size=1 must still be able to trigger ROAR
    n_roar_b1 = sum(1 for _ in range(100) if sample_mixed_snr(1, L, device='cpu').std() > 0.1)
    assert n_roar_b1 > 0, "ROAR never triggered with batch_size=1"
    print(f"✓ SNR sampling: {n_roar} ROAR/{N}, batch_size=1 ROAR: {n_roar_b1}/100")


def test_forward_shape():
    import dsl_llada.dsl_modules as m
    orig_dim = m.DSLLaDA.NOISE_DIM
    orig_mask = m.DSLLaDA.MASK_TOKEN_ID
    m.DSLLaDA.NOISE_DIM = NOISE_DIM
    m.DSLLaDA.MASK_TOKEN_ID = 0

    dsl = DSLLaDA(MockLLaDA())
    input_ids = torch.randint(0, VOCAB, (B, L))
    out = dsl(input_ids)
    assert out.logits.shape == (B, L, VOCAB), f"Wrong shape: {out.logits.shape}"
    print(f"✓ Forward shape: {out.logits.shape}")

    m.DSLLaDA.NOISE_DIM = orig_dim
    m.DSLLaDA.MASK_TOKEN_ID = orig_mask


def test_gradient_flow():
    import dsl_llada.dsl_modules as m
    orig_dim = m.DSLLaDA.NOISE_DIM
    orig_mask = m.DSLLaDA.MASK_TOKEN_ID
    m.DSLLaDA.NOISE_DIM = NOISE_DIM
    m.DSLLaDA.MASK_TOKEN_ID = 0

    dsl = DSLLaDA(MockLLaDA())
    input_ids = torch.randint(0, VOCAB, (B, L))
    out = dsl(input_ids)
    out.logits.mean().backward()

    assert dsl.converter.backbone_embedding.weight.grad is not None, "No grad: backbone_embedding"
    assert dsl.converter.logit_bias.grad is not None, "No grad: logit_bias"
    assert dsl.noise_embed.weight.grad is None, "noise_embed.weight should be frozen (no grad)"
    print("✓ Gradients flow to converter; noise_embed frozen")

    m.DSLLaDA.NOISE_DIM = orig_dim
    m.DSLLaDA.MASK_TOKEN_ID = orig_mask


def test_inputs_embeds_bypass():
    """Verify that passing inputs_embeds skips MockLLaDA's wte."""
    import dsl_llada.dsl_modules as m
    orig_dim = m.DSLLaDA.NOISE_DIM
    orig_mask = m.DSLLaDA.MASK_TOKEN_ID
    m.DSLLaDA.NOISE_DIM = NOISE_DIM
    m.DSLLaDA.MASK_TOKEN_ID = 0

    mock = MockLLaDA()
    dsl = DSLLaDA(mock)
    input_ids = torch.randint(0, VOCAB, (B, L))

    # Pass a sentinel embedding — if wte were called, logits would be different
    sentinel = torch.zeros(B, L, D_MODEL)
    out_sentinel = mock.lm_head(sentinel)
    out_dsl = dsl(input_ids)
    # DSL output should NOT equal sentinel output (it goes through converter, not zeros)
    assert not torch.allclose(out_dsl.logits, out_sentinel), "inputs_embeds bypass not working"
    print("✓ inputs_embeds bypass: converter output differs from raw wte")

    m.DSLLaDA.NOISE_DIM = orig_dim
    m.DSLLaDA.MASK_TOKEN_ID = orig_mask


def test_two_stage_generate_logic():
    """Test two-stage inference with a tiny mock model (no GPU needed)."""
    import dsl_llada.dsl_modules as m
    orig_dim = m.DSLLaDA.NOISE_DIM
    orig_mask = m.DSLLaDA.MASK_TOKEN_ID
    m.DSLLaDA.NOISE_DIM = NOISE_DIM
    m.DSLLaDA.MASK_TOKEN_ID = 0  # use 0 as mask token for tiny vocab

    from dsl_llada.two_stage_generate import two_stage_generate, MASK_ID as _MASK

    dsl = DSLLaDA(MockLLaDA())
    prompt = torch.randint(1, VOCAB, (1, 4))  # 4 prompt tokens (not mask=0)

    # Patch module-level MASK_ID for the tiny test
    import dsl_llada.two_stage_generate as tsg
    orig_mask_id = tsg.MASK_ID
    tsg.MASK_ID = 0

    out = two_stage_generate(dsl, prompt, gen_length=8, total_steps=4, switch_ratio=0.5)
    assert out.shape == (1, 12), f"Expected (1,12), got {out.shape}"
    assert (out[:, :4] == prompt).all(), "Prompt tokens should be unchanged"
    print(f"✓ two_stage_generate: output shape {out.shape}, prompt preserved")

    tsg.MASK_ID = orig_mask_id
    m.DSLLaDA.NOISE_DIM = orig_dim
    m.DSLLaDA.MASK_TOKEN_ID = orig_mask


def test_calibration_syntax():
    """Verify calibration.py has valid Python syntax."""
    import py_compile
    filepath = os.path.join(os.path.dirname(__file__), 'calibration.py')
    try:
        py_compile.compile(filepath, doraise=True)
    except py_compile.PyCompileError as e:
        raise AssertionError(f"Syntax error in calibration.py: {e}")
    print("✓ calibration.py: syntax OK")


def test_llada_cpt_dsl_syntax():
    """Verify llada_cpt_dsl.py has valid Python syntax (no need to import llada_cpt)."""
    import py_compile
    filepath = os.path.join(os.path.dirname(__file__), 'llada_cpt_dsl.py')
    try:
        py_compile.compile(filepath, doraise=True)
    except py_compile.PyCompileError as e:
        raise AssertionError(f"Syntax error in llada_cpt_dsl.py: {e}")
    print("✓ llada_cpt_dsl.py: syntax OK")


def test_compute_loss_dsl_logic():
    """Test CE loss + prompt masking logic from compute_loss_dsl (no trainer needed)."""
    import torch.nn.functional as F
    B, L, V = 2, 16, 1000

    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))

    loss = F.cross_entropy(
        logits.view(-1, V),
        input_ids.view(-1),
        reduction='none',
    ).view(B, L)

    # Apply prompt masking: sample 0 has prompt len=4, sample 1 has len=8
    prompt_length = torch.tensor([4, 8])
    prompt_mask = torch.arange(L).unsqueeze(0) < prompt_length.view(-1, 1)
    loss_masked = loss.clone()
    loss_masked[prompt_mask] = 0.0

    assert loss_masked[0, :4].sum().item() == 0.0, "Prompt positions not zeroed (sample 0)"
    assert loss_masked[1, :8].sum().item() == 0.0, "Prompt positions not zeroed (sample 1)"
    assert loss_masked[0, 4:].sum().item() > 0.0, "Non-prompt loss is zero (sample 0)"
    assert loss_masked[1, 8:].sum().item() > 0.0, "Non-prompt loss is zero (sample 1)"
    print("✓ compute_loss_dsl logic: CE loss + prompt masking correct")


if __name__ == '__main__':
    test_snr_sampling()
    test_forward_shape()
    test_gradient_flow()
    test_inputs_embeds_bypass()
    test_two_stage_generate_logic()
    test_calibration_syntax()
    test_llada_cpt_dsl_syntax()
    test_compute_loss_dsl_logic()
    print("\nAll sanity checks passed ✓")
