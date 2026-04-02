"""SDE Generation Demo — Heun integrator with norm init + sensitive schedule.
Watch tokens crystallize in real-time through continuous diffusion."""
import gradio as gr
import torch
import torch.nn.functional as F
import sys, os, math
import threading
import queue
import time
from functools import lru_cache
sys.path.insert(0, os.path.dirname(__file__))
os.environ['DSL_RESIDUAL'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
SDE_TOP_K = 512
DEFAULT_DEVICE = 'cuda:0'
MODEL_ROOT = os.environ.get('DSL_LLADA_MODEL_ROOT', '/data2/ylong030/models/dsl-llada')
TOKENIZER_PATH = os.path.join(MODEL_ROOT, 'LLaDA-8B-Instruct-tokenizer')
ORIGINAL_MODEL_ID = os.environ.get('DSL_LLADA_ORIGINAL_MODEL', os.path.join(MODEL_ROOT, 'DSL-LLaDA-Beta1'))
LOAD_HIGHPASS = os.environ.get('DSL_LLADA_LOAD_HIGHPASS', '0') == '1'
MODEL_PATHS = {
    'Beta1': os.path.join(MODEL_ROOT, 'DSL-LLaDA-Beta1'),
    'Beta2': os.path.join(MODEL_ROOT, 'DSL-LLaDA-Beta2'),
    'Highpass': os.path.join(MODEL_ROOT, 'DSL-LLaDA-Highpass'),
}
PRESET_CASES = {
    "Custom Prompt": "Write a short story about a robot discovering music.",
    "1. Arithmetic": "Question: What is 27 + 58? Show the reasoning briefly, then give the final answer.",
    "2. Word Math": "A store had 45 apples. It sold 18, then got 27 more. How many apples does it have now?",
    "3. GSM8K Style": "Lena reads 12 pages on Monday, 15 on Tuesday, and twice as many on Wednesday as Monday. How many pages did she read in total?",
    "4. Exact Digits": "Repeat this exactly at the end of your answer: 2027-04-19. First explain why exact copying can be hard.",
    "5. Logic": "There are three boxes: red, blue, green. Exactly one contains a coin. Red is empty. If blue has the coin, green is empty. Where can the coin be?",
    "6. Trip Plan": "Plan a 3-day Tokyo trip for a first-time visitor. Day 1, Day 2, Day 3. Keep each day to 3 items.",
    "7. Creative Story": "Write a short story about a robot hearing rain for the first time. Make it vivid but not cheesy.",
    "8. Style Control": "Write a paragraph in the style of a calm scientific field note describing a lighthouse at dusk.",
    "9. Correction": "The boye walkd to the markit and bought 3 appls.",
    "10. Format Constraint": "Output exactly 5 bullet points. Each bullet must contain exactly 6 words.",
}


def resolve_tokenizer_path():
    return TOKENIZER_PATH if os.path.isdir(TOKENIZER_PATH) else 'GSAI-ML/LLaDA-8B-Instruct'


def resolve_model_path(name, fallback=None):
    local_path = MODEL_PATHS[name]
    return local_path if os.path.isdir(local_path) else fallback


tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_path(), trust_remote_code=True)

MODELS = {}
STANDARD_MODEL = None

def get_model_devices():
    if not torch.cuda.is_available():
        return [DEFAULT_DEVICE]
    requested = os.environ.get('DSL_LLADA_APP_DEVICES')
    if requested:
        return [item.strip() for item in requested.split(',') if item.strip()]
    return [f'cuda:{idx}' for idx in range(max(1, min(torch.cuda.device_count(), 3)))]


def load_model(name, ckpt, model_device, use_residual=False):
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16).to(model_device).eval()
    ne, lb, bw, bb, bv, rw, rb = None, None, None, None, 2.5, None, None
    for fname in sorted(os.listdir(ckpt)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt, fname), device=model_device)
            if 'noise_embed.weight' in st:
                ne = st['noise_embed.weight'].float()
                lb = st['converter.logit_bias'].float()
                bw = st['converter.backbone_embedding.weight'].float()
                bb = st['converter.backbone_embedding.bias'].float()
                bv = st['converter.beta'].item()
                rw = st.get('converter.residual_proj.weight')
                rb = st.get('converter.residual_proj.bias')
                if rw is not None: rw = rw.float()
                if rb is not None: rb = rb.float()
                break
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=model_device)], dim=0) if ne is not None else None
    MODELS[name] = dict(model=model, ne=ne, lb=lb, bw=bw, bb=bb, bv=bv, rw=rw, rb=rb, K=K, use_res=use_residual, device=model_device)


def load_standard_model(model_device):
    global STANDARD_MODEL
    if STANDARD_MODEL is not None:
        return STANDARD_MODEL
    model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(model_device).eval()
    STANDARD_MODEL = dict(model=model, device=model_device)
    return STANDARD_MODEL


@lru_cache(maxsize=None)
def _is_digit_token(token_id):
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()
    return text.isdigit()

print("Loading models...", flush=True)
model_devices = get_model_devices()
print(f"  Devices: {model_devices}", flush=True)
print("  Loading Beta1...", flush=True)
load_model('Beta1', resolve_model_path('Beta1', 'checkpoints/beta1_d100_1k/checkpoint-1000'), model_devices[0], use_residual=False)
print("  Beta1 loaded.", flush=True)
print("  Loading Beta2...", flush=True)
load_model('Beta2', resolve_model_path('Beta2', 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000'), model_devices[min(1, len(model_devices) - 1)], use_residual=False)
print("  Beta2 loaded.", flush=True)
if LOAD_HIGHPASS:
    print("  Loading Highpass...", flush=True)
    load_model('Highpass', resolve_model_path('Highpass', 'checkpoints/pertoken_b2_highpass_1k/checkpoint-1000'), model_devices[min(2, len(model_devices) - 1)], use_residual=True)
    print("  Highpass loaded.", flush=True)
# Reuse Beta1's backbone for standard remasking (same LLaDA weights, avoids extra .to() call)
beta1 = MODELS['Beta1']
STANDARD_MODEL = dict(model=beta1['model'], device=beta1['device'])
print("  Standard model: reusing Beta1 backbone.", flush=True)
print("Models loaded!", flush=True)


def sde_heun_stream(prompt, model_name, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive, beta_infer=None):
    """Heun SDE integrator with streaming."""
    m = MODELS[model_name]
    model, ne, lb, bw, bb, bv, rw, rb, K = m['model'], m['ne'], m['lb'], m['bw'], m['bb'], m['bv'], m['rw'], m['rb'], m['K']
    if beta_infer is not None:
        bv = float(beta_infer)
    use_res = m['use_res']
    model_device = m['device']

    if ne is None:
        yield "No DSL weights"
        return

    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model_device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    GEN = gen_length
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=model_device)

    # SNR schedule
    snr_min = 0.01
    if use_sensitive:
        n1 = max(1, int(steps * 0.05)); n2 = max(1, int(steps * 0.90)); n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
        ]).to(model_device)
    else:
        snrs = torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1)).to(model_device)

    # Init
    torch.manual_seed(int(seed))
    if use_norm_init:
        y = F.normalize(torch.randn(1, GEN, nd, device=model_device), dim=-1)  # ||y||=1
    else:
        y = torch.randn(1, GEN, nd, device=model_device)  # ||y||≈√d≈10

    ns = noise_scale

    def get_xhat(yy, ss):
        z = ss * yy
        cl = bv * (z.float() @ K.T) + lb
        probs = F.softmax(cl.float(), dim=-1)
        h_low = F.linear(probs, bw, bb)
        if use_res and rw is not None:
            embed_w = ne.to(z.dtype)
            z_low = probs[:, :, :embed_w.shape[0]].to(z.dtype) @ embed_w
            z_ln = z_low / (z_low.norm(dim=-1, keepdim=True) + 1e-8)
            ps = (z * z_ln).sum(dim=-1, keepdim=True)
            z_high = z - ps * z_ln
            h_high = F.linear(z_high.to(rw.dtype), rw, rb)
            h = (h_low + h_high).to(torch.bfloat16)
        else:
            h = h_low.to(torch.bfloat16)
        embeds = torch.cat([pe.to(torch.bfloat16) if pe.dim() == 3 else pe.unsqueeze(0).to(torch.bfloat16), h], dim=1)
        with torch.no_grad():
            logits = model(input_ids=dummy, inputs_embeds=embeds).logits[:, pl:, :].float()
        backbone_probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = backbone_probs.topk(min(SDE_TOP_K, backbone_probs.shape[-1]), dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        xh = (top_vals.unsqueeze(-1) * ne[top_idx.clamp_max(ne.shape[0] - 1)]).sum(dim=-2)
        conv_conf = probs.max(dim=-1).values.mean().item()
        return xh, backbone_probs, conv_conf

    actual_steps = len(snrs) - 1
    prev_top1 = None

    for i in range(actual_steps):
        s = snrs[i]
        s_next = snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        xh, backbone_probs, conv_conf = get_xhat(y, s)
        top1 = backbone_probs[0].argmax(dim=-1)
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW
        xh_e, _, _ = get_xhat(y_euler, s_next)
        f_e = (xh_e - y_euler) / s_next
        g_e = ns / s_next
        y = y + 0.5 * (f + f_e) * ds + 0.5 * (g + g_e) * dW

        # Stream
        text = tokenizer.decode(top1, skip_special_tokens=True)
        conf = backbone_probs[0].max(dim=-1).values.mean().item()
        n_changed = (top1 != prev_top1).sum().item() if prev_top1 is not None else GEN
        y_norm = y.norm(dim=-1).mean().item()

        progress = f"**Step {i+1}/{actual_steps}** | SNR: {float(s):.2f} | Conf: {conf:.3f} | ConvConf: {conv_conf:.3f} | ||y||: {y_norm:.2f} | Changed: {n_changed}\n\n"
        yield progress + text
        prev_top1 = top1.clone()

    # Final decode
    z_final = snrs[-1] * y
    cl_final = bv * (z_final.float() @ K.T) + lb
    # Use backbone for final decode (more accurate than converter argmax)
    final_probs = F.softmax(cl_final.float(), dim=-1)
    h_final = F.linear(final_probs, bw, bb)
    if use_res and rw is not None:
        embed_w = ne.to(z_final.dtype)
        z_low_f = final_probs[:, :, :embed_w.shape[0]].to(z_final.dtype) @ embed_w
        z_ln_f = z_low_f / (z_low_f.norm(dim=-1, keepdim=True) + 1e-8)
        ps_f = (z_final * z_ln_f).sum(dim=-1, keepdim=True)
        z_high_f = z_final - ps_f * z_ln_f
        h_high_f = F.linear(z_high_f.to(rw.dtype), rw, rb)
        h_final = (h_final + h_high_f).to(torch.bfloat16)
    else:
        h_final = h_final.to(torch.bfloat16)
    embeds_final = torch.cat([pe.to(torch.bfloat16) if pe.dim() == 3 else pe.unsqueeze(0).to(torch.bfloat16), h_final], dim=1)
    with torch.no_grad():
        lo_final = model(input_ids=dummy, inputs_embeds=embeds_final).logits[:, pl:, :].float()
    final_toks = lo_final.argmax(dim=-1)[0]
    final_text = tokenizer.decode(final_toks, skip_special_tokens=True)
    yield f"**FINAL** | SNR: {float(snrs[-1]):.1f} | ||y||: {y.norm(dim=-1).mean().item():.2f}\n\n" + final_text


def std_remasking_stream(prompt, steps, gen_length, seed, digit_delay, sampling, temperature):
    """Standard discrete remasking using the original LLaDA backbone."""
    m = STANDARD_MODEL
    model = m['model']
    model_device = m['device']

    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model_device)
    pl = iids.shape[1]
    GEN = gen_length

    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=model_device)
    x[0, :pl] = iids[0]
    base = GEN // steps; rem = GEN % steps
    nt = [base] * steps
    for i in range(rem): nt[i] += 1

    torch.manual_seed(int(seed))
    delay_until = max(1, int(steps * 0.75))

    for s in range(steps):
        mi = (x == MASK_ID)
        logits = model(x).logits
        scaled_logits = logits / max(float(temperature), 1e-5)
        p = F.softmax(scaled_logits, dim=-1)
        if sampling:
            x0 = torch.multinomial(p.view(-1, p.shape[-1]), 1).view(p.shape[:-1])
        else:
            x0 = scaled_logits.argmax(dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p[~mi] = -float('inf')
        if digit_delay and s < delay_until:
            digit_mask = torch.zeros_like(x0_p, dtype=torch.bool)
            masked_positions = torch.nonzero(mi, as_tuple=False)
            for batch_idx, pos_idx in masked_positions.tolist():
                if _is_digit_token(int(x0[batch_idx, pos_idx])):
                    digit_mask[batch_idx, pos_idx] = True
            x0_p[digit_mask] *= 0.5
        _, idx = x0_p.sort(dim=-1, descending=True)
        x[0, idx[0, :nt[s]]] = x0[0, idx[0, :nt[s]]]

        n_masked = mi.sum().item()
        text = tokenizer.decode(x[0, pl:pl+GEN], skip_special_tokens=True)
        progress = f"**Step {s+1}/{steps}** | Masked: {n_masked}/{GEN}\n\n"
        yield progress + text

    final_text = tokenizer.decode(x[0, pl:pl+GEN], skip_special_tokens=True)
    yield f"**FINAL** | Masked: 0/{GEN}\n\n" + final_text


def generate_both(prompt, model_name, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive, beta_infer,
                  use_digit_delay, use_std_sampling, std_temperature):
    """Run SDE and Standard in parallel and stream both panels independently."""
    std_steps = steps * 4
    sde_queue = queue.Queue()
    std_queue = queue.Queue()

    def pump(gen, out_queue):
        try:
            for item in gen:
                out_queue.put(("update", item))
        except Exception as exc:
            out_queue.put(("error", f"**ERROR**\n\n{exc}"))
        finally:
            out_queue.put(("done", None))

    sde_thread = threading.Thread(
        target=pump,
        args=(sde_heun_stream(prompt, model_name, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive, beta_infer), sde_queue),
        daemon=True,
    )
    std_thread = threading.Thread(
        target=pump,
        args=(std_remasking_stream(prompt, std_steps, gen_length, seed, use_digit_delay, use_std_sampling, std_temperature), std_queue),
        daemon=True,
    )
    sde_thread.start()
    std_thread.start()

    sde_result = "*SDE loading...*"
    std_result = "*Standard loading original LLaDA...*"
    sde_done = False
    std_done = False
    yield sde_result, std_result

    while not (sde_done and std_done):
        changed = False
        for out_queue, side in ((sde_queue, "sde"), (std_queue, "std")):
            while True:
                try:
                    kind, payload = out_queue.get_nowait()
                except queue.Empty:
                    break
                changed = True
                if kind in ("update", "error"):
                    if side == "sde":
                        sde_result = payload
                    else:
                        std_result = payload
                elif kind == "done":
                    if side == "sde":
                        sde_done = True
                    else:
                        std_done = True
        if changed:
            yield sde_result, std_result
        else:
            time.sleep(0.05)


with gr.Blocks(title="DSL-LLaDA SDE Demo") as demo:
    gr.Markdown("# DSL-LLaDA: Heun SDE vs Standard Remasking")
    gr.Markdown("Left: DSL continuous SDE diffusion (Heun). Right: original LLaDA discrete remasking at 4x the displayed step count.")

    with gr.Row():
        with gr.Column(scale=1):
            preset_case = gr.Dropdown(
                choices=list(PRESET_CASES.keys()),
                value="6. Trip Plan",
                label="Preset Cases"
            )
            prompt = gr.Textbox(label="Prompt", value=PRESET_CASES["6. Trip Plan"],
                                lines=3)
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Beta1",
                label="Model (SDE)"
            )
            with gr.Row():
                steps = gr.Slider(4, 64, value=32, step=4, label="Steps")
                gen_length = gr.Slider(64, 512, value=256, step=64, label="Gen Length")
            with gr.Row():
                noise_scale = gr.Slider(0.0, 1.0, value=0.05, step=0.05, label="Noise Scale")
                snr_max = gr.Slider(10, 200, value=80, step=10, label="SNR Max")
            with gr.Row():
                beta_infer = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="β_infer (SDE)")
            with gr.Row():
                seed = gr.Number(value=42, label="Seed")
            with gr.Row():
                use_norm_init = gr.Checkbox(value=True, label="Norm Init (||y||=1)")
                use_sensitive = gr.Checkbox(value=True, label="Sensitive Schedule")
            with gr.Row():
                use_digit_delay = gr.Checkbox(value=False, label="Digit Delay Trick")
                use_std_sampling = gr.Checkbox(value=False, label="Standard Sampling")
            with gr.Row():
                std_temperature = gr.Slider(0.1, 2.0, value=0.1, step=0.1, label="Standard Temperature")
            generate_btn = gr.Button("Generate Both", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### SDE (Heun, continuous)")
            sde_output = gr.Markdown(value="*Click Generate to start...*")
        with gr.Column(scale=1):
            gr.Markdown("### Standard Remasking (original LLaDA, 4x steps)")
            std_output = gr.Markdown(value="*Waiting...*")

    generate_btn.click(
        fn=generate_both,
        inputs=[prompt, model_choice, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive, beta_infer, use_digit_delay, use_std_sampling, std_temperature],
        outputs=[sde_output, std_output],
    )
    preset_case.change(
        fn=lambda name: PRESET_CASES[name],
        inputs=[preset_case],
        outputs=[prompt],
    )

    gr.Markdown("""
    ### Settings
    - **Heun integrator**: 2nd-order predictor-corrector with two backbone passes per step
    - **Step matching**: Standard remasking runs at `4 x` the displayed step count, so `SDE 16` is compared against `Standard 64`
    - **Norm Init**: Start y on unit sphere (skip norm convergence phase)
    - **Sensitive Schedule**: 90% steps in SNR 7-74 (converter's discriminative zone)
    - **β_infer**: 3.0 = default. Softmax sharpness for converter during inference
    - **Noise Scale**: 0.05 = default (subtle stochasticity), 0.0 = ODE (deterministic)
    - **SNR Max**: 80 = default. Higher = more certain final decode
    - **Digit Delay Trick**: for Standard only, halves digit-token confidence during the first 75% of steps
    - **Standard Temperature**: affects the right panel only; it matters when `Standard Sampling` is enabled

    ### Models
    - **Beta1**: DSL-LLaDA β=1 (soft converter, best for SDE)
    - **Beta2**: DSL-LLaDA β=2 (sharper converter, better for reasoning)
    - **Highpass**: disabled by default; enable with `DSL_LLADA_LOAD_HIGHPASS=1`
    - **Standard panel**: original `GSAI-ML/LLaDA-8B-Instruct`
    """)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
