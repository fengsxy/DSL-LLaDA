"""SDE Generation Demo — Heun integrator with norm init + sensitive schedule.
Watch tokens crystallize in real-time through continuous diffusion."""
import gradio as gr
import torch
import torch.nn.functional as F
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
os.environ['DSL_RESIDUAL'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

MASK_ID = 126336
device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

MODELS = {}

def load_model(name, ckpt, use_residual=False):
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    ne, lb, bw, bb, bv, rw, rb = None, None, None, None, 2.5, None, None
    for fname in sorted(os.listdir(ckpt)):
        if fname.endswith('.safetensors'):
            st = load_file(os.path.join(ckpt, fname), device=device)
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
    K = torch.cat([ne, torch.zeros(1, ne.shape[1], device=device)], dim=0) if ne is not None else None
    MODELS[name] = dict(model=model, ne=ne, lb=lb, bw=bw, bb=bb, bv=bv, rw=rw, rb=rb, K=K, use_res=use_residual)

print("Loading models...")
load_model('Beta1', 'checkpoints/beta1_d100_1k/checkpoint-1000', use_residual=False)
load_model('Beta2', 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000', use_residual=False)
load_model('Highpass', 'checkpoints/pertoken_b2_highpass_1k/checkpoint-1000', use_residual=True)
print("Models loaded!")


def sde_heun_stream(prompt, model_name, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive):
    """Heun (2nd order) SDE integrator with streaming. The real one that works."""
    m = MODELS[model_name]
    model, ne, lb, bw, bb, bv, rw, rb, K = m['model'], m['ne'], m['lb'], m['bw'], m['bb'], m['bv'], m['rw'], m['rb'], m['K']
    use_res = m['use_res']

    if ne is None:
        yield "No DSL weights"
        return

    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    pe = model.model.transformer.wte(iids).float()
    nd = ne.shape[1]
    GEN = gen_length
    dummy = torch.zeros(1, pl + GEN, dtype=torch.long, device=device)

    # SNR schedule
    snr_min = 0.01
    if use_sensitive:
        n1 = max(1, int(steps * 0.05)); n2 = max(1, int(steps * 0.90)); n3 = steps - n1 - n2
        snrs = torch.cat([
            torch.exp(torch.linspace(math.log(snr_min), math.log(7), n1 + 1)),
            torch.exp(torch.linspace(math.log(7), math.log(74), n2 + 1))[1:],
            torch.exp(torch.linspace(math.log(74), math.log(snr_max), n3 + 1))[1:],
        ]).to(device)
    else:
        snrs = torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps + 1)).to(device)

    # Init
    torch.manual_seed(int(seed))
    if use_norm_init:
        y = F.normalize(torch.randn(1, GEN, nd, device=device), dim=-1)  # ||y||=1
    else:
        y = torch.randn(1, GEN, nd, device=device)  # ||y||≈√d≈10

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
        # top-p filter
        sp, si = backbone_probs.sort(dim=-1, descending=True)
        cm = sp.cumsum(dim=-1)
        mask = cm - sp > 0.9
        sp[mask] = 0; sp = sp / sp.sum(dim=-1, keepdim=True)
        filt = torch.zeros_like(backbone_probs); filt.scatter_(-1, si, sp)
        xh = filt[:, :, :ne.shape[0]] @ ne
        conv_conf = probs.max(dim=-1).values.mean().item()
        return xh, backbone_probs, conv_conf

    actual_steps = len(snrs) - 1
    prev_top1 = None

    for i in range(actual_steps):
        s = snrs[i]
        s_next = snrs[i + 1]
        ds = s_next - s
        dW = torch.sqrt(ds.abs()) * torch.randn_like(y)

        # Predictor (Euler step)
        xh, backbone_probs, conv_conf = get_xhat(y, s)
        top1 = backbone_probs[0].argmax(dim=-1)
        f = (xh - y) / s
        g = ns / s
        y_euler = y + f * ds + g * dW

        # Corrector (Heun: average of predictor and corrector slopes)
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


def std_remasking_stream(prompt, steps, gen_length, seed):
    """Standard discrete remasking for comparison."""
    # Use Beta1 model for std too
    m = MODELS['Beta1']
    model = m['model']

    msgs = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    iids = tokenizer(formatted, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    pl = iids.shape[1]
    GEN = gen_length

    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=device)
    x[0, :pl] = iids[0]
    base = GEN // steps; rem = GEN % steps
    nt = [base] * steps
    for i in range(rem): nt[i] += 1

    torch.manual_seed(int(seed))

    for s in range(steps):
        mi = (x == MASK_ID)
        logits = model(x).logits
        x0 = logits.argmax(dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p[~mi] = -float('inf')
        _, idx = x0_p.sort(dim=-1, descending=True)
        x[0, idx[0, :nt[s]]] = x0[0, idx[0, :nt[s]]]

        n_masked = mi.sum().item()
        text = tokenizer.decode(x[0, pl:pl+GEN], skip_special_tokens=True)
        progress = f"**Step {s+1}/{steps}** | Masked: {n_masked}/{GEN}\n\n"
        yield progress + text

    final_text = tokenizer.decode(x[0, pl:pl+GEN], skip_special_tokens=True)
    yield f"**FINAL** | Masked: 0/{GEN}\n\n" + final_text


def generate_both(prompt, model_name, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive):
    """Run SDE and Standard in sequence, yield both results."""
    # Run SDE
    sde_result = ""
    for sde_result in sde_heun_stream(prompt, model_name, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive):
        yield sde_result, "*Waiting for SDE to finish...*"

    # Run Standard
    for std_result in std_remasking_stream(prompt, steps, gen_length, seed):
        yield sde_result, std_result


with gr.Blocks(title="DSL-LLaDA SDE Demo") as demo:
    gr.Markdown("# DSL-LLaDA: SDE vs Standard Remasking")
    gr.Markdown("Left: continuous SDE diffusion (Heun). Right: discrete remasking. Same model, same prompt.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value="Write a short story about a robot discovering music.",
                                lines=3)
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Beta1",
                label="Model (SDE)"
            )
            with gr.Row():
                steps = gr.Slider(4, 64, value=16, step=4, label="Steps")
                gen_length = gr.Slider(64, 512, value=128, step=64, label="Gen Length")
            with gr.Row():
                noise_scale = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Noise Scale")
                snr_max = gr.Slider(10, 200, value=50, step=10, label="SNR Max")
            with gr.Row():
                seed = gr.Number(value=42, label="Seed")
            with gr.Row():
                use_norm_init = gr.Checkbox(value=True, label="Norm Init (||y||=1)")
                use_sensitive = gr.Checkbox(value=True, label="Sensitive Schedule")
            generate_btn = gr.Button("Generate Both", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### SDE (Heun, continuous)")
            sde_output = gr.Markdown(value="*Click Generate to start...*")
        with gr.Column(scale=1):
            gr.Markdown("### Standard Remasking (discrete)")
            std_output = gr.Markdown(value="*Waiting...*")

    generate_btn.click(
        fn=generate_both,
        inputs=[prompt, model_choice, steps, gen_length, noise_scale, snr_max, seed, use_norm_init, use_sensitive],
        outputs=[sde_output, std_output],
    )

    gr.Markdown("""
    ### Settings
    - **Heun integrator**: 2nd-order predictor-corrector (2 forward passes per step, more accurate than Euler)
    - **Norm Init**: Start y on unit sphere (skip norm convergence phase)
    - **Sensitive Schedule**: 90% steps in SNR 7-74 (converter's discriminative zone)
    - **Noise Scale**: 0.3 = default (more exploration), 0.0 = ODE (deterministic)
    - **SNR Max**: 50 = default. Higher = more certain final decode

    ### Models
    - **Beta1**: DSL-LLaDA β=1 (soft converter, best for SDE)
    - **Beta2**: DSL-LLaDA β=2 (sharper converter, better for reasoning)
    - **Highpass**: DSL-LLaDA β=2 + high-pass residual
    """)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
