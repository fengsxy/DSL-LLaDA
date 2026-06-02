"""Generate interactive HTML trace viewer with step slider.

Features:
- Slider to scrub through steps
- Previous/Next buttons
- Color-coded tokens (new/demoted/changed/kept/mask)
- Side-by-side Original vs v5 comparison
- Hover tooltips showing what changed
"""
import os, sys, torch, json, re
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaDA'))
from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336


def load_model(model_path, lora_path, device):
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if lora_path and os.path.isdir(lora_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model = model.to(device).eval()
    return model, tokenizer


def collect_trace(model, tokenizer, prompt_text, device, steps=64, gen_length=512):
    """Use LLaDA's generate() with save_trajectory=True to get real d3im trace."""
    from generate import generate

    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    P = input_ids.shape[1]

    with torch.no_grad():
        out = generate(
            model, input_ids,
            steps=steps, gen_length=gen_length, block_length=gen_length,
            temperature=0.0, cfg_scale=0.0, remasking='d3im',
            confidence_eos_eot_inf=True,
        )

    # generate() doesn't return trajectory, so we re-run step-by-step
    # but with proper EOS suppression
    L = gen_length
    x = torch.cat([
        input_ids,
        torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    ], dim=1)

    schedule = [int((i + 1) * L / steps) for i in range(steps)]
    schedule[-1] = L

    EOS_ID = 126081
    EOT_ID = 126348
    all_tokens = []

    with torch.no_grad():
        for step in range(steps):
            logits = model(x).logits[0, P:P+L]

            # Suppress EOS/EOT in confidence (like LLaDA's generate)
            logits_for_conf = logits.clone()
            logits_for_conf[:, EOS_ID] = -float('inf')
            logits_for_conf[:, EOT_ID] = -float('inf')

            x0 = logits_for_conf.argmax(dim=-1)
            probs = F.softmax(logits_for_conf.float(), dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            n_keep = schedule[step]
            if step < steps - 1:
                _, topk = conf.topk(min(n_keep, L))
                new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
                new_gen[topk] = x0[topk]
            else:
                # Last step: unmask everything
                new_gen = x0.clone()

            x[0, P:P+L] = new_gen
            all_tokens.append(new_gen.cpu().tolist())

    return all_tokens


def token_to_word(tokenizer, tid):
    if tid == MASK_ID:
        return "█"
    w = tokenizer.decode([tid])
    return w


def generate_interactive_html(traces, tokenizer, output_path):
    """Generate interactive HTML with step slider."""

    # Pre-decode all tokens
    all_data = []
    for trace in traces:
        trace_data = {
            "question": trace["question"][:200],
            "gold": str(trace["gold"]),
            "models": {}
        }
        for model_name, step_tokens in trace["models"].items():
            steps_data = []
            for step_idx, tokens in enumerate(step_tokens):
                words = []
                for tid in tokens:
                    w = token_to_word(tokenizer, tid)
                    words.append({"id": tid, "word": w})
                steps_data.append(words)
            trace_data["models"][model_name] = steps_data
        all_data.append(trace_data)

    # Serialize to JSON for JavaScript
    data_json = json.dumps(all_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>D3IM Interactive Trace Viewer</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
h1 {{ color: #58a6ff; text-align: center; }}
.controls {{ text-align: center; margin: 15px 0; background: #161b22; padding: 15px; border-radius: 8px; position: sticky; top: 0; z-index: 100; }}
.controls button {{ background: #238636; color: white; border: none; padding: 8px 20px; margin: 0 5px; border-radius: 5px; cursor: pointer; font-size: 14px; }}
.controls button:hover {{ background: #2ea043; }}
.controls input[type=range] {{ width: 400px; margin: 0 15px; vertical-align: middle; }}
.step-info {{ font-size: 18px; font-weight: bold; color: #58a6ff; margin: 0 15px; }}
.problem {{ background: #161b22; margin: 20px 0; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }}
.problem-header {{ color: #f0883e; font-size: 14px; margin-bottom: 10px; }}
.model-compare {{ display: flex; gap: 15px; }}
.model-panel {{ flex: 1; background: #0d1117; padding: 10px; border-radius: 5px; border: 1px solid #30363d; }}
.model-title {{ color: #58a6ff; font-weight: bold; margin-bottom: 8px; font-size: 13px; }}
.token-view {{ line-height: 2.0; word-break: break-all; font-size: 13px; min-height: 200px; }}
.tok {{ display: inline; padding: 1px 3px; border-radius: 3px; cursor: default; }}
.tok-mask {{ color: #484f58; }}
.tok-new {{ background: #1b4332; color: #95d5b2; }}
.tok-demoted {{ background: #6a040f; color: #ffb3c1; }}
.tok-changed {{ background: #7b2d26; color: #ffd166; border-bottom: 2px solid #ffd166; }}
.tok-kept {{ color: #adbac7; }}
.stats {{ color: #8b949e; font-size: 11px; margin-top: 8px; }}
.legend {{ text-align: center; margin: 10px 0; }}
.legend .tok {{ margin: 0 8px; font-size: 12px; }}
.answer {{ margin-top: 8px; padding: 5px; background: #161b22; border-radius: 4px; font-size: 12px; }}
.correct {{ color: #3fb950; }}
.wrong {{ color: #f85149; }}
.kbd {{ background: #30363d; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
</style>
</head><body>

<h1>🔍 D3IM Generation Trace Viewer</h1>

<div class="legend">
    <span class="tok tok-new">New (MASK→token)</span>
    <span class="tok tok-demoted">Demoted (token→MASK)</span>
    <span class="tok tok-changed">Changed (old→new)</span>
    <span class="tok tok-kept">Unchanged</span>
    <span class="tok tok-mask">████ MASK</span>
</div>

<div class="controls" id="controls">
    <button onclick="prevStep()">◀ Prev</button>
    <input type="range" id="slider" min="0" max="63" value="0" oninput="setStep(this.value)">
    <button onclick="nextStep()">Next ▶</button>
    <span class="step-info">Step: <span id="step-display">0</span> / 63</span>
    <br><small style="color:#8b949e">Keyboard: <span class="kbd">←</span> <span class="kbd">→</span> or <span class="kbd">A</span> <span class="kbd">D</span></small>
</div>

<div id="content"></div>

<script>
const DATA = {data_json};
const MASK_ID = {MASK_ID};
let currentStep = 0;

function classify(curr, prev) {{
    if (curr.id === MASK_ID) {{
        if (prev.id !== MASK_ID) return 'demoted';
        return 'mask';
    }}
    if (prev.id === MASK_ID) return 'new';
    if (prev.id !== curr.id) return 'changed';
    return 'kept';
}}

function escapeHtml(s) {{
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function renderStep(step) {{
    let html = '';
    DATA.forEach((prob, pi) => {{
        html += '<div class="problem">';
        html += '<div class="problem-header">Q' + pi + ': ' + escapeHtml(prob.question) + '<br>Gold: <b>' + prob.gold + '</b></div>';
        html += '<div class="model-compare">';

        const modelNames = Object.keys(prob.models);
        modelNames.forEach(mname => {{
            const steps = prob.models[mname];
            const curr = steps[step];
            const prev = step > 0 ? steps[step-1] : curr.map(t => ({{id: MASK_ID, word: '█'}}));

            let nNew=0, nDem=0, nChg=0, nMask=0;
            let tokHtml = '';

            for (let i = 0; i < curr.length; i++) {{
                const cls = classify(curr[i], prev[i]);
                const w = escapeHtml(curr[i].word);
                let title = '';

                if (cls === 'mask') {{ nMask++; tokHtml += '<span class="tok tok-mask">█</span>'; }}
                else if (cls === 'new') {{ nNew++; tokHtml += '<span class="tok tok-new" title="NEW">' + w + '</span>'; }}
                else if (cls === 'demoted') {{ nDem++; tokHtml += '<span class="tok tok-demoted" title="was: ' + escapeHtml(prev[i].word) + '">█</span>'; }}
                else if (cls === 'changed') {{ nChg++; tokHtml += '<span class="tok tok-changed" title="' + escapeHtml(prev[i].word) + ' → ' + w + '">' + w + '</span>'; }}
                else {{ tokHtml += '<span class="tok tok-kept">' + w + '</span>'; }}
            }}

            // Check for #### answer
            const fullText = curr.filter(t => t.id !== MASK_ID).map(t => t.word).join('');
            const ansMatch = fullText.match(/####\\s*(-?[\\d,]+\\.?\\d*)/);
            const pred = ansMatch ? ansMatch[1].replace(/,/g,'').trim() : '';
            const isCorrect = pred === prob.gold;

            html += '<div class="model-panel">';
            html += '<div class="model-title">' + escapeHtml(mname) + '</div>';
            html += '<div class="token-view">' + tokHtml + '</div>';
            html += '<div class="stats">+' + nNew + ' new | △' + nChg + ' changed | ▼' + nDem + ' demoted | ' + nMask + ' masks remaining</div>';
            if (pred) {{
                html += '<div class="answer">Answer: <span class="' + (isCorrect ? 'correct' : 'wrong') + '">' + pred + '</span> ' + (isCorrect ? '✅' : '❌') + '</div>';
            }}
            html += '</div>';
        }});

        html += '</div></div>';
    }});

    document.getElementById('content').innerHTML = html;
    document.getElementById('step-display').textContent = step;
    document.getElementById('slider').value = step;
}}

function setStep(s) {{
    currentStep = parseInt(s);
    renderStep(currentStep);
}}

function prevStep() {{
    if (currentStep > 0) setStep(currentStep - 1);
}}

function nextStep() {{
    if (currentStep < 63) setStep(currentStep + 1);
}}

document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowLeft' || e.key === 'a') prevStep();
    if (e.key === 'ArrowRight' || e.key === 'd') nextStep();
}});

// Initial render
renderStep(0);
</script>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Interactive HTML saved to {output_path}")


if __name__ == "__main__":
    device = "cuda:0"

    data = json.load(open("eval_data/gsm8k_100.json"))
    problem_indices = [4, 8, 9]
    problems = [data[i] for i in problem_indices]

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    traces = []
    for item in problems:
        prompt = (f"Solve step by step. Put final answer after ####.\n"
                  f"Question: {item['question']}\nAnswer:")
        trace_entry = {
            "question": item["question"],
            "gold": item["gold_answer"],
            "models": {},
        }

        for name, lora in [("Original", ""), ("D3IM-A v5 r=64", "checkpoints/d3im_lora_v5_r64/checkpoint-500")]:
            print(f"Tracing {name} on Q{item['id']}...")
            model, _ = load_model("GSAI-ML/LLaDA-8B-Instruct", lora, device)
            step_tokens = collect_trace(model, tokenizer, prompt, device, steps=64, gen_length=512)
            trace_entry["models"][name] = step_tokens
            del model
            torch.cuda.empty_cache()

        traces.append(trace_entry)

    generate_interactive_html(traces, tokenizer, "paper_figures/d3im_trace_interactive.html")
