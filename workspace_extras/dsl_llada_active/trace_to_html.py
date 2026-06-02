"""Generate HTML visualization of d3im traces.

Shows token-by-token evolution across steps, with color coding:
- Green: newly committed (MASK → token)
- Red: demoted (token → MASK)
- Yellow: changed (token → different token, correction!)
- Gray: MASK
- White: unchanged committed
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
    """Run d3im and collect per-step token arrays."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    P = input_ids.shape[1]
    L = gen_length

    x = torch.cat([
        input_ids,
        torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    ], dim=1)

    schedule = [int((i + 1) * L / steps) for i in range(steps)]
    schedule[-1] = L

    # Store token ids at each step
    step_tokens = []  # list of (L,) tensors
    step_confs = []

    with torch.no_grad():
        for step in range(steps):
            logits = model(x).logits[0, P:P+L]
            probs = F.softmax(logits.float(), dim=-1)
            x0 = logits.argmax(dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            n_keep = schedule[step]
            _, topk = conf.topk(min(n_keep, L))
            new_gen = torch.full((L,), MASK_ID, dtype=torch.long, device=device)
            new_gen[topk] = x0[topk]
            x[0, P:P+L] = new_gen

            step_tokens.append(new_gen.cpu().clone())
            step_confs.append(conf.cpu().clone())

    return step_tokens, step_confs


def generate_html(traces, tokenizer, output_path):
    """Generate HTML visualization from multiple model traces on multiple problems."""

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>D3IM Trace Visualization</title>
<style>
body { font-family: monospace; font-size: 12px; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { color: #e94560; }
h2 { color: #0f3460; background: #e94560; padding: 8px; margin-top: 30px; }
h3 { color: #16213e; background: #0f3460; color: #eee; padding: 5px; }
.step-row { margin: 2px 0; white-space: pre-wrap; word-break: break-all; line-height: 1.6; }
.step-label { display: inline-block; width: 80px; color: #888; }
.tok { display: inline; padding: 1px 2px; border-radius: 2px; }
.mask { color: #555; }
.new { background: #1b5e20; color: #a5d6a7; }
.demoted { background: #b71c1c; color: #ef9a9a; }
.changed { background: #e65100; color: #ffcc80; }
.kept { color: #ccc; }
.legend { margin: 10px 0; padding: 10px; background: #16213e; border-radius: 5px; }
.legend span { margin-right: 15px; padding: 3px 8px; border-radius: 3px; }
.summary { background: #16213e; padding: 10px; margin: 10px 0; border-radius: 5px; }
.correct { color: #4caf50; font-weight: bold; }
.wrong { color: #f44336; font-weight: bold; }
</style>
</head><body>
<h1>D3IM Generation Trace</h1>
<div class="legend">
    <b>Legend:</b>
    <span class="tok new">New (MASK→token)</span>
    <span class="tok demoted">Demoted (token→MASK)</span>
    <span class="tok changed">Changed (token→different)</span>
    <span class="tok kept">Kept (unchanged)</span>
    <span class="tok mask">████ (MASK)</span>
</div>
"""

    for trace in traces:
        question = trace["question"]
        gold = trace["gold"]

        html += f'<h2>Question: {question[:100]}...</h2>\n'
        html += f'<p>Gold answer: <b>{gold}</b></p>\n'

        for model_name, step_tokens in trace["models"].items():
            html += f'<h3>{model_name}</h3>\n'

            # Show selected steps
            show_steps = [0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 60, 63]
            show_steps = [s for s in show_steps if s < len(step_tokens)]

            total_dem = 0
            total_chg = 0

            for si, step in enumerate(show_steps):
                tokens = step_tokens[step]
                prev = step_tokens[step - 1] if step > 0 else torch.full_like(tokens, MASK_ID)

                html += f'<div class="step-row"><span class="step-label">Step {step:>3}:</span>'

                n_dem = 0
                n_chg = 0
                n_new = 0

                for pos in range(len(tokens)):
                    tid = tokens[pos].item()
                    pid = prev[pos].item()

                    if tid == MASK_ID:
                        if pid != MASK_ID:
                            # Demoted
                            word = tokenizer.decode([pid])
                            html += f'<span class="tok demoted" title="demoted: {word}">█</span>'
                            n_dem += 1
                        else:
                            html += '<span class="tok mask">█</span>'
                    else:
                        word = tokenizer.decode([tid]).replace('<', '&lt;').replace('>', '&gt;')
                        if pid == MASK_ID:
                            # New
                            html += f'<span class="tok new" title="new">{word}</span>'
                            n_new += 1
                        elif pid != tid:
                            # Changed
                            old_word = tokenizer.decode([pid]).replace('<', '&lt;').replace('>', '&gt;')
                            html += f'<span class="tok changed" title="{old_word}→{word}">{word}</span>'
                            n_chg += 1
                        else:
                            html += f'<span class="tok kept">{word}</span>'

                total_dem += n_dem
                total_chg += n_chg
                html += f' <span style="color:#888">  [+{n_new} △{n_chg} ▼{n_dem}]</span>'
                html += '</div>\n'

            # Final answer
            final = step_tokens[-1]
            final_text = tokenizer.decode(final, skip_special_tokens=True)
            match = re.search(r'####\s*(-?[\d,]+\.?\d*)', final_text)
            pred = match.group(1).replace(',', '').strip() if match else "none"
            is_correct = pred == str(gold).strip()
            cls = "correct" if is_correct else "wrong"

            html += f'<div class="summary">'
            html += f'Predicted: <span class="{cls}">{pred}</span> | '
            html += f'Gold: <b>{gold}</b> | '
            html += f'{"✅ CORRECT" if is_correct else "❌ WRONG"} | '
            html += f'Total demotions: {total_dem} | Total changes: {total_chg}'
            html += f'</div>\n'

    html += "</body></html>"

    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML saved to {output_path}")


if __name__ == "__main__":
    device = "cuda:0"

    data = json.load(open("eval_data/gsm8k_100.json"))
    # Pick interesting problems: Q4(273), Q8(60), Q9(122) — v5 correct, Original wrong
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
            step_tokens, _ = collect_trace(model, tokenizer, prompt, device)
            trace_entry["models"][name] = step_tokens
            del model
            torch.cuda.empty_cache()

        traces.append(trace_entry)

    generate_html(traces, tokenizer, "paper_figures/d3im_trace.html")
