#!/bin/bash
# ============================================================
# Master experiment runner — 8-GPU parallel
# ============================================================
# Executes all experiments from experiment_plan.MD
# Each experiment runs on a dedicated GPU with full logging.
#
# Usage:
#   bash dsl_llada/run_all_experiments.sh          # Run all
#   bash dsl_llada/run_all_experiments.sh status    # Check progress
#
# Logs: eval_results/sde_gen_formal/logs/
# Results: eval_results/sde_gen_formal/

set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SCRIPT="dsl_llada/eval_sde_gen_formal.py"
BLOCK_SDE="dsl_llada/eval_block_sde.py"
PROMPTS="eval_data/sde_prompts_200.json"
PREFIXES="eval_data/wikitext_prefix_200.json"
LOGDIR="eval_results/sde_gen_formal/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# Status check mode
# ============================================================
if [ "${1:-}" = "status" ]; then
    echo "=== Result files ==="
    ls eval_results/sde_gen_formal/*.json 2>/dev/null | wc -l
    echo "=== Running processes ==="
    ps aux | grep "eval_sde_gen_formal\|eval_block_sde\|eval_context_robust" | grep python | grep -v grep | \
        awk '{print $2, $12, $13, $14, $15}'
    echo "=== GPU usage ==="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    exit 0
fi

# ============================================================
# GPU 0: Exp 1.1 — β=1 remask + noEOS, NFE curve
# ============================================================
(
    log "GPU0: Exp 1.1 — β=1 remask+noEOS NFE curve"
    for nfe in 8 16 32 64 128; do
        log "GPU0: b1 remask+noEOS nfe=$nfe"
        CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
            --model_key b1 --method remask --nfe $nfe \
            --tag nfe_curve --gpu 0 --suppress_eos --block_length 32 \
            --prompts "$PROMPTS" --seed 42
    done
    # Also β=1 remask + eos_inf (EOS patch, not noEOS trick)
    for nfe in 8 16 32 64 128; do
        log "GPU0: b1 remask+eosInf nfe=$nfe"
        CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
            --model_key b1 --method remask --nfe $nfe \
            --tag nfe_curve --gpu 0 --eos_inf \
            --prompts "$PROMPTS" --seed 42
    done
    log "GPU0: DONE"
) > "$LOGDIR/gpu0_exp1.1_${TIMESTAMP}.log" 2>&1 &
PID0=$!

# ============================================================
# GPU 1: Exp 1.1 — Original + eos_inf + b32+noEOS (if not done)
# ============================================================
(
    log "GPU1: Exp 1.1 — Original eos_inf + b32+noEOS NFE curve"
    # eos_inf settings
    for nfe in 8 16 32 64 128; do
        log "GPU1: original eos_inf nfe=$nfe"
        CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
            --model_key original --method remask --nfe $nfe \
            --tag nfe_curve --gpu 0 --eos_inf \
            --prompts "$PROMPTS" --seed 42
    done
    # b32 + noEOS
    for nfe in 8 16 32 64 128; do
        log "GPU1: original b32+noEOS nfe=$nfe"
        CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
            --model_key original --method remask --nfe $nfe \
            --tag nfe_curve --gpu 0 --block_length 32 --suppress_eos \
            --prompts "$PROMPTS" --seed 42
    done
    log "GPU1: DONE"
) > "$LOGDIR/gpu1_exp1.1_orig_${TIMESTAMP}.log" 2>&1 &
PID1=$!

# ============================================================
# GPU 2: Exp B.1 — Block SDE NFE-matched (configs 1-4)
# GSM8K full 1319 problems, gen=256, seed=42
# ============================================================
(
    log "GPU2: Exp B.1 — Block SDE (B=32, S=4,8,16,32)"
    for spb in 4 8 16 32; do
        nfe=$((spb * 2 * 256 / 32))  # Heun NFE per block × num blocks
        log "GPU2: block_sde B=32 S=$spb (NFE=$nfe)"
        CUDA_VISIBLE_DEVICES=2 python "$BLOCK_SDE" \
            --gpu 0 --n_problems 1319 --gen_length 256 \
            --block_sizes 32 --steps_per_block $spb \
            --skip_baselines \
            --only "block_32_${spb}" \
            --bi 2.0 --ns 0.01
    done
    log "GPU2: DONE"
) > "$LOGDIR/gpu2_expB1_block32_${TIMESTAMP}.log" 2>&1 &
PID2=$!

# ============================================================
# GPU 3: Exp B.1 — Discrete block remask NFE-matched
# ============================================================
(
    log "GPU3: Exp B.1 — Discrete block remask (B=32, S=8,16,32)"
    for steps in 8 16 32; do
        nfe=$((steps * 256 / 32))  # steps per block × num blocks
        log "GPU3: remask B=32 steps=$steps (NFE=$nfe)"
        CUDA_VISIBLE_DEVICES=3 python "$BLOCK_SDE" \
            --gpu 0 --n_problems 1319 --gen_length 256 \
            --skip_baselines \
            --only "remask_${steps}_b32"
    done
    log "GPU3: DONE"
) > "$LOGDIR/gpu3_expB1_remask_${TIMESTAMP}.log" 2>&1 &
PID3=$!

# ============================================================
# GPU 4: Exp 2.1 — Context robustness degradation curve
# 200 texts, mask=30%, corruption 0-50%, 5 models
# ============================================================
(
    log "GPU4: Exp 2.1 — Context robustness degradation curve"
    CUDA_VISIBLE_DEVICES=4 python3 -c "
import torch, sys, os, json, time
sys.path.insert(0, 'LLaDA')
sys.path.insert(0, 'dsl_llada')
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MASK_ID = 126336
device = 'cuda:0'

# Load texts
texts_data = json.load(open('eval_data/texts_100.json'))[:200] if os.path.exists('eval_data/texts_100.json') else []
# Use sde_prompts if texts not enough
if len(texts_data) < 200:
    texts_data = json.load(open('eval_data/texts_100.json'))

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

models = {
    'original': 'GSAI-ML/LLaDA-8B-Instruct',
    'mdm_cpt': 'checkpoints/mdm_baseline_1k/checkpoint-1000',
    'xdlm': 'checkpoints/xdlm_baseline_1k/checkpoint-1000',
    'sm_b2': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
    'replace': 'checkpoints/pertoken_b2_replace_1k/checkpoint-1000',
}

corruption_rates = [0, 5, 10, 15, 20, 25, 30, 40, 50]
mask_rate = 0.3
seed = 42
results = {}

for model_key, ckpt in models.items():
    print(f'\\nLoading {model_key}...')
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    model_results = {}
    for c_rate in corruption_rates:
        torch.manual_seed(seed)
        np.random.seed(seed)

        total_correct = 0
        total_masked = 0

        for item in texts_data:
            text = item['text']
            ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, add_special_tokens=False)['input_ids'].to(device)
            L = ids.shape[1]
            if L < 10: continue

            # Create mask: 30% masked
            n_mask = int(L * mask_rate)
            perm = torch.randperm(L)
            mask_pos = perm[:n_mask]
            unmask_pos = perm[n_mask:]

            # Corrupt c% of unmasked positions
            n_corrupt = int(len(unmask_pos) * c_rate / 100)
            if n_corrupt > 0:
                corrupt_idx = unmask_pos[torch.randperm(len(unmask_pos))[:n_corrupt]]
                corrupted_ids = ids.clone()
                corrupted_ids[0, corrupt_idx] = torch.randint(100, 126000, (n_corrupt,), device=device)
            else:
                corrupted_ids = ids.clone()

            # Replace masked positions with MASK
            input_ids = corrupted_ids.clone()
            input_ids[0, mask_pos] = MASK_ID

            # Forward pass
            with torch.no_grad():
                logits = model(input_ids).logits

            # Check accuracy on masked positions
            preds = logits[0, mask_pos].argmax(dim=-1)
            gold = ids[0, mask_pos]
            total_correct += (preds == gold).sum().item()
            total_masked += len(mask_pos)

        acc = total_correct / total_masked if total_masked > 0 else 0
        model_results[c_rate] = round(acc, 4)
        print(f'  {model_key} corrupt={c_rate}%: acc={acc:.4f}')

    results[model_key] = model_results
    del model
    torch.cuda.empty_cache()

# Save
out_path = 'eval_results/sde_gen_formal/context_robustness_curve.json'
with open(out_path, 'w') as f:
    json.dump({'mask_rate': mask_rate, 'corruption_rates': corruption_rates, 'seed': seed, 'results': results}, f, indent=2)
print(f'\\nSaved to {out_path}')
"
    log "GPU4: DONE"
) > "$LOGDIR/gpu4_exp2.1_robustness_${TIMESTAMP}.log" 2>&1 &
PID4=$!

# ============================================================
# GPU 5: Exp 2.2 — Multi mask rate robustness
# ============================================================
(
    log "GPU5: Exp 2.2 — Multi mask rate robustness"
    CUDA_VISIBLE_DEVICES=5 python3 -c "
import torch, sys, os, json, numpy as np
sys.path.insert(0, 'LLaDA')
from transformers import AutoModelForCausalLM, AutoTokenizer

MASK_ID = 126336
device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
texts_data = json.load(open('eval_data/texts_100.json'))

models = {
    'original': 'GSAI-ML/LLaDA-8B-Instruct',
    'mdm_cpt': 'checkpoints/mdm_baseline_1k/checkpoint-1000',
    'sm_b2': 'checkpoints/pertoken_b2_d100_1k/checkpoint-1000',
    'replace': 'checkpoints/pertoken_b2_replace_1k/checkpoint-1000',
}

mask_rates = [0.3, 0.5, 0.7]
corruption_rates = [0, 10, 20, 30]
seed = 42
results = {}

for model_key, ckpt in models.items():
    print(f'\\nLoading {model_key}...')
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    model_results = {}

    for mr in mask_rates:
        for cr in corruption_rates:
            torch.manual_seed(seed); np.random.seed(seed)
            total_correct = total_masked = 0

            for item in texts_data:
                ids = tokenizer(item['text'], return_tensors='pt', truncation=True, max_length=512, add_special_tokens=False)['input_ids'].to(device)
                L = ids.shape[1]
                if L < 10: continue
                n_mask = int(L * mr)
                perm = torch.randperm(L)
                mask_pos = perm[:n_mask]
                unmask_pos = perm[n_mask:]
                n_corrupt = int(len(unmask_pos) * cr / 100)
                corrupted_ids = ids.clone()
                if n_corrupt > 0:
                    ci = unmask_pos[torch.randperm(len(unmask_pos))[:n_corrupt]]
                    corrupted_ids[0, ci] = torch.randint(100, 126000, (n_corrupt,), device=device)
                input_ids = corrupted_ids.clone()
                input_ids[0, mask_pos] = MASK_ID
                with torch.no_grad():
                    logits = model(input_ids).logits
                preds = logits[0, mask_pos].argmax(dim=-1)
                total_correct += (preds == ids[0, mask_pos]).sum().item()
                total_masked += len(mask_pos)

            acc = total_correct / total_masked if total_masked > 0 else 0
            key = f'mask{int(mr*100)}_corrupt{cr}'
            model_results[key] = round(acc, 4)
            print(f'  {model_key} mask={mr} corrupt={cr}%: acc={acc:.4f}')

    results[model_key] = model_results
    del model; torch.cuda.empty_cache()

with open('eval_results/sde_gen_formal/multi_mask_robustness.json', 'w') as f:
    json.dump({'mask_rates': mask_rates, 'corruption_rates': corruption_rates, 'seed': seed, 'results': results}, f, indent=2)
print('\\nSaved.')
"
    log "GPU5: DONE"
) > "$LOGDIR/gpu5_exp2.2_multimask_${TIMESTAMP}.log" 2>&1 &
PID5=$!

# ============================================================
# GPU 6: Exp B.2 — Block size sweep (B=16,32,64,128,256, S=16 fixed)
# ============================================================
(
    log "GPU6: Exp B.2 — Block size sweep"
    for bs in 16 32 64 128; do
        log "GPU6: block_sde B=$bs S=16"
        CUDA_VISIBLE_DEVICES=6 python "$BLOCK_SDE" \
            --gpu 0 --n_problems 200 --gen_length 256 \
            --block_sizes $bs --steps_per_block 16 \
            --skip_baselines \
            --only "block_${bs}_16" \
            --bi 2.0 --ns 0.01
    done
    # Also pure SDE and remask baselines for B sweep
    log "GPU6: pure SDE baseline"
    CUDA_VISIBLE_DEVICES=6 python "$BLOCK_SDE" \
        --gpu 0 --n_problems 200 --gen_length 256 \
        --only "pure_sde_32" --bi 2.0 --ns 0.01
    log "GPU6: DONE"
) > "$LOGDIR/gpu6_expB2_bsweep_${TIMESTAMP}.log" 2>&1 &
PID6=$!

# ============================================================
# GPU 7: Original baselines (continuing what was running)
# Already running via eval_original_baselines.sh
# If done, run Exp B.3 MATH-500 validation
# ============================================================
(
    log "GPU7: Waiting for original baselines to finish..."
    # Check if original baselines are still running
    while ps aux | grep "eval_original_baselines" | grep -v grep > /dev/null 2>&1; do
        sleep 30
    done
    log "GPU7: Original baselines done, starting B.3 MATH-500"

    # Exp B.3: MATH-500 with best Block SDE config
    log "GPU7: Block SDE B=32 S=16 on MATH-500"
    CUDA_VISIBLE_DEVICES=7 python "$BLOCK_SDE" \
        --gpu 0 --n_problems 500 --gen_length 512 \
        --block_sizes 32 --steps_per_block 16 \
        --skip_baselines \
        --only "block_32_16" \
        --bi 2.0 --ns 0.01
    log "GPU7: DONE"
) > "$LOGDIR/gpu7_expB3_math_${TIMESTAMP}.log" 2>&1 &
PID7=$!

# ============================================================
# Summary
# ============================================================
log "All 8 GPUs launched:"
log "  GPU 0 (PID $PID0): Exp 1.1 — β=1 remask variants"
log "  GPU 1 (PID $PID1): Exp 1.1 — Original eos_inf + b32+noEOS"
log "  GPU 2 (PID $PID2): Exp B.1 — Block SDE (B=32)"
log "  GPU 3 (PID $PID3): Exp B.1 — Discrete block remask"
log "  GPU 4 (PID $PID4): Exp 2.1 — Context robustness curve"
log "  GPU 5 (PID $PID5): Exp 2.2 — Multi mask rate robustness"
log "  GPU 6 (PID $PID6): Exp B.2 — Block size sweep"
log "  GPU 7 (PID $PID7): Exp B.3 — MATH-500 + remaining"
log ""
log "Monitor with: bash dsl_llada/run_all_experiments.sh status"
log "Logs in: $LOGDIR/"

# Wait for all
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7
log "========== ALL EXPERIMENTS COMPLETE =========="
