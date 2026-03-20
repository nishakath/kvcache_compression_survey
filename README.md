# KV Cache Compression Survey
### ECE 226 — Optimization and Acceleration of Deep Learning on Hardware Platforms
### UC San Diego | Group 6

---

## Overview

This project benchmarks KV cache compression techniques for large language model inference
on resource-constrained devices. As context length grows, the KV cache can exceed model
weights in memory — causing OOM crashes or severe latency on edge hardware.

We use **MHA (Multi-Head Attention)** as our baseline and evaluate three compression techniques:

| Technique | Category | Memory Reduction | Affects Context? |
|---|---|---|---|
| **GQA** (Grouped Query Attention) | Architectural | 4× | ❌ No |
| **H2O** (Heavy Hitter Oracle) | Eviction-based | Unbounded | ✅ Yes |
| **FP8 / INT8 Quantization** | Quantization | 2× | Minimal |

**Core finding:** GQA and FP8 reduce memory without sacrificing context retention.
H2O achieves the most aggressive compression but collapses the effective context
window from 4,096 → 728 tokens — a 6× reduction for only a 4× cache saving.

---

## Repository Structure
```
.
├── code/
│   ├── kv_cache_scaling.ipynb      # Experiment 1: KV cache size vs sequence length
│   ├── niah_h2o.py                 # Experiment 2: Needle-in-a-Haystack with H2O eviction
│   ├── niah_fp8.ipynb              # Experiment 2: Needle-in-a-Haystack with FP8/INT8
│   └── h2o_utils.py                # H2O eviction engine
├── plots/                          # Generated result figures
└── README.md
```

---

## Experiments

### Experiment 1 — KV Cache Size vs Sequence Length
Measures how KV cache memory (GB) scales with sequence length for MHA, GQA, and H2O.
Uses LLaMA-3-8B architecture config (no pretrained weights needed).

- MHA and GQA sizes are computed theoretically from the model config
- H2O is simulated token-by-token with a fixed cache budget of 1,024 tokens
- Key result: H2O plateaus at a fixed memory cap; GQA scales at exactly 4× below MHA

### Experiment 2 — Needle in a Haystack (NIAH)
Evaluates whether compression causes the model to forget facts buried deep in context.
A random 6-digit magic number is hidden at varying depths and the model is asked to retrieve it.

**H2O NIAH** — measures the effective context window under eviction:
- Baseline (no eviction): retrieval succeeds across the full 4,096 token context
- H2O (budget=1,024): effective context window collapses to ~728 tokens
- Result: 4× cache compression causes 6× context loss

**FP8 / INT8 NIAH** — measures whether quantization degrades retrieval:
- INT8 quantization preserves retrieval accuracy across all tested context depths
- Confirms that precision reduction does not meaningfully shrink the effective context window

---

## Setup
```bash
pip install torch transformers accelerate bitsandbytes seaborn matplotlib
```

### Model Access
LLaMA-2 and LLaMA-3 are gated on Hugging Face — accept the license and authenticate:
```python
from huggingface_hub import notebook_login
notebook_login()
```

### Running H2O NIAH
```bash
# With H2O eviction
python code/niah_h2o.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --budget 1024 \
  --lengths 512 1024 2048 4096 \
  --depths 0.1 0.3 0.5 0.7 0.9

# Baseline (no eviction)
python code/niah_h2o.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --budget 1024 \
  --baseline
```

All other experiments are self-contained Jupyter notebooks —
open in Google Colab and run top to bottom.

---

## Hardware

All experiments run on a **G4 GPU (15GB VRAM)** via Google Colab.
LLaMA-3-8B config is used for memory scaling experiments (random weights, no download needed).
LLaMA-2-7b-chat-hf is used for NIAH quality experiments (pretrained weights required).

---

## Results Summary

| Experiment | MHA (baseline) | GQA | H2O | FP8/INT8 |
|---|---|---|---|---|
| Cache size @ 8k tokens | ~8 GB | ~2 GB | ~0.12 GB (capped) | ~4 GB |
| Effective context window | 4,096 tokens | 4,096 tokens | **728 tokens** | ~4,096 tokens |

---

## Authors

- Abhinandan Sharma
- Sanskriti Goyal  
- Nisha Kumar

ECE 226 — Group 6 | UC San Diego | March 2026
