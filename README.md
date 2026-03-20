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
---

## Repository Structure
```
.
├── code/
│   ├── kv_cache_scaling.ipynb      # Experiment 1: KV cache size vs sequence length for MHA vs GQA vs H2O
│   ├── FP8_Cache_vs_SL.ipynb       # Experiment 1: KV cache size vs sequence length for FP16 vs FP8 quantisation
│   ├── niah_exp_h2o                # Experiment 2: Needle-in-a-Haystack with H2O eviction
│   ├── niah_fp8.ipynb              # Experiment 2: Needle-in-a-Haystack with FP8/INT8
├── plots/                          # Generated result figures
└── README.md
```

---

## Experiments

### Experiment 1 — KV Cache Size vs Sequence Length
Measures how KV cache memory (GB) scales with sequence length for MHA, GQA, H2O and FP8 quantisation.
Uses LLaMA-3-8B architecture config (no pretrained weights needed).

- MHA and GQA sizes are computed theoretically from the model config
- H2O is simulated token-by-token with a fixed cache budget of 1,024 tokens
- FP8 size is derived analytically from the MHA cache size

### Experiment 2 — Needle in a Haystack (NIAH)
Evaluates whether compression causes the model to forget facts buried deep in context.
A random 6-digit magic number is hidden at varying depths and the model is asked to retrieve it.

**H2O NIAH** — measures the effective context window under eviction:

**FP8 / INT8 NIAH** — measures whether quantization degrades retrieval:

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

All other experiments are self-contained Jupyter notebooks —
open in Google Colab and run top to bottom.

---

## Hardware

All experiments run on a **G4 GPU (32GB VRAM)** via Google Colab.
LLaMA-3-8B config is used for memory scaling experiments (random weights, no download needed).


## Authors

- Abhinandan Sharma
- Sanskriti Goyal  
- Nisha Kumar

ECE 226 — Group 6 | UC San Diego | March 2026
