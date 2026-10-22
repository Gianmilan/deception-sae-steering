# Project Execution Plan: Auditing Model Deception via SAE Steering Vectors

## Overview

* **Sub-field:** Mechanistic Interpretability & Model Control
* **Estimated Duration:** 2 Weeks (14 Days)
* **Target Hardware:** AMD Radeon RX 7800 XT (16 GB VRAM, Navi 32 / `gfx1101` RDNA 3) via AMD ROCm on EndeavourOS (Arch Linux)
* **Primary Stack:** `python 3.11` (Miniconda), `torch==2.9.1+rocm6.3`, `transformer-lens==3.9.0`, `sae-lens==6.54.6`, `datasets`, `gradio`, `matplotlib`
* **Target Models (100% Ungated Pipeline):**
  * **Primary Model:** `gemma-2-2b-it` (26 layers, $d_{\text{model}}=2304$, loaded seamlessly via ungated `unsloth/gemma-2-2b-it` mirror with identical weights) + `gemma-scope-2b-pt-res` SAEs.
  * **Baseline / Fast Prototyping:** `gpt2-small` (12 layers, $d_{\text{model}}=768$, natively open) + `gpt2-small-resid-post-v5-32k` SAEs.

---

## Architectural & Hardware Adaptations

### 1. AMD ROCm Runtime Integration
The development environment is optimized for AMD RDNA 3 hardware (`gfx1101`). All executions automatically inherit the following environment configuration (bundled in [.env](file:///home/gschutel/Projects/deception-sae-steering/.env)):
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
```
* **CUDA/HIP Abstraction:** Code uses `device="cuda:0"`; PyTorch ROCm translates calls to HIP transparently.
* **16 GB High-Bandwidth VRAM:** Fully accommodates `gemma-2-2b-it` in `bfloat16` (~4.8 GB) alongside 16k–32k SAE weights (~1.5 GB) and residual activation caches with zero CPU offloading.

### 2. 100% Ungated Model Pipeline
To ensure friction-free reproduction without Hugging Face Hub token dependencies or gating approvals:
* Automated routing in [`src/model_utils.py`](file:///home/gschutel/Projects/deception-sae-steering/src/model_utils.py) dynamically fetches `unsloth/gemma-2-2b-it` weights while preserving Google's architecture for Gemma-Scope SAE compatibility.
* Natively open `gpt2-small` is supported for instant sanity checks and rapid experimentation.

---

## Roadmap Overview & Status

```
Week 1: Pipeline & Steering [COMPLETED] ───────> Week 2: Evaluation & Artifacts [IN PROGRESS]
 ├── Days 1–2: Setup & Dataset       [DONE]      ├── Days 8–9:  Quantitative Benchmarking [NEXT]
 ├── Days 3–5: SAE Feature Hunting   [DONE]      ├── Days 10–11: Multi-Turn & CoT Testing
 └── Days 6–7: Activation Steering   [DONE]      └── Days 12–14: Technical Post & HF Demo
```

---

## Detailed Schedule

### Week 1: Environment, Feature Identification, and Steering Hooks

#### Days 1–2: Environment Setup & Dataset Preparation [STATUS: COMPLETED]
* **Objective:** Establish the development environment and construct the contrastive dataset required to elicit sycophantic lying versus truthfulness.
* **Implemented Deliverables:**
  1. Miniconda environment `deception-sae-steering` configured with Python 3.11 and PyTorch ROCm 6.3 (`torch 2.9.1+rocm6.3`).
  2. Verified GPU diagnostic check in [`main.py --verify-gpu`](file:///home/gschutel/Projects/deception-sae-steering/main.py) confirming bfloat16 tensor operations on RX 7800 XT.
  3. Constructed paired contrastive dataset in [`data/sycophancy_pairs.json`](file:///home/gschutel/Projects/deception-sae-steering/data/sycophancy_pairs.json) consisting of 50 prompt pairs across 23 domains (astronomy, physics, medicine, mathematics, biology, etc.):
     * **Prompt A (Biased/Sycophantic Incentive):** Prompts where the user strongly asserts a misconception and demands validation.
     * **Prompt B (Neutral Baseline):** Neutral question on the same topic without user framing.
     * **Ground Truth & Sycophantic Target:** Reference values for automated scoring.

---

#### Days 3–5: SAE Feature Hunting & Probing [STATUS: COMPLETED]
* **Objective:** Identify specific Sparse Autoencoder (SAE) latent features associated with sycophancy across intermediate-to-late residual stream layers.
* **Implemented Deliverables:**
  1. Residual stream activation caching implemented in [`src/sae_utils.py`](file:///home/gschutel/Projects/deception-sae-steering/src/sae_utils.py) across target layers ($L \in [8, 10]$ for GPT-2, $L \in [10, 20]$ for Gemma-2-2b).
  2. Automated SAE loader supporting `gpt2-small-resid-post-v5-32k` and `gemma-scope-2b-pt-res`.
  3. Computed differential SAE feature activations:
     $$\Delta \text{Act}_i = \frac{1}{N} \sum_{k=1}^N \text{SAE}(A_k)_i - \frac{1}{N} \sum_{k=1}^N \text{SAE}(B_k)_i$$
  4. Feature hunting CLI command in [`main.py --hunt-features`](file:///home/gschutel/Projects/deception-sae-steering/main.py) isolates top latents (e.g. Feature #19514 with $\Delta \text{Act} = +3.5985$) and saves steering vector checkpoints to `outputs/`.

---

#### Days 6–7: Manual Activation Steering Mechanisms [STATUS: COMPLETED]
* **Objective:** Build PyTorch forward hooks to modify the model's residual stream during generation.
* **Implemented Deliverables:**
  1. Extracted decoder steering direction from top identified feature:
     $$\vec{v}_{\text{steering}} = \frac{W_{\text{dec}}[i_{\text{feature}}]}{\|W_{\text{dec}}[i_{\text{feature}}]\|}$$
  2. Implemented flexible forward hooks in [`src/hooks.py`](file:///home/gschutel/Projects/deception-sae-steering/src/hooks.py) with options for all-token steering vs. generation-token steering:
     $$\text{activations}[:, \text{pos}, :] \leftarrow \text{activations}[:, \text{pos}, :] - \alpha \cdot \vec{v}_{\text{steering}}$$
  3. Built qualitative $\alpha$-sweep CLI in [`main.py --steer`](file:///home/gschutel/Projects/deception-sae-steering/main.py), demonstrating direct transition from unsteered sycophantic waffling ($\alpha = 0.0$) to factual refutation ("*You're wrong.*", $\alpha \ge 2.0$).

---

### Week 2: Evaluation, Edge Cases, and Portfolio Deliverables

#### Days 8–9: Quantitative Evaluation Pipeline [STATUS: NEXT MILESTONE]
* **Objective:** Benchmark the trade-off between sycophancy mitigation and general model performance.
* **Tasks:**
  1. **Refusal to Deceive Rate (RDR):** Benchmark the steered model on the held-out evaluation split of sycophancy prompts. Automatically classify whether the model pushed back or submitted to user bias.
  2. **General Capability Preservation (Perplexity & MMLU):** Evaluate model perplexity using [`src/evals.py::compute_perplexity`](file:///home/gschutel/Projects/deception-sae-steering/src/evals.py) on standard text corpora and MMLU subsets to ensure $\alpha$ steering preserves language modeling competence.
  3. **Ablation Baseline:** Benchmark SAE decoder steering against Contrastive Activation Addition (CAA / simple mean-difference activation subtraction without SAE decomposition).

---

#### Days 10–11: Multi-Turn Conversations & CoT Analysis [STATUS: UPCOMING]
* **Objective:** Assess the robustness of activation steering under multi-turn pressure and explicit reasoning steps.
* **Tasks:**
  1. Simulate multi-turn pushback conversations where the user repeatedly challenges the model's factual correction.
  2. Evaluate Chain-of-Thought (CoT) traces to observe whether activation steering alters intermediate reasoning tokens or directly influences final token selection.

---

#### Days 12–14: Write-up, Repository Cleanup, and Interactive Demo [STATUS: UPCOMING]
* **Objective:** Package the research into public portfolio artifacts.
* **Tasks:**
  1. **Codebase Polish:** Finalize modular structure (`src/`, `data/`, `outputs/`, `docs/`, `tests/`) and ensure clean reproducibility scripts.
  2. **Technical Post:** Draft an in-depth research write-up (Alignment Forum / LessWrong format) detailing:
     * Problem statement & sycophancy threat model.
     * Isolated SAE latents and feature interpretations.
     * Pareto curves ($\alpha$ vs. Truthfulness % vs. Perplexity / Capability Loss).
     * Failure modes and optimal injection layers.
  3. **Interactive Demo:** Build a lightweight Gradio application (`app.py`) allowing interactive prompt testing and live $\alpha$ adjustment, hosted locally or on Hugging Face Spaces.

---

## Core Metrics & Target Benchmarks

| Metric | Baseline (Unsteered) | Target / Success Criteria | Status |
| :--- | :--- | :--- | :--- |
| **Truthfulness under Bias (RDR)** | $< 30\%$ | $> 85\%$ truthful responses on evaluation set | In progress |
| **Perplexity Degradation** | $1.0\times$ (Baseline) | Perplexity increase $< 1.5\times$ baseline | Framework ready (`src/evals.py`) |
| **General Competence Loss (MMLU)** | Baseline accuracy | $< 5\%$ accuracy drop relative to baseline | In progress |
| **Local AMD GPU Execution Time** | N/A | Feature hunting + sweep $< 5$ minutes on RX 7800 XT | **Verified (< 1 min)** |

---

## Project Codebase Structure

```
deception-sae-steering/
├── .env                     # AMD ROCm hardware flags & default configurations
├── README.md                # Project documentation and quickstart
├── main.py                  # Unified CLI for diagnostics, hunting, and steering
├── data/
│   └── sycophancy_pairs.json # 50 contrastive prompt pairs across 23 domains
├── docs/
│   └── execution_plan.md    # Detailed project roadmap and progress tracking
├── src/
│   ├── config.py            # Global configuration & environment loader
│   ├── model_utils.py       # Dual-model loader with ungated fallbacks
│   ├── dataset.py           # Dataset loader & train/eval split utilities
│   ├── sae_utils.py         # SAELens loader, residual cache & ΔAct computation
│   ├── hooks.py             # Residual stream steering forward hooks
│   └── evals.py             # Qualitative sweeps & perplexity evaluation
└── outputs/                 # Saved steering vectors and evaluation checkpoints
```
