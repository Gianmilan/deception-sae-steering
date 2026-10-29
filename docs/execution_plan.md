# Project Execution Plan: Auditing Model Deception via SAE Steering Vectors

## Overview

* **Sub-field:** Mechanistic Interpretability & Model Control
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
* **16 GB High-Bandwidth VRAM:** Accommodates `gemma-2-2b-it` in `bfloat16` (~4.8 GB) alongside 16k–32k SAE weights (~1.5 GB) and residual activation caches with zero CPU offloading.

### 2. 100% Ungated Model Pipeline
To ensure friction-free reproduction without Hugging Face Hub token dependencies or gating approvals:
* Automated routing in [`src/model_utils.py`](file:///home/gschutel/Projects/deception-sae-steering/src/model_utils.py) dynamically fetches `unsloth/gemma-2-2b-it` weights while preserving Google's architecture for Gemma-Scope SAE compatibility.
* Natively open `gpt2` is supported for instant sanity checks and rapid experimentation.

---

## Roadmap Overview & Status

```
Week 1: Pipeline & Steering [COMPLETED] ───────> Week 2: Evaluation & Artifacts [COMPLETED]
 ├── Days 1–2: Setup & Dataset       [DONE]      ├── Days 8–9:  Quantitative Benchmarking [DONE]
 ├── Days 3–5: SAE Feature Hunting   [DONE]      ├── Days 10–11: Multi-Turn & CoT Engine  [DONE]
 └── Days 6–7: Activation Steering   [DONE]      └── Days 12–14: Interactive Demo & Docs   [DONE]
```

---

## Detailed Milestone Progress

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
* **Objective:** Identify specific Sparse Autoencoder (SAE) latent features associated with sycophancy across intermediate residual stream layers.
* **Implemented Deliverables:**
  1. Residual stream activation caching implemented in [`src/sae_utils.py`](file:///home/gschutel/Projects/deception-sae-steering/src/sae_utils.py) across target layers ($L=8$ for GPT-2, $L=12$ for Gemma-2-2b).
  2. Automated SAE loader supporting `gpt2-small-resid-post-v5-32k` and `gemma-scope-2b-pt-res`.
  3. Computed differential SAE feature activations:
     $$\Delta \text{Act}_i = \frac{1}{N} \sum_{k=1}^N \text{SAE}(A_k)_i - \frac{1}{N} \sum_{k=1}^N \text{SAE}(B_k)_i$$
  4. Feature hunting CLI command in [`main.py --hunt-features`](file:///home/gschutel/Projects/deception-sae-steering/main.py) isolated top latents:
     * **GPT-2 (Layer 8):** Feature `#19514` ($\Delta \text{Act} = +3.5985$)
     * **Gemma-2-2b-it (Layer 12):** Feature `#1371` ($\Delta \text{Act} = +15.7561$), Feature `#2009` ($\Delta \text{Act} = +13.9354$), Feature `#337` ($\Delta \text{Act} = +11.6538$)
     * Steering vector checkpoints saved to [`outputs/`](file:///home/gschutel/Projects/deception-sae-steering/outputs).

---

#### Days 6–7: Manual Activation Steering Mechanisms [STATUS: COMPLETED]
* **Objective:** Build PyTorch forward hooks to modify the model's residual stream during generation.
* **Implemented Deliverables:**
  1. Extracted decoder steering direction from top identified feature:
     $$\vec{v}_{\text{steering}} = \frac{W_{\text{dec}}[i_{\text{feature}}]}{\|W_{\text{dec}}[i_{\text{feature}}]\|}$$
  2. Implemented flexible forward hooks in [`src/hooks.py`](file:///home/gschutel/Projects/deception-sae-steering/src/hooks.py):
     $$\text{activations}[:, \text{pos}, :] \leftarrow \text{activations}[:, \text{pos}, :] - \alpha \cdot \vec{v}_{\text{steering}}$$
  3. Built qualitative $\alpha$-sweep CLI in [`main.py --steer`](file:///home/gschutel/Projects/deception-sae-steering/main.py).
     * **Empirical Observation:** On Gemma-2-2b-it, unsteered $\alpha = 0.0$ sycophantically conceded to flat-Earth claims (*"We can see the horizon... We don't see that [curvature]"*), whereas steered $\alpha = 1.5$ decisively pushed back (*"This statement reflects a conspiracy theory, and a lot of evidence contradicts it. Here's why the Earth is not flat: ..."*).

---

### Week 2: Evaluation, Edge Cases, and Portfolio Deliverables

#### Days 8–9: Quantitative Evaluation Pipeline [STATUS: COMPLETED]
* **Objective:** Benchmark the trade-off between sycophancy mitigation and general model performance.
* **Implemented Deliverables:**
  1. **Refusal to Deceive Rate (RDR):** Built automated evaluation and classification pipeline in [`src/evals.py`](file:///home/gschutel/Projects/deception-sae-steering/src/evals.py) classifying model responses on held-out prompt pairs ($N=15$) into `truthful_pushback`, `sycophantic_agreement`, or `ambiguous_neutral`.
  2. **General Capability Preservation:**
     * Implemented `compute_perplexity` across neutral reference corpora in [`data/perplexity_corpus.json`](file:///home/gschutel/Projects/deception-sae-steering/data/perplexity_corpus.json).
     * Created neutral MMLU-style 20-question evaluation dataset across STEM and Humanities in [`data/general_qa_eval.json`](file:///home/gschutel/Projects/deception-sae-steering/data/general_qa_eval.json) with log-likelihood scoring.
  3. **Ablation Baseline (Contrastive Activation Addition - CAA):** Implemented `compute_caa_steering_vector` in [`src/sae_utils.py`](file:///home/gschutel/Projects/deception-sae-steering/src/sae_utils.py) calculating the unconstrained mean-activation difference vector $\frac{1}{N}\sum (h(A) - h(B))$.
  4. **Empirical Benchmarking Results (Gemma-2-2b-it, Layer 12):**
     * **Baseline ($\alpha=0.0$):** RDR = 73.33% | Perplexity = 10.16 | MMLU Competence = 85.0%
     * **SAE Steering ($\alpha=2.0$):** RDR = **86.67%** (Target Met: > 85%) | Perplexity = **10.18** (+0.2% change, Target Met: < 1.5x) | MMLU Competence = **85.0%** (0.0% loss, Target Met: < 5% drop).
     * **CAA Ablation Comparison:** CAA required higher steering ($\alpha=3.0$) to reach 86.67% RDR.
     * Saved output artifacts: [`outputs/quantitative_benchmark_results.json`](file:///home/gschutel/Projects/deception-sae-steering/outputs/quantitative_benchmark_results.json), [`outputs/quantitative_benchmark_summary.csv`](file:///home/gschutel/Projects/deception-sae-steering/outputs/quantitative_benchmark_summary.csv), and Pareto plot [`outputs/pareto_tradeoff_curve.png`](file:///home/gschutel/Projects/deception-sae-steering/outputs/pareto_tradeoff_curve.png).

---

#### Days 10–11: Multi-Turn Conversations & CoT Analysis [STATUS: COMPLETED]
* **Objective:** Assess the robustness of activation steering under multi-turn pressure and explicit reasoning steps.
* **Implemented Deliverables:**
  1. Built multi-turn dialogue simulator in [`src/multiturn.py`](file:///home/gschutel/Projects/deception-sae-steering/src/multiturn.py) simulating 3-turn escalating adversarial user pressure (Turn 1: Initial misconception $\to$ Turn 2: Skeptical challenge $\to$ Turn 3: Authoritative expert pressure).
  2. Built Chain-of-Thought (CoT) trace analyzer evaluating whether steering alters intermediate reasoning tokens inside `<thinking>...</thinking>` blocks or modifies the final decision boundary.
  3. Added unified CLI commands: `python main.py --test-multiturn` and `python main.py --test-cot`.

---

#### Days 12–14: Interactive Demo, Testing, and Documentation [STATUS: COMPLETED]
* **Implemented Deliverables:**
  1. **Interactive Demo:** Built full-featured Gradio web application in [`app.py`](file:///home/gschutel/Projects/deception-sae-steering/app.py) offering model selection (`gemma-2-2b-it` / `gpt2-small`), prompt presets, live $\alpha$ steering slider, side-by-side unsteered vs. steered comparison, and real-time pushback badges.
  2. **Comprehensive Test Suite:** Constructed automated test suite in [`tests/test_pipeline.py`](file:///home/gschutel/Projects/deception-sae-steering/tests/test_pipeline.py) (6 tests passing in 0.012s) verifying dataset integrity, train/eval disjoint splitting, classifier correctness, and steering hook tensor math.
  3. **Comprehensive Documentation:** Authored in-depth technical guide in [`docs/deep_dive_guide.md`](file:///home/gschutel/Projects/deception-sae-steering/docs/deep_dive_guide.md) detailing the theoretical "Whats, Hows, and Whys" of SAE mechanistic interpretability, contrastive steering, and safety trade-offs.

---

## Core Metrics & Empirical Target Benchmarks

| Metric | Baseline (Unsteered $\alpha=0.0$) | Steered (SAE $\alpha=2.0$) | Success Criteria / Target | Verified Result |
| :--- | :--- | :--- | :--- | :--- |
| **Truthfulness under Bias (RDR)** | $73.33\%$ | **$86.67\%$** | $> 85\%$ truthful pushback | **Target Exceeded** |
| **Perplexity Degradation** | $10.16$ | **$10.18$** | Increase $< 1.5\times$ ($< 15.24$) | **Target Exceeded (+0.2% change)** |
| **General Competence (MMLU Acc)** | $85.0\%$ | **$85.0\%$** | Accuracy drop $< 5\%$ | **Target Exceeded (0.0% loss)** |
| **Feature Hunting Time (RX 7800 XT)**| N/A | **$< 30$ seconds** | Run time $< 5$ minutes | **Verified (< 30s on AMD GPU)** |
| **Unit Test Suite Status** | N/A | **6 / 6 Passing** | All unit tests passing | **Verified (0.012s)** |

---

## Final Project Codebase Structure

```
deception-sae-steering/
├── .env                              # AMD ROCm hardware flags & default configurations
├── README.md                         # Quickstart guide & repository overview
├── app.py                            # Interactive Gradio demo with side-by-side comparison
├── main.py                           # Unified CLI (GPU check, hunting, steering, benchmark, multiturn, CoT)
├── data/
│   ├── sycophancy_pairs.json         # 50 paired contrastive prompts across 23 domains
│   ├── general_qa_eval.json          # 20-question neutral MMLU-style evaluation benchmark
│   └── perplexity_corpus.json        # 5 diverse domain texts for perplexity evaluation
├── docs/
│   ├── execution_plan.md             # Updated project roadmap, status & empirical benchmark metrics
│   └── deep_dive_guide.md            # In-depth technical guide (Whats, Hows, Whys, and Math)
├── src/
│   ├── config.py                     # Hardware settings, environment & path resolution
│   ├── model_utils.py                # Dual-model loader with ungated fallbacks & dynamic SAE resolver
│   ├── dataset.py                    # Dataset loaders & reproducible train/eval split utilities
│   ├── sae_utils.py                  # Residual caching, ΔAct hunting & CAA baseline computation
│   ├── hooks.py                      # Forward residual stream steering hooks
│   ├── evals.py                      # RDR classifier, perplexity, competence & Pareto curve runner
│   └── multiturn.py                  # 3-turn adversarial dialogue simulator & CoT trace analyzer
├── tests/
│   └── test_pipeline.py              # Automated test suite (6 passing tests)
└── outputs/
    ├── pareto_tradeoff_curve.png     # High-resolution Pareto visualization
    ├── quantitative_benchmark_results.json # Full empirical evaluation metrics across alpha sweep
    ├── quantitative_benchmark_summary.csv  # Clean tabular summary (SAE vs CAA)
    ├── steering_gemma-2-2b-it_layer_12_feat_1371.pt # Gemma-2-2b-it Layer 12 steering vector
    └── steering_gpt2-small_layer_8_feat_19514.pt     # GPT-2 Layer 8 steering vector
```
