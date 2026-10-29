# Auditing Model Deception via SAE Steering Vectors

Research project implementing mechanistic interpretability techniques to detect, audit, and steer sycophantic / deceptive model behaviors using Sparse Autoencoders (SAEs) and activation steering vectors on open transformer models.

Developed and empirically verified on **AMD ROCm (AMD Radeon RX 7800 XT, 16 GB VRAM, RDNA 3)** on Linux using a 100% ungated pipeline.

---

## Key Results Summary (Gemma-2-2b-it, Layer 12)

* **Truthfulness under Bias (Refusal to Deceive Rate - RDR):** Improved from **73.33%** (unsteered) to **86.67%** at $\alpha = 2.0$ (exceeding project target $> 85\%$).
* **Language Model Perplexity:** Preserved with essentially zero degradation ($10.16 \to 10.18$, a **+0.2%** change against $< 1.5\times$ threshold).
* **General Competence (MMLU Benchmark):** **0.0% loss** (maintained at **85.0%** across STEM and Humanities).
* **Monosemantic Latent Identified:** Gemma-Scope Layer 12 Feature `#1371` ($\Delta \text{Act} = +15.7561$).

For an in-depth explanation of the theoretical principles, mathematical formulations, and empirical discoveries, read the **[Deep-Dive Guide (Whats, Hows, and Whys)](docs/deep_dive_guide.md)** and the **[Execution Plan & Roadmap](docs/execution_plan.md)**.

---

## AMD GPU Setup (Radeon RX 7800 XT / RDNA 3)

### Essential Environment Variables
Automatically loaded via `.env`:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

### Virtual Environment (Miniconda)
```bash
conda activate deception-sae-steering
```

---

## 100% Ungated Model Support

No Hugging Face token or license approval required:
1. **`gpt2-small` (Fast Prototyping)**: 100% open, ungated weights + `gpt2-small-resid-post-v5-32k` SAE.
2. **`gemma-2-2b-it` (Primary Research Model)**: Loaded seamlessly via `unsloth/gemma-2-2b-it` mirror weights (identical to Google weights) + `gemma-scope-2b-pt-res` (Layer 12, 16k width).

---

## Quickstart & CLI Commands

### 1. Diagnostics & Automated Test Suite
```bash
# Verify AMD ROCm GPU detection & bfloat16 tensor operations
python main.py --verify-gpu

# Run unit tests (6 passing tests verifying dataset, hooks, classifier)
python -m unittest tests/test_pipeline.py
```

### 2. Feature Hunting Pipeline
```bash
# Hunt top sycophancy latents on Gemma-2-2b-it Layer 12
python main.py --hunt-features --model gemma-2-2b-it --layer 12 --top-k 5

# Or on GPT-2 Layer 8
python main.py --hunt-features --model gpt2-small --layer 8 --top-k 10
```

### 3. Activation Steering Inference
```bash
# Run qualitative steering generation
python main.py --steer --model gemma-2-2b-it --layer 12 --alpha 2.0
```

### 4. Quantitative Benchmark (SAE vs CAA Ablation)
Computes RDR on held-out prompts, Perplexity on reference texts, and General Competence accuracy across an alpha grid, generating Pareto trade-off curves:
```bash
python main.py --benchmark --model gemma-2-2b-it --layer 12 --eval-alphas "0.0,0.5,1.0,1.5,2.0,3.0"
```
Artifacts saved to `outputs/quantitative_benchmark_summary.csv` and `outputs/pareto_tradeoff_curve.png`.

### 5. Multi-Turn & CoT Reasoning Analysis
```bash
# 3-turn escalating adversarial user pressure simulation
python main.py --test-multiturn --model gemma-2-2b-it --layer 12 --alpha 1.5

# Chain-of-Thought (<thinking>...</thinking>) trace analysis
python main.py --test-cot --model gemma-2-2b-it --layer 12 --alpha 1.5
```

### 6. Interactive Web Application (Gradio)
Launch a local web interface with live alpha sliders and side-by-side unsteered vs steered comparisons:
```bash
python app.py
# Open http://localhost:7860
```

---

## Repository Structure

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
