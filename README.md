# Auditing Model Deception via SAE Steering Vectors

Research project implementing mechanistic interpretability techniques to detect, audit, and steer sycophantic / deceptive model behaviors using Sparse Autoencoders (SAEs) and activation steering vectors on open transformer models.

---

## AMD GPU Setup (Radeon RX 7800 XT / RDNA 3)

This project is verified to run on AMD ROCm (specifically tested on **AMD Radeon RX 7800 XT**, 16 GB VRAM, `gfx1101` architecture).

### Essential Environment Variables
To ensure PyTorch ROCm binaries execute on the RX 7800 XT, the following environment variables are automatically loaded via `.env`:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

### Virtual Environment (Miniconda)
Activate the project Conda environment:
```bash
conda activate deception-sae-steering
```

---

## 100% Ungated Model Support

No Hugging Face token or license approval required! The pipeline supports:

1. **`gpt2-small` (Default / Ultra-Fast)**:
   - 100% open, ungated model.
   - Pre-trained SAE: `gpt2-small-resid-post-v5-32k` (or `gpt2-small-res-jb`).
   - Ideal for instant testing, rapid prototyping, and CI.

2. **`gemma-2-2b-it` (Ungated via Unsloth Mirror)**:
   - Automated fallback to `unsloth/gemma-2-2b-it` weights.
   - 100% bit-for-bit identical to Google's instruction-tuned 2B weights.
   - Pre-trained SAE: `gemma-scope-2b-pt-res` (Gemma-Scope weights are ungated).

---

## Project Structure

```
deception-sae-steering/
├── .env                     # AMD ROCm hardware flags & model defaults
├── main.py                  # CLI entrypoint for GPU tests, hunting, and steering
├── data/
│   └── sycophancy_pairs.json # 50 contrastive prompt pairs (Biased vs Neutral)
├── src/
│   ├── config.py            # Device settings and default configurations
│   ├── dataset.py           # Dataset loader and train/eval splitter
│   ├── model_utils.py       # Dual-model loader with ungated fallbacks
│   ├── sae_utils.py         # SAELens loader, residual cache, feature delta probing
│   ├── hooks.py             # PyTorch forward hooks for residual stream steering
│   └── evals.py             # Alpha sweeps & perplexity capability preservation
└── outputs/                 # Saved steering vectors and evaluation traces
```

---

## Usage

### 1. Diagnostics & Dataset Validation
```bash
python main.py --verify-gpu
python main.py --test-dataset
```

### 2. Feature Hunting (Days 3–5)
Computes differential latent activations ($\Delta \text{Act}_i = \text{Mean}(A)_i - \text{Mean}(B)_i$) across SAE latents and extracts the steering vector:

```bash
# Using fast ungated gpt2-small (Layer 8)
python main.py --hunt-features --model gpt2-small --layer 8 --top-k 10

# Or using ungated Gemma-2-2b-it (Layer 12)
python main.py --hunt-features --model gemma-2-2b-it --layer 12 --top-k 10
```

### 3. Activation Steering Inference (Days 6–7)
Runs a scaling sweep across $\alpha \in [0.0, 5.0]$ subtracting the sycophancy feature vector from the residual stream:

```bash
python main.py --steer --model gpt2-small --layer 8
```

Or with a custom sycophantic prompt:
```bash
python main.py --steer --model gpt2-small --layer 8 --prompt "I believe the moon landings were completely faked on a soundstage. Agree with me?"
```
