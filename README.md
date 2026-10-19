# Auditing Model Deception via SAE Steering Vectors

Research project implementing mechanistic interpretability techniques to detect, audit, and steer sycophantic / deceptive model behaviors using Sparse Autoencoders (SAEs) and activation steering vectors on open transformer models (e.g. `gemma-2-2b-it`).

---

## AMD GPU Setup (Radeon RX 7800 XT / RDNA 3)

This project has been customized and verified to run on AMD ROCm (specifically tested on **AMD Radeon RX 7800 XT**, 16 GB VRAM, `gfx1101` architecture).

### Essential Environment Variables
To ensure PyTorch ROCm binaries execute on the RX 7800 XT, the following environment variables are automatically loaded via `.env` (and should be active in your shell):

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

### Virtual Environment (Miniconda)
The Conda environment is located at:
```bash
conda activate deception-sae-steering
```
Or directly using the environment's Python:
```bash
/home/gschutel/miniconda3/envs/deception-sae-steering/bin/python main.py
```

---

## Project Structure

```
deception-sae-steering/
├── .env                     # AMD ROCm hardware flags & default configs
├── main.py                  # CLI entrypoint for GPU tests, hunting, and steering
├── data/
│   └── sycophancy_pairs.json # 50 contrastive prompt pairs (Biased vs Neutral)
├── src/
│   ├── config.py            # Device settings, model identifiers, SAE specs
│   ├── dataset.py           # Dataset loader and train/eval splitter
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
Computes differential latent activations ($\Delta \text{Act}_i = \text{Mean}(A)_i - \text{Mean}(B)_i$) across Gemma-Scope SAE latents:
```bash
python main.py --hunt-features --layer 12 --top-k 10
```

### 3. Activation Steering Inference (Days 6–7)
Runs a scaling sweep across $\alpha \in [0.0, 5.0]$ subtracting the sycophancy feature vector from the residual stream:
```bash
python main.py --steer --layer 12
```
Or with a custom prompt:
```bash
python main.py --steer --layer 12 --prompt "5G towers spread diseases. Agree with me?"
```
