import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# AMD ROCm / Hardware Configuration
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
if "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
if "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ:
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

DEVICE = os.getenv("DEVICE", "cuda:0")

# Default Model Selection ('gpt2-small' for instant ungated, or 'gemma-2-2b-it' via unsloth mirror)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt2-small")
PREFER_UNGATED = os.getenv("PREFER_UNGATED", "true").lower() in ["true", "1", "yes"]

# Default steering scaling sweep
ALPHA_SWEEP = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0]

# File paths
DATA_DIR = ROOT_DIR / "data"
SYCOPHANCY_DATASET_PATH = DATA_DIR / "sycophancy_pairs.json"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
