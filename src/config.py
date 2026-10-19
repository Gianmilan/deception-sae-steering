import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# AMD ROCm / Hardware Configuration
# Set defaults if not already present in environment
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
if "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
if "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ:
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

DEVICE = os.getenv("DEVICE", "cuda:0")

# Model & SAE Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2b-pt-res")
DEFAULT_SAE_ID = os.getenv("DEFAULT_SAE_ID", "layer_12/width_16k/canonical")

# Default exploration range for Gemma 2 2B (26 layers total)
FEATURE_HUNTING_LAYERS = list(range(10, 21))

# Default steering scaling sweep
ALPHA_SWEEP = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

# File paths
DATA_DIR = ROOT_DIR / "data"
SYCOPHANCY_DATASET_PATH = DATA_DIR / "sycophancy_pairs.json"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
