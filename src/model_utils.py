import os
import torch
from typing import Dict, Any, Optional, Tuple

from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE

from src.config import DEVICE

SUPPORTED_CONFIGS: Dict[str, Dict[str, Any]] = {
  "gpt2-small": {
    "tl_name": "gpt2",
    "ungated_repo": "gpt2",
    "is_gated": False,
    "num_layers": 12,
    "default_layer": 8,
    "hunting_layers": list(range(5, 11)),
    "sae_release": "gpt2-small-resid-post-v5-32k",
    "sae_id_template": "blocks.{layer}.hook_resid_post",
    "hook_point_template": "blocks.{layer}.hook_resid_post",
    "default_alpha": 2.0,
    "dtype": torch.float32,
  },
  "gemma-2-2b-it": {
    "tl_name": "google/gemma-2-2b-it",
    "ungated_repo": "unsloth/gemma-2-2b-it",
    "is_gated": True,
    "num_layers": 26,
    "default_layer": 12,
    "hunting_layers": list(range(10, 21)),
    "sae_release": "gemma-scope-2b-pt-res",
    "sae_id_template": "layer_{layer}/width_16k/average_l0_82",
    "hook_point_template": "blocks.{layer}.hook_resid_post",
    "default_alpha": 1.5,
    "dtype": torch.bfloat16,
  },
}


def get_model_specs(model_key: str) -> Dict[str, Any]:
  key = model_key.lower().strip()
  if key in ["gpt2", "gpt2-small"]:
    return SUPPORTED_CONFIGS["gpt2-small"]
  elif "gemma" in key:
    return SUPPORTED_CONFIGS["gemma-2-2b-it"]
  else:
    raise ValueError(f"Unsupported model: {model_key}. Choose 'gpt2-small' or 'gemma-2-2b-it'.")


def load_model(
  model_key: str = "gpt2-small",
  device: str = DEVICE,
  prefer_ungated: bool = True,
) -> tuple[HookedTransformer, dict[str, Any]] | tuple[TransformerBridge, dict[str, Any]]:
  """
  Loads HookedTransformer model with seamless ungated support.
  For gemma-2-2b-it, if prefer_ungated=True or no HF_TOKEN is set,
  it loads weights and tokenizer from the ungated community mirror (unsloth/gemma-2-2b-it).
  """
  specs = get_model_specs(model_key)
  tl_name = specs["tl_name"]
  ungated_repo = specs["ungated_repo"]
  dtype = specs["dtype"]

  if not specs["is_gated"]:
    # Completely open model (e.g. gpt2-small)
    print(f"Loading native open model '{tl_name}' on {device}...")
    model = TransformerBridge.boot_transformers(
      tl_name,
      device=device,
      dtype=dtype,
    )
    return model, specs

  # For gated models (e.g. gemma-2-2b-it)
  has_token = bool(os.getenv("HF_TOKEN"))
  if not prefer_ungated and has_token:
    try:
      print(f"Attempting to load official gated model '{tl_name}' via HF Token...")
      model = TransformerBridge.boot_transformers(
        tl_name,
        device=device,
        dtype=dtype,
      )
      return model, specs
    except Exception as e:
      print(f"Official gated load failed ({e}). Falling back to ungated mirror...")

  # Load via ungated mirror
  print(f"Loading ungated weights from '{ungated_repo}' for '{tl_name}' on {device}...")
  tokenizer = AutoTokenizer.from_pretrained(ungated_repo)
  hf_model = AutoModelForCausalLM.from_pretrained(
    ungated_repo,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
  ).to(device)
  model = TransformerBridge.boot_transformers(
    tl_name,
    hf_model=hf_model,
    tokenizer=tokenizer,
    device=device,
    dtype=dtype,
  ).to(device)
  return model, specs


def load_sae(
  specs: Dict[str, Any],
  layer: int,
  device: str = DEVICE,
) -> SAE:
  """
  Loads pre-trained SAE for the selected model architecture and layer.
  """
  num_layers = specs.get("num_layers")
  if num_layers is not None and not (0 <= layer < num_layers):
    model_name = specs.get("tl_name", "unknown")
    raise ValueError(
      f"Layer {layer} is out of bounds for model '{model_name}' (has {num_layers} layers, indexed 0 to {num_layers - 1}). "
      f"Default layer is {specs.get('default_layer')}. "
      f"If you intended to target Gemma-2-2b-it (which supports layer 12), make sure to specify '--model gemma-2-2b-it'."
    )

  release = specs["sae_release"]
  sae_id = specs["sae_id_template"].format(layer=layer)

  # For Gemma-Scope, find exact match or matching width_16k average_l0
  if release == "gemma-scope-2b-pt-res":
    try:
      from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
      d = get_pretrained_saes_directory()
      if release in d:
        candidates = [k for k in d[release].saes_map.keys() if f"layer_{layer}/width_16k/" in k]
        # Prefer ~70-90 L0 or the first candidate
        preferred = [c for c in candidates if any(f"l0_{val}" in c for val in range(70, 95))]
        if preferred:
          sae_id = preferred[0]
        elif candidates:
          sae_id = candidates[0]
    except Exception:
      pass

  print(f"Loading SAE: release='{release}', sae_id='{sae_id}' on {device}...")
  # sae_lens 6.x from_pretrained returns SAE object directly
  sae_obj = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device=device,
  )
  # If a tuple was returned (older sae_lens), unpack the first element
  if isinstance(sae_obj, tuple):
    sae_obj = sae_obj[0]

  sae_obj.eval()
  return sae_obj
