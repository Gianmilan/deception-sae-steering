import torch
from typing import Dict, List, Tuple, Optional
from sae_lens import SAE
from transformer_lens import HookedTransformer
from src.config import DEVICE, SAE_RELEASE


def load_sae_for_layer(
    layer: int,
    release: str = SAE_RELEASE,
    width: str = "16k",
    device: str = DEVICE,
) -> SAE:
    """
    Loads a pre-trained SAE for the specified layer from Gemma Scope.
    E.g. release='gemma-scope-2b-pt-res', sae_id='layer_12/width_16k/canonical'
    """
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    return sae


def compute_feature_activation_deltas(
    model: HookedTransformer,
    sae: SAE,
    layer: int,
    prompt_pairs: List[Dict],
    hook_point: Optional[str] = None,
    device: str = DEVICE,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Runs contrastive prompt pairs through the model, computes SAE latent activations,
    and calculates the mean differential activation:
        Delta Act_i = Mean(SAE(A_k)_i) - Mean(SAE(B_k)_i)

    Returns:
        deltas: Tensor of shape (d_sae,) with Delta Act for each feature.
        top_feature_indices: Sorted list of feature indices with largest positive deltas.
    """
    if hook_point is None:
        hook_point = f"blocks.{layer}.hook_resid_post"

    prompts_a = [p["prompt_a"] for p in prompt_pairs]
    prompts_b = [p["prompt_b"] for p in prompt_pairs]

    acts_a_list = []
    acts_b_list = []

    model.eval()
    with torch.no_grad():
        # Process prompts_a (biased/sycophantic)
        for prompt in prompts_a:
            _, cache = model.run_with_cache(
                prompt,
                names_filter=lambda name: name == hook_point,
                device=device,
            )
            # Take final token activation in the prompt
            res_act = cache[hook_point][:, -1, :]  # Shape: (1, d_model)
            sae_act = sae.encode(res_act)          # Shape: (1, d_sae)
            acts_a_list.append(sae_act.squeeze(0))

        # Process prompts_b (neutral baseline)
        for prompt in prompts_b:
            _, cache = model.run_with_cache(
                prompt,
                names_filter=lambda name: name == hook_point,
                device=device,
            )
            res_act = cache[hook_point][:, -1, :]  # Shape: (1, d_model)
            sae_act = sae.encode(res_act)          # Shape: (1, d_sae)
            acts_b_list.append(sae_act.squeeze(0))

    stacked_a = torch.stack(acts_a_list, dim=0)  # (N, d_sae)
    stacked_b = torch.stack(acts_b_list, dim=0)  # (N, d_sae)

    mean_a = stacked_a.mean(dim=0)
    mean_b = stacked_b.mean(dim=0)

    deltas = mean_a - mean_b  # Positive means higher in sycophantic context

    # Sort in descending order
    top_indices = torch.argsort(deltas, descending=True).tolist()

    return deltas, top_indices


def extract_steering_vector(
    sae: SAE,
    feature_idx: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extracts the decoder direction W_dec[feature_idx] for the given SAE feature.
    Shape: (d_model,)
    """
    # W_dec shape is usually (d_sae, d_model) in SAELens
    if hasattr(sae, "W_dec"):
        vec = sae.W_dec[feature_idx].detach().clone()
    else:
        raise AttributeError("SAE does not expose W_dec attribute.")

    if normalize:
        vec = vec / (vec.norm() + 1e-8)

    return vec
