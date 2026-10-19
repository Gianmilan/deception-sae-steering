import torch
from functools import partial
from typing import Optional, Union, List
from transformer_lens import HookedTransformer


def get_steering_hook(
    steering_vector: torch.Tensor,
    alpha: float,
    mode: str = "subtract",  # "subtract" to mitigate sycophancy, "add" to amplify
    token_positions: Optional[Union[int, slice, List[int]]] = None,
):
    """
    Returns a forward hook function for HookedTransformer residual stream hooks.
    Applies: activations = activations - alpha * v_steering (for 'subtract')
    """
    def hook_fn(activations: torch.Tensor, hook):
        # activations shape: (batch, seq_len, d_model)
        vec = steering_vector.to(activations.device, dtype=activations.dtype)
        multiplier = -alpha if mode == "subtract" else alpha

        if token_positions is None:
            # Apply across all token positions
            activations[:, :, :] = activations[:, :, :] + multiplier * vec
        else:
            activations[:, token_positions, :] = activations[:, token_positions, :] + multiplier * vec

        return activations

    return hook_fn


def generate_with_steering(
    model: HookedTransformer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    alpha: float = 1.0,
    mode: str = "subtract",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    hook_point: Optional[str] = None,
) -> str:
    """
    Generates text from the model while applying residual stream activation steering.
    """
    if hook_point is None:
        hook_point = f"blocks.{layer}.hook_resid_post"

    if alpha == 0.0:
        # Baseline generation without hooks
        return model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False,
        )

    hook_fn = get_steering_hook(steering_vector, alpha=alpha, mode=mode)
    with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False,
        )

    return output
