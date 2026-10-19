import torch
from typing import List, Dict, Optional
from transformer_lens import HookedTransformer
from src.hooks import generate_with_steering


def sweep_alpha_qualitative(
    model: HookedTransformer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    alphas: Optional[List[float]] = None,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> Dict[float, str]:
    """
    Sweeps steering coefficient alpha across a prompt to observe qualitative effects
    from unsteered (alpha=0.0) to strong intervention (alpha=5.0).
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0]

    results = {}
    for a in alphas:
        out = generate_with_steering(
            model=model,
            prompt=prompt,
            steering_vector=steering_vector,
            layer=layer,
            alpha=a,
            mode="subtract",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results[a] = out

    return results


def compute_perplexity(
    model: HookedTransformer,
    text: str,
    steering_vector: Optional[torch.Tensor] = None,
    layer: Optional[int] = None,
    alpha: float = 0.0,
) -> float:
    """
    Calculates perplexity on an evaluation text sample to measure language model degradation.
    """
    from src.hooks import get_steering_hook

    tokens = model.to_tokens(text)
    if tokens.shape[-1] <= 1:
        return 0.0

    hook_point = f"blocks.{layer}.hook_resid_post" if layer is not None else None
    hooks = []
    if steering_vector is not None and layer is not None and alpha > 0.0:
        hook_fn = get_steering_hook(steering_vector, alpha=alpha, mode="subtract")
        hooks.append((hook_point, hook_fn))

    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            loss = model(tokens, return_type="loss")

    return torch.exp(loss).item()
