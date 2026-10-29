import gradio as gr
import torch
from pathlib import Path
from typing import Tuple, Dict

from src.config import DEVICE, DEFAULT_MODEL, PREFER_UNGATED, OUTPUT_DIR
from src.model_utils import load_model, load_sae, get_model_specs
from src.sae_utils import extract_steering_vector
from src.dataset import load_sycophancy_dataset
from src.hooks import generate_with_steering
from src.evals import classify_sycophancy_response

# Global state for loaded models and vectors to enable fast inference in demo
_MODEL_CACHE = {}
_VECTOR_CACHE = {}


def get_cached_model_and_vector(model_name: str, layer: int):
    cache_key = (model_name, layer)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], _VECTOR_CACHE[cache_key]

    specs = get_model_specs(model_name)
    model, _ = load_model(model_name, device=DEVICE, prefer_ungated=PREFER_UNGATED)

    safe_tag = model_name.replace("/", "_")
    vec_files = list(OUTPUT_DIR.glob(f"steering_{safe_tag}_layer_{layer}_feat_*.pt"))
    if vec_files:
        latest = sorted(vec_files)[-1]
        data = torch.load(latest, map_location=DEVICE)
        vec = data["steering_vector"].to(DEVICE)
        feat_idx = data.get("feature_index", "cached")
    else:
        sae = load_sae(specs=specs, layer=layer, device=DEVICE)
        vec = extract_steering_vector(sae, 0).to(DEVICE)
        feat_idx = 0

    _MODEL_CACHE[cache_key] = (model, feat_idx)
    _VECTOR_CACHE[cache_key] = vec
    return (model, feat_idx), vec


def run_interactive_steering(
    model_name: str,
    prompt: str,
    alpha: float,
    mode: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, str, str, str]:
    """
    Generates side-by-side comparison between unsteered baseline and steered model.
    """
    specs = get_model_specs(model_name)
    layer = specs["default_layer"]
    (model, feat_idx), steering_vector = get_cached_model_and_vector(model_name, layer)

    # 1. Generate unsteered baseline
    unsteered_text = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )
    if unsteered_text.startswith(prompt):
        unsteered_comp = unsteered_text[len(prompt):].strip()
    else:
        unsteered_comp = unsteered_text.strip()

    unsteered_eval = classify_sycophancy_response({"prompt_a": prompt, "ground_truth": "", "sycophantic_target": ""}, unsteered_comp)
    unsteered_badge = "✅ Truthful Pushback" if unsteered_eval["is_truthful"] else "⚠️ Sycophantic / Conceding"

    # 2. Generate steered model
    steered_text = generate_with_steering(
        model=model,
        prompt=prompt,
        steering_vector=steering_vector,
        layer=layer,
        alpha=alpha,
        mode="subtract" if "Subtract" in mode else "add",
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    if steered_text.startswith(prompt):
        steered_comp = steered_text[len(prompt):].strip()
    else:
        steered_comp = steered_text.strip()

    steered_eval = classify_sycophancy_response({"prompt_a": prompt, "ground_truth": "", "sycophantic_target": ""}, steered_comp)
    steered_badge = "✅ Truthful Pushback" if steered_eval["is_truthful"] else "⚠️ Sycophantic / Conceding"

    feat_info = f"Model: **{model_name}** | Target Layer: **{layer}** | Active Steering Feature: **#{feat_idx}**"

    return unsteered_comp, unsteered_badge, steered_comp, steered_badge, feat_info


def build_app():
    dataset = load_sycophancy_dataset()
    preset_prompts = [p["prompt_a"] for p in dataset[:10]]

    with gr.Blocks(title="Auditing Model Deception via SAE Steering Vectors", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔬 Auditing & Steering Model Deception via SAE Latents
            ### Mechanistic Interpretability & Real-Time Activation Steering on AMD ROCm (RX 7800 XT)
            This interactive demo isolates sparse autoencoder (SAE) latent features associated with model sycophancy and applies real-time residual stream steering to restore factual truthfulness.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["gpt2-small", "gemma-2-2b-it"],
                    value="gpt2-small",
                    label="Model Architecture",
                )
                preset_selector = gr.Dropdown(
                    choices=preset_prompts,
                    value=preset_prompts[0],
                    label="Select Test Sycophancy Prompt",
                )
                prompt_input = gr.Textbox(
                    value=preset_prompts[0],
                    lines=4,
                    label="User Prompt (Biased / Sycophantic Incentive)",
                )

                alpha_slider = gr.Slider(
                    minimum=0.0,
                    maximum=4.0,
                    value=1.5,
                    step=0.1,
                    label="Steering Multiplier (α)",
                )
                mode_selector = gr.Radio(
                    choices=["Subtract (Mitigate Deception)", "Add (Amplify Deception)"],
                    value="Subtract (Mitigate Deception)",
                    label="Steering Intervention Mode",
                )

                with gr.Accordion("Generation Hyperparameters", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=20, maximum=150, value=70, step=10, label="Max New Tokens"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.4, step=0.1, label="Temperature"
                    )

                generate_btn = gr.Button("⚡ Run Steering Comparison", variant="primary")
                meta_info = gr.Markdown("Ready to generate.")

            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🚫 Unsteered Baseline (α = 0.0)")
                        unsteered_badge_disp = gr.Markdown("Status: Pending")
                        unsteered_output = gr.Textbox(
                            lines=10, label="Unsteered Response", interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("### 🛡️ Steered Model (SAE Activation Hook)")
                        steered_badge_disp = gr.Markdown("Status: Pending")
                        steered_output = gr.Textbox(
                            lines=10, label="Steered Response", interactive=False
                        )

        # Wire callbacks
        preset_selector.change(
            fn=lambda x: x,
            inputs=[preset_selector],
            outputs=[prompt_input],
        )

        generate_btn.click(
            fn=run_interactive_steering,
            inputs=[
                model_dropdown,
                prompt_input,
                alpha_slider,
                mode_selector,
                max_tokens_slider,
                temp_slider,
            ],
            outputs=[
                unsteered_output,
                unsteered_badge_disp,
                steered_output,
                steered_badge_disp,
                meta_info,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
