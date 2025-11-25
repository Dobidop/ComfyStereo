"""
Model Loading Utilities

Functions for loading Stable Diffusion models from HuggingFace or local paths.
"""

import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler


# Global model cache
_model_cache = {}


def clear_model_cache():
    """Clear the model cache to force reloading."""
    global _model_cache
    _model_cache = {}


def load_sd_model(model_id_or_path: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
    """
    Load or retrieve cached Stable Diffusion model.

    Args:
        model_id_or_path: Either a HuggingFace model ID (e.g., "runwayml/stable-diffusion-v1-5")
                          or a local path to a diffusers-format model directory.
        device: Device to load the model on ("cuda" or "cpu")

    Returns:
        StableDiffusionPipeline instance
    """
    global _model_cache

    if model_id_or_path in _model_cache:
        return _model_cache[model_id_or_path]

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False
    )

    # Check if it's a local path
    is_local = os.path.isdir(model_id_or_path)

    # Use float32 for null-text optimization (requires gradients)
    # float16 can cause NaN due to limited precision in gradient computations
    try:
        model = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            scheduler=scheduler,
            torch_dtype=torch.float32,  # Use float32 for gradient stability
            local_files_only=is_local,
            safety_checker=None,  # Disable safety checker for faster loading
            requires_safety_checker=False,
        ).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Attempting to download from HuggingFace...")
        model = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            scheduler=scheduler,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    try:
        model.disable_xformers_memory_efficient_attention()
    except AttributeError:
        pass

    _model_cache[model_id_or_path] = model
    return model
