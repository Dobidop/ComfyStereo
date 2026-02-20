"""
Model Loading Utilities

Functions for loading Stable Diffusion models from HuggingFace or local paths.
"""

import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DDIMScheduler, EulerDiscreteScheduler


# Global model cache
_model_cache = {}


def clear_model_cache():
    """Clear the model cache to force reloading."""
    global _model_cache
    _model_cache = {}


def load_sd_model(model_id_or_path: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda",
                  scheduler_type: str = "ddim"):
    """
    Load or retrieve cached Stable Diffusion model.

    Args:
        model_id_or_path: Either a HuggingFace model ID (e.g., "runwayml/stable-diffusion-v1-5")
                          or a local path to a diffusers-format model directory.
        device: Device to load the model on ("cuda" or "cpu")
        scheduler_type: "ddim" for standard DDIM scheduler, "euler" for Euler scheduler
                        (used with turbo/LCM models)

    Returns:
        StableDiffusionPipeline instance
    """
    global _model_cache

    # Include scheduler_type in cache key so different schedulers are cached separately
    cache_key = f"{model_id_or_path}:{scheduler_type}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
        )
    else:
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )

    # Check if it's a local path
    is_local = os.path.isdir(model_id_or_path)

    # Use float32 for standard pipeline (null-text optimization requires gradients)
    # Use float16 for fast/euler pipeline (turbo models - no gradients, faster inference)
    model_dtype = torch.float16 if (scheduler_type == "euler" and device != "cpu") else torch.float32

    try:
        model = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            scheduler=scheduler,
            torch_dtype=model_dtype,
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
            torch_dtype=model_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    try:
        model.disable_xformers_memory_efficient_attention()
    except AttributeError:
        pass

    _model_cache[cache_key] = model
    return model


def load_inpainting_model(model_id_or_path: str = "runwayml/stable-diffusion-inpainting",
                           device: str = "cuda"):
    """
    Load or retrieve cached Stable Diffusion Inpainting model.

    Args:
        model_id_or_path: HuggingFace model ID or local path to an inpainting model.
        device: Device to load the model on ("cuda" or "cpu")

    Returns:
        StableDiffusionInpaintPipeline instance
    """
    global _model_cache

    cache_key = f"inpaint:{model_id_or_path}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    is_local = os.path.isdir(model_id_or_path)
    model_dtype = torch.float16 if device != "cpu" else torch.float32

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=model_dtype,
            local_files_only=is_local,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
    except Exception as e:
        print(f"Failed to load inpainting model: {e}")
        print("Attempting to download from HuggingFace...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=model_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    try:
        pipe.disable_xformers_memory_efficient_attention()
    except AttributeError:
        pass

    _model_cache[cache_key] = pipe
    return pipe
