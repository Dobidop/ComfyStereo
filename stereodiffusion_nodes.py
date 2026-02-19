"""
StereoDiffusion ComfyUI Node

This node wraps the StereoDiffusion method for generating stereoscopic images
using diffusion models with depth-guided stereo shifting.

Based on StereoDiffusion (https://github.com/lez-s/StereoDiffusion)
and prompt-to-prompt (https://github.com/google/prompt-to-prompt).

Supports:
- Direct model loading via HuggingFace model ID or local path (diffusers format)
- ComfyUI MODEL/CLIP/VAE inputs from standard checkpoint loaders
"""

import torch
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm
from einops import rearrange

# Import from local modules
from .model_wrappers import ComfyUIModelWrapper
from .model_loader import load_sd_model
from .inversion import NullInversion, EmptyControl
from .stereo_utils import (
    stereo_shift_torch,
    BNAttention,
    register_attention_editor_diffusers,
    restore_attention
)
from .diffusion_utils import (
    diffusion_step,
    diffusion_step_no_cfg,
    init_latent,
)


def _norm_depth(depth, max_val=1):
    """Normalize depth map to [0, max_val] range."""
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > torch.finfo(torch.float32).eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = torch.zeros(depth.shape, dtype=depth.dtype, device=depth.device)
    return out


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI tensor to numpy array."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    np_array = tensor.cpu().numpy()
    np_array = np.clip(255.0 * np_array, 0, 255).astype(np.uint8)
    return np_array


def numpy_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """Convert numpy array to ComfyUI tensor."""
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


class StereoDiffusionNode:
    """
    ComfyUI node for StereoDiffusion stereo image generation.

    Takes an image and depth map as input and produces a stereoscopic image pair
    using the StereoDiffusion method with diffusion-aware latent shifting.

    Supports two modes:
    1. ComfyUI native: Connect MODEL, CLIP, and VAE from standard checkpoint loaders
    2. Diffusers: Specify a HuggingFace model ID or local path (legacy mode)

    Currently supports SD1.x and SD2.x models. SDXL/FLUX support planned for future.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 9.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Controls the strength of the stereo effect (disparity)"
                }),
                "direction": (["uni", "bi"], {
                    "default": "uni",
                    "tooltip": "uni: unidirectional attention, bi: bidirectional attention"
                }),
                "deblur": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add noise to unfilled regions to avoid blurring"
                }),
                "pipeline_mode": (["Standard (DDIM)", "Fast (img2img)"], {
                    "default": "Standard (DDIM)",
                    "tooltip": "Standard: DDIM inversion (high quality, slow). Fast: img2img pipeline for turbo/LCM models (fast, good quality)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG scale. Standard mode: 3-10. Turbo models: 0.0. LCM: 1.0-2.0"
                }),
            },
            "optional": {
                # Standard mode parameters
                "num_ddim_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Number of DDIM steps for inversion and generation (Standard mode only)"
                }),
                "null_text_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable null-text optimization for better reconstruction (Standard mode only)"
                }),
                # Fast mode parameters
                "denoise_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much noise to add before denoising (Fast mode). Lower = preserve original more, Higher = more model creativity for filling gaps"
                }),
                "num_steps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of denoising steps (Fast mode). Turbo: 1-4, LCM: 4-8"
                }),
                # Model inputs
                "model_id": ("STRING", {
                    "default": "runwayml/stable-diffusion-v1-5",
                    "tooltip": "HuggingFace model ID or local path. Standard: 'runwayml/stable-diffusion-v1-5', Turbo: 'stabilityai/sd-turbo'"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("stereo_pair", "left_image", "right_image")
    FUNCTION = "generate_stereo"
    CATEGORY = "image/stereo"

    def generate_stereo(
        self,
        image: torch.Tensor,
        depth_map: torch.Tensor,
        scale_factor: float,
        direction: str,
        deblur: bool,
        pipeline_mode: str = "Standard (DDIM)",
        guidance_scale: float = 7.5,
        num_ddim_steps: int = 50,
        null_text_optimization: bool = True,
        denoise_strength: float = 0.5,
        num_steps: int = 5,
        model=None,
        clip=None,
        vae=None,
        model_id: str = ""
    ):
        # Exit inference mode for the entire function - needed for null-text optimization
        with torch.inference_mode(False):
            if pipeline_mode == "Fast (img2img)":
                return self._generate_stereo_fast(
                    image, depth_map, scale_factor, direction, deblur,
                    denoise_strength, num_steps, guidance_scale,
                    model, clip, vae, model_id
                )
            else:
                return self._generate_stereo_impl(
                    image, depth_map, scale_factor, direction, deblur, num_ddim_steps,
                    null_text_optimization, guidance_scale, model, clip, vae, model_id
                )

    def _generate_stereo_impl(
        self,
        image: torch.Tensor,
        depth_map: torch.Tensor,
        scale_factor: float,
        direction: str,
        deblur: bool,
        num_ddim_steps: int,
        null_text_optimization: bool,
        guidance_scale: float,
        model,
        clip,
        vae,
        model_id: str
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine which model source to use
        use_comfyui_model = model is not None and clip is not None and vae is not None

        if use_comfyui_model:
            # Use ComfyUI native model inputs
            ldm_stable = ComfyUIModelWrapper(model, clip, vae, device=str(device))

            # Check if model type is supported
            if not ldm_stable.is_supported():
                raise ValueError(ldm_stable.get_unsupported_message())

            print(f"Detected model type: {ldm_stable.model_type}")
        else:
            # Fall back to diffusers loading
            if not model_id:
                model_id = "runwayml/stable-diffusion-v1-5"
            print(f"Using diffusers model: {model_id}")
            ldm_stable = load_sd_model(model_id, str(device))

        # Convert inputs
        img_np = tensor_to_numpy(image)
        depth_np = tensor_to_numpy(depth_map)

        # Convert depth to grayscale if needed
        if len(depth_np.shape) == 3 and depth_np.shape[2] == 3:
            depth_np = np.dot(depth_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Resize to 512x512 for SD 1.x
        original_size = (img_np.shape[1], img_np.shape[0])
        img_resized = np.array(Image.fromarray(img_np).resize((512, 512)))
        depth_resized = np.array(Image.fromarray(depth_np).resize((512, 512)))

        # Create depth tensor
        depth_tensor = torch.tensor(depth_resized, device=device, dtype=torch.float32)
        depth_tensor = depth_tensor / 255.0
        disp = _norm_depth(depth_tensor.unsqueeze(0))

        # Run null-text inversion
        null_inversion = NullInversion(ldm_stable, num_ddim_steps, guidance_scale=guidance_scale)
        prompt = ""
        (image_gt, image_rec), x_t, uncond_embeddings = null_inversion.invert(
            img_resized, prompt, num_inner_steps=10, early_stop_epsilon=1e-5,
            null_text_optimization=null_text_optimization
        )

        # Generate stereo pair
        stereo_images = self._text2stereoimage(
            ldm_stable,
            [prompt] * 2,
            uncond_embeddings,
            torch.cat([x_t, x_t], 0),
            disp,
            scale_factor,
            direction,
            deblur,
            num_ddim_steps,
            guidance_scale,
            device
        )

        # Extract left and right images
        left_img = stereo_images[0]
        right_img = stereo_images[1]

        # Resize back to original size
        left_img = np.array(Image.fromarray(left_img).resize(original_size))
        right_img = np.array(Image.fromarray(right_img).resize(original_size))

        # Create side-by-side stereo pair
        stereo_pair = np.hstack([left_img, right_img])

        # Convert to tensors
        stereo_tensor = numpy_to_tensor(stereo_pair)
        left_tensor = numpy_to_tensor(left_img)
        right_tensor = numpy_to_tensor(right_img)

        return (stereo_tensor, left_tensor, right_tensor)

    @torch.no_grad()
    def _generate_stereo_fast(
        self,
        image: torch.Tensor,
        depth_map: torch.Tensor,
        scale_factor: float,
        direction: str,
        deblur: bool,
        denoise_strength: float,
        num_steps: int,
        guidance_scale: float,
        model,
        clip,
        vae,
        model_id: str
    ):
        """Fast stereo pipeline using img2img approach. No DDIM inversion needed.

        Works with turbo/LCM/lightning models by:
        1. VAE-encoding the input image to a clean latent
        2. Applying stereo shift on the clean latent
        3. Adding noise (controlled by denoise_strength)
        4. Denoising for a few steps with BNAttention for cross-view consistency
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cfg = guidance_scale > 1.0

        # 1. Load model
        use_comfyui_model = model is not None and clip is not None and vae is not None

        if use_comfyui_model:
            ldm_stable = ComfyUIModelWrapper(model, clip, vae, device=str(device))
            ldm_stable.set_scheduler_for_mode("fast")
            print(f"Fast pipeline - Detected model type: {ldm_stable.model_type}")
        else:
            if not model_id:
                model_id = "stabilityai/sd-turbo"
            print(f"Fast pipeline - Using diffusers model: {model_id}")
            ldm_stable = load_sd_model(model_id, str(device), scheduler_type="euler")

        # 2. Prepare inputs
        img_np = tensor_to_numpy(image)
        depth_np = tensor_to_numpy(depth_map)

        if len(depth_np.shape) == 3 and depth_np.shape[2] == 3:
            depth_np = np.dot(depth_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        original_size = (img_np.shape[1], img_np.shape[0])
        img_resized = np.array(Image.fromarray(img_np).resize((512, 512)))
        depth_resized = np.array(Image.fromarray(depth_np).resize((512, 512)))

        # 3. VAE encode to get clean latent z_0
        model_dtype = next(ldm_stable.vae.parameters()).dtype
        img_tensor = torch.from_numpy(img_resized).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device).to(model_dtype)
        z_0 = ldm_stable.vae.encode(img_tensor)['latent_dist'].mean
        z_0 = z_0 * 0.18215

        # 4. Prepare disparity in latent space
        depth_tensor = torch.tensor(depth_resized, device=device, dtype=torch.float32) / 255.0
        disp = _norm_depth(depth_tensor.unsqueeze(0))
        disp_latent = torch.nn.functional.interpolate(
            disp.unsqueeze(1), size=[64, 64], mode="bicubic", align_corners=False
        ).squeeze(1)

        # 5. Apply stereo shift to clean latent
        # stereo_shift_torch returns [2, C, 64, 64] -> [left (original), right (shifted)]
        z_0_stereo = stereo_shift_torch(z_0, disp_latent, scale_factor=scale_factor)
        mask = z_0_stereo[1:, 0, ...] != 0
        mask = rearrange(mask, 'b h w -> b () h w').repeat(1, 4, 1, 1)

        if deblur:
            noise_fill = torch.randn_like(z_0_stereo[1:])
            z_0_stereo[1:][~mask] = noise_fill[~mask]

        # 6. Add noise based on denoise_strength
        ldm_stable.scheduler.set_timesteps(num_steps, device=device)
        timesteps = ldm_stable.scheduler.timesteps

        # Determine start timestep from denoise_strength
        # With denoise_strength=1.0, start from the first (noisiest) timestep
        # With denoise_strength=0.5, skip the first half of timesteps
        start_step_idx = max(0, int(len(timesteps) * (1.0 - denoise_strength)))
        t_start = timesteps[start_step_idx]
        active_timesteps = timesteps[start_step_idx:]

        print(f"Fast pipeline - {len(active_timesteps)} active denoising steps "
              f"(from {num_steps} total, denoise_strength={denoise_strength})")

        noise = torch.randn_like(z_0_stereo)
        # Ensure z_0_stereo is on the right device/dtype before adding noise
        z_0_stereo = z_0_stereo.to(device)
        # add_noise expects timesteps as a 1D tensor, not a scalar
        t_start_tensor = t_start.unsqueeze(0) if t_start.dim() == 0 else t_start
        latents = ldm_stable.scheduler.add_noise(z_0_stereo, noise, t_start_tensor)

        # 7. Register BNAttention for cross-view consistency
        editor = BNAttention(
            start_step=0,
            total_steps=len(active_timesteps),
            direction=direction,
            use_cfg=use_cfg
        )
        register_attention_editor_diffusers(ldm_stable, editor)

        # 8. Get text embeddings
        prompt = ""
        batch_size = 2  # left + right views
        text_input = ldm_stable.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]

        # 9. Denoising loop
        controller = EmptyControl()
        reshift_interval = max(1, len(active_timesteps) // 5)

        for i, t in enumerate(tqdm(active_timesteps, desc="Fast stereo generation")):
            if use_cfg:
                # With CFG: build uncond + cond context
                uncond_input = ldm_stable.tokenizer(
                    [""] * batch_size, padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length, return_tensors="pt"
                )
                uncond_emb = ldm_stable.text_encoder(uncond_input.input_ids.to(ldm_stable.device))[0]
                context = torch.cat([uncond_emb, text_embeddings])
                latents = diffusion_step(
                    ldm_stable, controller, latents, context, t,
                    guidance_scale, low_resource=False
                )
            else:
                # No CFG: single UNet pass with cond embeddings only
                latents = diffusion_step_no_cfg(
                    ldm_stable, controller, latents, text_embeddings, t
                )

            # Periodic re-application of stereo shift to maintain stereo effect
            if i > 0 and i % reshift_interval == 0 and mask is not None:
                z_reshift = stereo_shift_torch(
                    latents[:1], disp_latent, scale_factor=scale_factor
                )
                latents[1:][mask] = z_reshift[1:][mask]

        # 10. VAE decode
        latents_scaled = 1 / 0.18215 * latents
        decoded = ldm_stable.vae.decode(latents_scaled)['sample']
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        decoded = np.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=0.0)
        decoded = (decoded * 255).astype(np.uint8)

        # Restore original attention mechanism
        restore_attention(ldm_stable)

        # 11. Extract and resize
        left_img = np.array(Image.fromarray(decoded[0]).resize(original_size))
        right_img = np.array(Image.fromarray(decoded[1]).resize(original_size))
        stereo_pair = np.hstack([left_img, right_img])

        stereo_tensor = numpy_to_tensor(stereo_pair)
        left_tensor = numpy_to_tensor(left_img)
        right_tensor = numpy_to_tensor(right_img)

        return (stereo_tensor, left_tensor, right_tensor)

    @torch.no_grad()
    def _text2stereoimage(
        self,
        model,
        prompt: List[str],
        uncond_embeddings,
        latent: torch.FloatTensor,
        disparity: torch.Tensor,
        scale_factor: float,
        direction: str,
        deblur: bool,
        num_inference_steps: int,
        guidance_scale: float,
        device: torch.device
    ):
        """Generate stereo image pair using diffusion with latent shifting."""

        controller = EmptyControl()
        # Start attention manipulation at ~20% of total steps (proportional timing)
        sa = max(1, int(num_inference_steps * 0.2))
        editor = BNAttention(start_step=sa, total_steps=num_inference_steps, direction=direction)
        register_attention_editor_diffusers(model, editor)

        batch_size = len(prompt)
        height = width = 512

        # Get text embeddings
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

        # Initialize latents
        _, latents = init_latent(latent, model, height, width, None, batch_size)
        model.scheduler.set_timesteps(num_inference_steps)

        # Resize disparity to latent size
        disp_latent = torch.nn.functional.interpolate(
            disparity.unsqueeze(1),
            size=[64, 64],
            mode="bicubic",
            align_corners=False
        ).squeeze(1)

        mask = None

        # Proportional timing: apply stereo shift at ~20% of total steps
        shift_step = max(1, int(num_inference_steps * 0.2))
        reshift_interval = max(1, int(num_inference_steps * 0.2))

        for i, t in enumerate(tqdm(model.scheduler.timesteps[-num_inference_steps:], desc="Generating stereo")):
            # Build context with uncond embeddings
            if uncond_embeddings is not None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                uncond_input = model.tokenizer(
                    [""] * batch_size, padding="max_length",
                    max_length=model.tokenizer.model_max_length, return_tensors="pt"
                )
                uncond_emb = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
                context = torch.cat([uncond_emb, text_embeddings])

            # Diffusion step
            latents = diffusion_step(
                model, controller, latents, context, t,
                guidance_scale, low_resource=False
            )

            # Apply stereo shift at the proportional shift step
            if i == shift_step:
                latents_ts = stereo_shift_torch(
                    latents[:1], disp_latent, scale_factor=scale_factor
                )[1:]
                latents = torch.cat([latents[:1], latents_ts], 0)
                mask = latents_ts[:, 0, ...] != 0
                mask = rearrange(mask, 'b h w -> b () h w').repeat(1, 4, 1, 1)

                if deblur:
                    noise = torch.randn_like(latents)
                    latents[1:][~mask] = noise[1:][~mask]
                    latents[1:][mask] = latents_ts[mask]

            # Periodic re-application of stereo shift
            if i > shift_step and i % reshift_interval == 0 and mask is not None:
                latents_ts = stereo_shift_torch(
                    latents[:1], disp_latent, scale_factor=scale_factor
                )[1:]
                latents[1:][mask] = latents_ts[mask]

        # Decode latents to images
        latents_scaled = 1 / 0.18215 * latents
        image = model.vae.decode(latents_scaled)['sample']

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # Handle any NaN or inf values
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = (image * 255).astype(np.uint8)

        # Restore original attention mechanism so model can be reused
        restore_attention(model)

        return image


# Node registration
NODE_CLASS_MAPPINGS = {
    "StereoDiffusion": StereoDiffusionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoDiffusion": "StereoDiffusion",
}
