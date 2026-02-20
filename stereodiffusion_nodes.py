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
from .model_loader import load_sd_model, load_inpainting_model
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
                "pipeline_mode": (["Standard (DDIM)", "Fast (Warp + Inpaint)"], {
                    "default": "Standard (DDIM)",
                    "tooltip": "Standard: DDIM inversion (high quality, slow). Fast: Warp image with depth, then AI-inpaint only the gap regions (fast, works with turbo/LCM models)"
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
                "inpaint_model_id": ("STRING", {
                    "default": "runwayml/stable-diffusion-inpainting",
                    "tooltip": "HuggingFace inpainting model ID (Fast mode). Default: 'runwayml/stable-diffusion-inpainting'"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional text prompt to guide inpainting (Fast mode). Describe the image content for better gap filling."
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
        model_id: str = "",
        inpaint_model_id: str = "runwayml/stable-diffusion-inpainting",
        prompt: str = ""
    ):
        # Exit inference mode for the entire function - needed for null-text optimization
        with torch.inference_mode(False):
            if pipeline_mode == "Fast (Warp + Inpaint)":
                return self._generate_stereo_fast(
                    image, depth_map, scale_factor, direction, deblur,
                    denoise_strength, num_steps, guidance_scale,
                    model, clip, vae, inpaint_model_id, prompt
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
        inpaint_model_id: str,
        prompt: str
    ):
        """Warp + Inpaint pipeline for fast stereo generation.

        1. Warps the image using depth to create the right eye view (GPU grid_sample)
        2. Identifies disoccluded gap regions from the warping
        3. Uses a diffusion model to inpaint ONLY the gap regions (repaint technique)
        4. Left eye = original image (untouched)

        This preserves the original image perfectly in non-gap areas and only
        uses AI to fill the small disoccluded strips.
        """
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load inpainting model
        if not inpaint_model_id:
            inpaint_model_id = "runwayml/stable-diffusion-inpainting"
        print(f"Warp+Inpaint pipeline - Using inpainting model: {inpaint_model_id}")
        inpaint_pipe = load_inpainting_model(inpaint_model_id, str(device))

        # 2. Prepare inputs
        img_np = tensor_to_numpy(image)
        depth_np = tensor_to_numpy(depth_map)

        if len(depth_np.shape) == 3 and depth_np.shape[2] == 3:
            depth_np = np.dot(depth_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        original_size = (img_np.shape[1], img_np.shape[0])  # (W, H)
        img_resized = np.array(Image.fromarray(img_np).resize((512, 512)))
        depth_resized = np.array(Image.fromarray(depth_np).resize((512, 512)))

        # 3. Compute warp grid and warp with NEAREST interpolation
        # We do our own warp instead of using apply_stereo_divergence_gpu_with_fill
        # because we need: nearest mode (no cross-depth blending), access to the
        # warp field for artifact detection, and border fill for gap content.
        img_t = torch.from_numpy(img_resized).float().permute(2, 0, 1).to(device) / 255.0  # [C,H,W] 0-1
        depth_t = torch.from_numpy(depth_resized).float().to(device)  # [H,W] 0-255

        H, W = 512, 512
        divergence_px = (scale_factor / 100.0) * W

        # Normalize depth to 0-1
        d = depth_t.clone()
        if d.max() > 1.0:
            d = d / 255.0
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 1e-6:
            d = (d - d_min) / (d_max - d_min)
        else:
            d = torch.zeros_like(d)
        d = d - 0.5  # convergence_point = 0.5

        # Compute per-pixel warp offset (in pixels)
        pixel_offset = d * (-divergence_px)  # negative divergence for right eye
        offset_normalized = pixel_offset / (W / 2)  # convert to grid_sample coords [-1, 1]

        # Build warp grid
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_x_warped = grid_x - offset_normalized
        grid = torch.stack([grid_x_warped, grid_y], dim=-1).unsqueeze(0)

        # Warp image with BILINEAR for smooth results
        warped_right = F.grid_sample(
            img_t.unsqueeze(0), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0)  # [C, H, W]

        # Valid mask: source coordinate is within bounds
        valid_mask = (grid_x_warped >= -1) & (grid_x_warped <= 1)  # [H, W]

        # 4. Detect disoccluded regions by comparing depth at output vs source positions
        # Warp the depth map with nearest to see what depth each output pixel actually got
        d_for_warp = d + 0.5  # back to 0-1 range for comparison
        warped_depth = F.grid_sample(
            d_for_warp.unsqueeze(0).unsqueeze(0), grid,
            mode='nearest', padding_mode='border', align_corners=True
        ).squeeze()  # [H, W]

        # Output pixel's own depth (what it SHOULD be)
        output_depth = d_for_warp  # [H, W]

        # Disocclusion: output pixel expects background (low depth) but the source
        # pixel it sampled from is foreground (high depth). This means the actual
        # background content was hidden behind the foreground in the source image.
        # The threshold scales with divergence - larger divergence = wider disocclusion zones.
        depth_diff = warped_depth - output_depth  # positive = sampled foreground for background
        disoccluded = (depth_diff > 0.05)  # [H, W]

        # Small dilation to catch edge pixels and ensure clean inpainting boundary
        if disoccluded.any():
            disoccluded_dilated = F.max_pool2d(
                disoccluded.float().unsqueeze(0).unsqueeze(0),
                kernel_size=3, stride=1, padding=1
            )
            disoccluded = (disoccluded_dilated.squeeze() > 0.5)

        # Combine: out-of-bounds gaps + disoccluded regions
        inpaint_mask_px = ~valid_mask | disoccluded

        num_mask_pixels = inpaint_mask_px.sum().item()
        total_pixels = H * W
        print(f"Warp+Inpaint - {num_mask_pixels} masked pixels ({100*num_mask_pixels/total_pixels:.1f}% of image) "
              f"(gaps + stretch artifacts)")

        # If no mask pixels, just return warped image directly
        if num_mask_pixels == 0:
            left_img = img_resized
            right_img = (warped_right.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            left_img = np.array(Image.fromarray(left_img).resize(original_size))
            right_img = np.array(Image.fromarray(right_img).resize(original_size))
            stereo_pair = np.hstack([left_img, right_img])
            return (numpy_to_tensor(stereo_pair), numpy_to_tensor(left_img), numpy_to_tensor(right_img))

        # 5. Dilate combined mask slightly for smoother blending at boundaries
        dilated = F.max_pool2d(
            inpaint_mask_px.float().unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        )
        inpaint_mask_px = (dilated.squeeze() > 0.5)

        # 6. Fill masked regions by interpolating between border pixels
        # Like "Fill - Naive interpolating": find the valid pixel on each side of
        # each gap and linearly interpolate between them.
        filled_right = warped_right.clone()  # [C, H, W]
        mask_2d = inpaint_mask_px  # [H, W]

        # Scan left-to-right: track last valid pixel color and distance
        has_left = torch.zeros(H, dtype=torch.bool, device=device)
        left_val = torch.zeros(3, H, device=device)
        left_dist = torch.zeros(H, W, device=device)
        left_colors = torch.zeros(3, H, W, device=device)

        for col in range(W):
            valid_here = ~mask_2d[:, col]  # [H]
            # Update last valid from non-masked pixels
            for c in range(3):
                left_val[c] = torch.where(valid_here, warped_right[c, :, col], left_val[c])
            has_left = has_left | valid_here
            left_colors[:, :, col] = left_val
            if col == 0:
                left_dist[:, col] = torch.where(valid_here, torch.zeros(H, device=device),
                                                torch.ones(H, device=device))
            else:
                left_dist[:, col] = torch.where(valid_here, torch.zeros(H, device=device),
                                                left_dist[:, col - 1] + 1)

        # Scan right-to-left
        has_right = torch.zeros(H, dtype=torch.bool, device=device)
        right_val = torch.zeros(3, H, device=device)
        right_dist = torch.zeros(H, W, device=device)
        right_colors = torch.zeros(3, H, W, device=device)

        for col in range(W - 1, -1, -1):
            valid_here = ~mask_2d[:, col]
            for c in range(3):
                right_val[c] = torch.where(valid_here, warped_right[c, :, col], right_val[c])
            has_right = has_right | valid_here
            right_colors[:, :, col] = right_val
            if col == W - 1:
                right_dist[:, col] = torch.where(valid_here, torch.zeros(H, device=device),
                                                 torch.ones(H, device=device))
            else:
                right_dist[:, col] = torch.where(valid_here, torch.zeros(H, device=device),
                                                 right_dist[:, col + 1] + 1)

        # Compute interpolation weight
        total_dist = left_dist + right_dist
        total_dist = torch.clamp(total_dist, min=1.0)
        t = left_dist / total_dist  # [H, W], 0=left border, 1=right border

        # Handle edge cases: if only one border exists, use it entirely
        # has_left/has_right are per-row; expand to [H, W]
        only_right = (~has_left).unsqueeze(1).expand_as(t)  # no left border found in this row
        only_left = (~has_right).unsqueeze(1).expand_as(t)   # no right border found in this row
        t = torch.where(only_right, torch.ones_like(t), t)
        t = torch.where(only_left, torch.zeros_like(t), t)

        # Apply interpolation only to masked pixels
        for c in range(3):
            interpolated = left_colors[c] * (1 - t) + right_colors[c] * t
            filled_right[c] = torch.where(mask_2d, interpolated, warped_right[c])

        # 7. Prepare PIL images for inpainting pipeline
        # The inpainting model expects PIL Image inputs
        warped_pil = Image.fromarray(
            (filled_right.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        # Mask: white = inpaint, black = keep (PIL convention for SD inpainting)
        mask_np = (inpaint_mask_px.cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode='L')

        # 8. Run inpainting pipeline
        inpaint_prompt = prompt if prompt else "high quality, detailed"
        print(f"Warp+Inpaint - Running inpainting with {num_steps} steps, "
              f"strength={denoise_strength}, guidance={guidance_scale}")
        print(f"Warp+Inpaint - Prompt: '{inpaint_prompt}'")

        result = inpaint_pipe(
            prompt=inpaint_prompt,
            image=warped_pil,
            mask_image=mask_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=denoise_strength,
        )
        inpainted_pil = result.images[0]
        inpainted_np = np.array(inpainted_pil)

        # 9. Pixel-space final blend: use original warped pixels for non-masked areas
        # This avoids any VAE quality loss in regions that don't need inpainting.
        warped_np = (warped_right.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask_bool = inpaint_mask_px.cpu().numpy()  # [H, W]
        mask_3ch = np.stack([mask_bool] * 3, axis=-1)  # [H, W, 3]
        right_result = np.where(mask_3ch, inpainted_np, warped_np)

        # 10. Left eye = original, right eye = warped + inpainted
        left_img = np.array(Image.fromarray(img_resized).resize(original_size))
        right_img = np.array(Image.fromarray(right_result).resize(original_size))
        stereo_pair = np.hstack([left_img, right_img])

        return (numpy_to_tensor(stereo_pair), numpy_to_tensor(left_img), numpy_to_tensor(right_img))

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
