"""
DDIM Inversion with Null-Text Optimization

Implements null-text inversion for converting images to latent space with
improved reconstruction quality through null-text optimization.

Based on: https://github.com/google/prompt-to-prompt
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as nnf
from torch.optim.adam import Adam

from .diffusion_utils import register_attention_control

# Import functional_call for gradient-enabled forward passes
try:
    from torch.func import functional_call
except ImportError:
    try:
        from functorch import functional_call
    except ImportError:
        functional_call = None


class EmptyControl:
    """Placeholder controller that passes attention through unchanged."""

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class NullInversion:
    """
    Implements DDIM inversion with null-text optimization for image-to-latent conversion.
    """

    def __init__(self, model, num_ddim_steps: int = 50, guidance_scale: float = 7.5):
        self.model = model
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        # Check if using ComfyUI model (which doesn't support gradient-based optimization)
        self._is_comfyui = hasattr(model, 'comfy_model')

    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    def _get_device(self):
        """Get the actual device for model operations."""
        device = self.model.device
        # Ensure we return a torch.device, not a string
        if isinstance(device, str):
            device = torch.device(device)
        return device

    @torch.no_grad()
    def image2latent(self, image):
        device = self._get_device()
        # Get the dtype from the model (float16 or float32)
        model_dtype = next(self.model.vae.parameters()).dtype

        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            latents = image.to(device=device, dtype=model_dtype)
        else:
            # Use float32 for encoding to avoid precision issues
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            # Convert to model dtype for VAE encoding
            image = image.to(dtype=model_dtype)
            latents = self.model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
            # Ensure latents are on the correct device
            latents = latents.to(device)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        device = self._get_device()

        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(device))[0]
        # Ensure embeddings are on the correct device
        uncond_embeddings = uncond_embeddings.to(device)

        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(device))[0]
        # Ensure embeddings are on the correct device
        text_embeddings = text_embeddings.to(device)

        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]

        for i in tqdm(range(self.num_ddim_steps), desc="Null-text optimization"):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        return uncond_embeddings_list

    def invert(self, image, prompt: str, num_inner_steps=10, early_stop_epsilon=1e-5, null_text_optimization=True):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)

        # Ensure image is the right size (512x512 for SD 1.x)
        if isinstance(image, np.ndarray):
            if image.shape[0] != 512 or image.shape[1] != 512:
                image = np.array(Image.fromarray(image).resize((512, 512)))
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)

        print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image)

        # Only perform null-text optimization if enabled
        if null_text_optimization:
            if self._is_comfyui:
                # Check if functional_call is available for gradient-enabled optimization
                if functional_call is not None:
                    print("Null-text optimization (ComfyUI gradient mode)...")
                    # Enable gradient mode on the UNet wrapper
                    self.model.unet.enable_gradient_mode()
                    try:
                        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
                    finally:
                        # Always disable gradient mode and free memory
                        self.model.unet.disable_gradient_mode()
                else:
                    # Fallback: functional_call not available
                    print("Using simple unconditional embeddings (functional_call not available)...")
                    uncond_embeddings, _ = self.context.chunk(2)
                    uncond_embeddings = [uncond_embeddings.clone().detach() for _ in range(self.num_ddim_steps)]
            else:
                print("Null-text optimization...")
                uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        else:
            # Skip null-text optimization - use simple unconditional embeddings
            print("Skipping null-text optimization (disabled by user)...")
            uncond_embeddings, _ = self.context.chunk(2)
            uncond_embeddings = [uncond_embeddings.clone().detach() for _ in range(self.num_ddim_steps)]

        return (image, image_rec), ddim_latents[-1], uncond_embeddings
