# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Diffusion model utilities for prompt-to-prompt style image manipulation.

This module contains helper functions for running diffusion steps,
initializing latents, and controlling attention in diffusion models.

Based on prompt-to-prompt (https://github.com/google/prompt-to-prompt).
"""

import torch
from typing import Optional, Tuple
from einops import rearrange


def diffusion_step(
    model,
    controller,
    latents: torch.Tensor,
    context: torch.Tensor,
    t: int,
    guidance_scale: float,
    low_resource: bool = False
) -> torch.Tensor:
    """
    Perform a single diffusion step with classifier-free guidance.

    Args:
        model: A diffusers StableDiffusionPipeline or similar model
        controller: A controller object with step_callback method
        latents: Current latent tensor
        context: Text embeddings context (concatenated uncond and cond)
        t: Current timestep
        guidance_scale: Classifier-free guidance scale
        low_resource: If True, run uncond and cond separately to save memory

    Returns:
        Updated latent tensor after the diffusion step
    """
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def init_latent(
    latent: Optional[torch.Tensor],
    model,
    height: int,
    width: int,
    generator: Optional[torch.Generator],
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize or expand latents for diffusion.

    Args:
        latent: Optional initial latent tensor, or None to generate random
        model: A diffusers model with unet attribute
        height: Image height in pixels
        width: Image width in pixels
        generator: Optional torch generator for reproducibility
        batch_size: Number of images to generate

    Returns:
        Tuple of (original latent, expanded latents for batch)
    """
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def _get_unet_for_attention(model):
    """
    Extract the UNet from various model wrapper types for attention registration.

    Args:
        model: A diffusers pipeline, ComfyUIModelWrapper, or raw model

    Returns:
        The UNet module
    """
    # ComfyUIModelWrapper stores the wrapped UNet
    if hasattr(model, 'comfy_model'):
        return model.comfy_model.model.diffusion_model

    # Try diffusers pipeline style
    if hasattr(model, 'unet'):
        return model.unet

    # Try ComfyUI raw model style
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        return model.model.diffusion_model

    # Assume it's already a UNet
    return model


def register_attention_control(model, controller):
    """
    Register attention control hooks on a diffusers model.

    This allows intercepting and modifying attention weights during inference.
    Used for prompt-to-prompt style editing and attention visualization.

    Args:
        model: A diffusers StableDiffusionPipeline, ComfyUIModelWrapper, or similar model
        controller: A controller object that processes attention, or None for dummy
    """
    # Check if this is a ComfyUI model (has comfy_model attribute)
    is_comfyui = hasattr(model, 'comfy_model')

    def ca_forward_diffusers(self, place_in_unet):
        """Forward function for diffusers models."""
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    def ca_forward_comfyui(self, place_in_unet):
        """Forward function for ComfyUI models - accepts value and transformer_options."""
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        # ComfyUI uses dim_head instead of scale, compute scale here
        scale = self.dim_head ** -0.5

        def forward(x, context=None, value=None, mask=None, transformer_options=None, **kwargs):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            # Use provided value or compute from context
            if value is not None:
                v = self.to_v(value)
            else:
                v = self.to_v(context)

            # Use einops rearrange instead of diffusers reshape methods
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            # Rearrange back to original shape
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return to_out(out)

        return forward

    # Select the appropriate forward function
    ca_forward = ca_forward_comfyui if is_comfyui else ca_forward_diffusers

    class DummyController:
        """Dummy controller that passes attention through unchanged."""

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0

    # Get the UNet - handle different model structures
    unet = _get_unet_for_attention(model)
    sub_nets = unet.named_children()

    for net in sub_nets:
        if "down" in net[0] or "input" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0] or "output" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
