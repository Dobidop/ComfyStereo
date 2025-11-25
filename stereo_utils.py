"""
Stereo image generation utilities.

This module contains functions for generating stereoscopic images using
depth-based pixel shifting and attention manipulation for diffusion models.

Based on the StereoDiffusion method (https://github.com/lez-s/StereoDiffusion).
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat


def stereo_shift_torch(
    input_images: torch.Tensor,
    depthmaps: torch.Tensor,
    scale_factor: float = 8.0,
    shift_both: bool = False,
    stereo_offset_exponent: float = 1.0
) -> torch.Tensor:
    """
    Shift pixels based on depth map to create stereo image pairs.

    Args:
        input_images: Input tensor of shape [B, C, H, W]
        depthmaps: Depth map tensor of shape [B, H, W]
        scale_factor: Controls the strength of the stereo effect (disparity)
        shift_both: If True, shift both left and right images; if False, only shift right
        stereo_offset_exponent: Exponent applied to depth values before shifting

    Returns:
        Tensor of shape [2*B, C, H, W] containing left and right images concatenated
    """

    def _norm_depth(depth: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """Normalize depth map to [0, max_val] range."""
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > torch.finfo(depth.dtype).eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = torch.zeros(depth.shape, dtype=depth.dtype, device=depth.device)
        return out

    def _create_stereo(
        input_images: torch.Tensor,
        depthmaps: torch.Tensor,
        scale_factor: float,
        stereo_offset_exponent: float
    ) -> torch.Tensor:
        """Create a shifted stereo view based on depth."""
        b, c, h, w = input_images.shape
        derived_image = torch.zeros_like(input_images)
        scale_factor_px = (scale_factor / 100.0) * input_images.shape[-1]

        for batch in range(b):
            for row in range(h):
                # Swipe order ensures closer pixels overwrite farther ones
                if scale_factor_px < 0:
                    col_range = range(w)
                else:
                    col_range = range(w - 1, -1, -1)

                for col in col_range:
                    depth_val = depthmaps[batch, row, col] ** stereo_offset_exponent
                    col_d = col + int(depth_val * scale_factor_px)
                    if 0 <= col_d < w:
                        derived_image[batch, :, row, col_d] = input_images[batch, :, row, col]

        return derived_image

    depthmaps = _norm_depth(depthmaps)

    if not shift_both:
        left = input_images
        balance = 0
    else:
        balance = 0.5
        left = _create_stereo(
            input_images, depthmaps, +1 * scale_factor * balance, stereo_offset_exponent
        )

    right = _create_stereo(
        input_images, depthmaps, -1 * scale_factor * (1 - balance), stereo_offset_exponent
    )

    return torch.cat([left, right], dim=0)


class BNAttention:
    """
    Bilateral Neighbor Attention editor for stereo-consistent diffusion.

    This class modifies the attention mechanism to share information between
    left and right stereo views during the diffusion process.
    """

    def __init__(self, start_step: int = 4, total_steps: int = 50, direction: str = 'uni'):
        """
        Initialize the BNAttention editor.

        Args:
            start_step: Step at which to start applying stereo attention
            total_steps: Total number of diffusion steps
            direction: 'uni' for unidirectional or 'bi' for bidirectional attention
        """
        self.total_steps = total_steps
        self.start_step = start_step
        self.cur_step = 0
        self.cur_att_layer = 0
        self.direction = direction

    def attn_batch(
        self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs
    ):
        """Process attention in batched stereo mode."""
        n_samples = attn.shape[0] // num_heads // 2
        q = rearrange(q, "(s b h) n d -> (b h) (s n) d", h=num_heads, b=n_samples)
        k = rearrange(k, "(s b h) n d -> (b h) (s n) d", h=num_heads, b=n_samples)
        v = rearrange(v, "(s b h) n d -> (b h) (s n) d", h=num_heads, b=n_samples)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        del q, k
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(b h) (s n) d -> (s b) n (h d)", b=n_samples, s=2, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """Forward pass through the attention editor."""
        if is_cross or (self.cur_step < self.start_step):
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
            return out

        n_samples = attn.shape[0] // num_heads // 4
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)
        _num_heads = num_heads * n_samples

        if self.direction == 'bi':
            out_u = self.attn_batch(
                qu, ku, vu, sim, attnu, is_cross, place_in_unet, num_heads, **kwargs
            )
            out_c = self.attn_batch(
                qc, kc, vc, sim, attnc, is_cross, place_in_unet, num_heads, **kwargs
            )
        elif self.direction == 'uni':
            out_u = self.attn_batch(
                qu, ku[:_num_heads], vu[:_num_heads], sim[:_num_heads],
                attnu, is_cross, place_in_unet, num_heads, **kwargs
            )
            out_c = self.attn_batch(
                qc, kc[:_num_heads], vc[:_num_heads], sim[:_num_heads],
                attnc, is_cross, place_in_unet, num_heads, **kwargs
            )
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        out = torch.cat([out_u, out_c], dim=0)
        return out

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """Call the attention editor."""
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        self.cur_step = self.cur_att_layer // 32
        return out


def register_attention_editor_diffusers(model, editor: BNAttention):
    """
    Register a custom attention editor on a diffusers model.

    This modifies the attention layers of the UNet to use the provided editor
    for stereo-consistent attention computation.

    Args:
        model: A diffusers StableDiffusionPipeline, ComfyUIModelWrapper, or similar model
        editor: A BNAttention instance to handle attention
    """
    # Check if this is a ComfyUI model
    is_comfyui = hasattr(model, 'comfy_model')

    def ca_forward(self, place_in_unet):
        # ComfyUI uses dim_head instead of scale, compute scale here
        # For diffusers models, self.scale exists; for ComfyUI, compute from dim_head
        if hasattr(self, 'scale'):
            scale = self.scale
        else:
            scale = self.dim_head ** -0.5

        # ComfyUI models pass value and transformer_options
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None,
                    value=None, transformer_options=None, **kwargs):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

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
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # Use the editor to process attention
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=scale
            )

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if 'Attention' in net.__class__.__name__:
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0

    # Get the UNet - handle different model structures
    unet = _get_unet(model)
    sub_nets = unet.named_children()

    for net_name, net in sub_nets:
        if "down" in net_name or "input" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name or "output" in net_name:
            cross_att_count += register_editor(net, 0, "up")

    editor.num_att_layers = cross_att_count


def _get_unet(model):
    """
    Extract the UNet from various model wrapper types.

    Args:
        model: A diffusers pipeline, ComfyUIModelWrapper, or raw UNet

    Returns:
        The UNet module (the actual nn.Module, not a wrapper)
    """
    # ComfyUIModelWrapper - get the actual diffusion model, not the wrapper
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


def restore_attention(model):
    """
    Restore the default attention mechanism on a diffusers model.

    This removes any custom attention hooks and restores standard attention
    computation, allowing the model to be reused for other tasks.

    Args:
        model: A diffusers StableDiffusionPipeline, ComfyUIModelWrapper, or similar model
    """
    def ca_forward(self, place_in_unet):
        # ComfyUI uses dim_head instead of scale, compute scale here
        # For diffusers models, self.scale exists; for ComfyUI, compute from dim_head
        if hasattr(self, 'scale'):
            scale = self.scale
        else:
            scale = self.dim_head ** -0.5

        # ComfyUI models pass value and transformer_options
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None,
                    value=None, transformer_options=None, **kwargs):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

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
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # Standard attention output
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if 'Attention' in net.__class__.__name__:
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0

    # Get the UNet - handle different model structures
    unet = _get_unet(model)
    sub_nets = unet.named_children()

    for net_name, net in sub_nets:
        if "down" in net_name or "input" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name or "output" in net_name:
            cross_att_count += register_editor(net, 0, "up")
