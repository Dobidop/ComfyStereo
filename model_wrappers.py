"""
ComfyUI Model Wrappers

Wrapper classes to make ComfyUI MODEL/CLIP/VAE compatible with diffusers-style interfaces.
This allows the stereo generation code to work seamlessly with both diffusers pipelines
and ComfyUI native models.
"""

import torch
from diffusers import DDIMScheduler

# Import functional_call for gradient-enabled forward passes
# Available in PyTorch 2.0+ as torch.func, fallback to functorch for older versions
try:
    from torch.func import functional_call
except ImportError:
    try:
        from functorch import functional_call
    except ImportError:
        functional_call = None  # Will fall back to simple embeddings if not available


# Supported model types for stereo generation
SUPPORTED_MODEL_TYPES = ["SD1", "SD2"]


class VAEWrapper:
    """
    Wrapper to make ComfyUI VAE compatible with diffusers-style interface.

    Diffusers VAE returns {'latent_dist': distribution, 'sample': tensor}
    ComfyUI VAE returns tensors directly.
    """

    def __init__(self, comfy_vae, device="cuda"):
        self.comfy_vae = comfy_vae
        self.first_stage_model = comfy_vae.first_stage_model
        self._device = torch.device(device)

    def parameters(self):
        """Return model parameters for dtype detection."""
        return self.first_stage_model.parameters()

    def _get_device(self):
        """Get the actual device where VAE is loaded."""
        try:
            return next(self.first_stage_model.parameters()).device
        except StopIteration:
            return self._device

    def encode(self, x):
        """
        Encode image to latent space with diffusers-compatible return format.

        Args:
            x: Image tensor in [-1, 1] range, shape [B, C, H, W]

        Returns:
            Dict with 'latent_dist' containing object with 'mean' attribute
        """
        # ComfyUI VAE expects [B, C, H, W] in [0, 1] range
        # Convert from diffusers [-1, 1] to ComfyUI [0, 1]
        x_01 = (x + 1.0) / 2.0

        # ComfyUI's encode expects [B, H, W, C] format
        x_comfy = x_01.permute(0, 2, 3, 1).contiguous()

        # Ensure on CPU for ComfyUI's encode (it handles device internally)
        x_comfy = x_comfy.cpu()

        # Use ComfyUI's encode which handles memory management and device
        latents = self.comfy_vae.encode(x_comfy)

        # Create a simple object to mimic diffusers latent_dist.mean
        class LatentDist:
            def __init__(self, mean_val):
                self.mean = mean_val

        return {'latent_dist': LatentDist(latents)}

    def decode(self, z):
        """
        Decode latent to image with diffusers-compatible return format.

        Args:
            z: Latent tensor, shape [B, C, H, W]

        Returns:
            Dict with 'sample' containing image tensor in [-1, 1] range
        """
        # Use ComfyUI's decode which handles device internally
        images = self.comfy_vae.decode(z)

        # ComfyUI returns [B, H, W, C] in [0, 1], convert to [B, C, H, W] in [-1, 1]
        images = images.permute(0, 3, 1, 2)
        images = images * 2.0 - 1.0

        return {'sample': images}


class TextEncoderWrapper:
    """
    Wrapper to make ComfyUI CLIP text encoder compatible with diffusers-style interface.

    Uses ComfyUI's native encode method for simplicity and correctness.
    """

    def __init__(self, comfy_clip, device="cuda"):
        self.comfy_clip = comfy_clip
        self.cond_stage_model = comfy_clip.cond_stage_model
        self._device = torch.device(device)
        self._embedding_cache = {}

    def encode_text(self, text: str, target_device=None):
        """
        Encode a text string directly to embeddings.

        This bypasses tokenization and uses ComfyUI's native encoding.
        """
        cache_key = text
        if cache_key in self._embedding_cache:
            cond, pooled = self._embedding_cache[cache_key]
            if target_device is not None:
                cond = cond.to(target_device)
                if pooled is not None:
                    pooled = pooled.to(target_device)
            return cond, pooled

        # Use ComfyUI's encode method which handles tokenization internally
        tokens = self.comfy_clip.tokenize(text)
        cond, pooled = self.comfy_clip.encode_from_tokens(tokens, return_pooled=True)

        self._embedding_cache[cache_key] = (cond, pooled)

        if target_device is not None:
            cond = cond.to(target_device)
            if pooled is not None:
                pooled = pooled.to(target_device)

        return cond, pooled

    def __call__(self, input_ids):
        """
        Encode token IDs to text embeddings.

        Note: For ComfyUI, this method is less efficient than encode_text().
        It's provided for compatibility with code that expects diffusers interface.

        Args:
            input_ids: Token IDs tensor, shape [B, seq_len]

        Returns:
            Tuple of (embeddings, pooled) where embeddings has shape [B, seq_len, hidden_dim]
        """
        # For empty prompts (which is the main use case in stereo generation),
        # use the cached empty string encoding
        batch_size = input_ids.shape[0]
        target_device = input_ids.device

        # Encode empty string and repeat for batch
        cond, pooled = self.encode_text("", target_device=target_device)
        embeddings = cond.repeat(batch_size, 1, 1)

        return (embeddings, pooled)


class TokenizerWrapper:
    """
    Wrapper to make ComfyUI tokenizer compatible with diffusers-style interface.
    """

    def __init__(self, comfy_clip):
        self.comfy_clip = comfy_clip
        # SD1.x uses 77 tokens max
        self.model_max_length = 77

    def __call__(self, text, padding="max_length", max_length=None, truncation=True, return_tensors="pt"):
        """
        Tokenize text with diffusers-compatible interface.

        Args:
            text: String or list of strings to tokenize
            padding: Padding strategy
            max_length: Maximum sequence length
            truncation: Whether to truncate
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Object with input_ids attribute
        """
        if max_length is None:
            max_length = self.model_max_length

        if isinstance(text, str):
            text = [text]

        # Use ComfyUI's tokenizer
        all_input_ids = []
        for t in text:
            tokens = self.comfy_clip.tokenize(t)

            # tokens is a dict with clip_l, clip_g keys etc.
            # For SD1.x, we want clip_l ('l')
            if 'l' in tokens:
                token_data = tokens['l']
            elif 'g' in tokens:
                token_data = tokens['g']
            else:
                # Fallback - get first key
                token_data = list(tokens.values())[0]

            # token_data is list of lists of (token_id, weight) tuples
            # Get first batch entry
            batch = token_data[0] if token_data else []
            ids = [tok[0] for tok in batch]  # Extract just token IDs

            # Pad or truncate
            if len(ids) < max_length:
                ids = ids + [49407] * (max_length - len(ids))  # 49407 is typical end token for CLIP
            elif len(ids) > max_length and truncation:
                ids = ids[:max_length]
            all_input_ids.append(ids)

        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)

        class TokenizerOutput:
            def __init__(self, ids):
                self.input_ids = ids

        return TokenizerOutput(input_ids_tensor)


class UNetWrapper:
    """
    Wrapper to make ComfyUI UNet compatible with diffusers-style interface.

    Supports a gradient-enabled mode for null-text optimization by using
    functional_call with cloned parameters.
    """

    def __init__(self, comfy_model, device="cuda"):
        self.comfy_model = comfy_model
        self.diffusion_model = comfy_model.model.diffusion_model
        self._device = torch.device(device)
        # Get in_channels for compatibility
        self.in_channels = self.diffusion_model.in_channels if hasattr(self.diffusion_model, 'in_channels') else 4
        # Gradient mode flag - when True, use functional_call with cloned params
        self._gradient_mode = False
        # Cached cloned parameters for gradient mode
        self._cloned_params = None
        self._cloned_buffers = None

    def _get_model_device(self):
        """Get the device where the model parameters are located."""
        try:
            # Try to get device from model parameters
            return next(self.diffusion_model.parameters()).device
        except StopIteration:
            return self._device

    def enable_gradient_mode(self):
        """
        Enable gradient mode by cloning model parameters.
        This allows gradients to flow through the model for null-text optimization.
        """
        if not self._gradient_mode:
            # Clone all parameters to escape inference mode
            self._cloned_params = {
                name: param.clone().detach()
                for name, param in self.diffusion_model.named_parameters()
            }
            # Also clone buffers (like running_mean, running_var in BatchNorm)
            self._cloned_buffers = {
                name: buf.clone().detach()
                for name, buf in self.diffusion_model.named_buffers()
            }
            self._gradient_mode = True

    def disable_gradient_mode(self):
        """Disable gradient mode and free cloned parameters."""
        self._gradient_mode = False
        self._cloned_params = None
        self._cloned_buffers = None

    def __call__(self, latents, timestep, encoder_hidden_states=None):
        """
        Run UNet forward pass with diffusers-compatible interface.

        Args:
            latents: Noisy latent tensor
            timestep: Current timestep
            encoder_hidden_states: Text embeddings (context)

        Returns:
            Dict with 'sample' key containing noise prediction
        """
        # Get the actual device where the model is loaded
        model_device = self._get_model_device()
        model_dtype = next(self.diffusion_model.parameters()).dtype

        # Only clone tensors if in gradient mode, otherwise just move to device/dtype
        # Cloning in inference mode causes "Inference tensors do not track version counter" error
        if self._gradient_mode:
            latents = latents.clone().to(device=model_device, dtype=model_dtype)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.clone().to(device=model_device, dtype=model_dtype)
        else:
            latents = latents.to(device=model_device, dtype=model_dtype)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(device=model_device, dtype=model_dtype)

        # ComfyUI's diffusion model uses different argument names
        # and expects timestep as a tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=model_device, dtype=model_dtype)
        else:
            if self._gradient_mode:
                timestep = timestep.clone().to(device=model_device, dtype=model_dtype)
            else:
                timestep = timestep.to(device=model_device, dtype=model_dtype)

        # Expand timestep to batch size if needed
        if timestep.dim() == 0 or (timestep.dim() == 1 and timestep.shape[0] == 1):
            timestep = timestep.expand(latents.shape[0])

        if self._gradient_mode and self._cloned_params is not None and functional_call is not None:
            # Use functional_call with cloned parameters for gradient computation
            noise_pred = functional_call(
                self.diffusion_model,
                (self._cloned_params, self._cloned_buffers),
                args=(latents, timestep),
                kwargs={'context': encoder_hidden_states}
            )
        else:
            # Standard forward pass
            noise_pred = self.diffusion_model(
                latents,
                timestep,
                context=encoder_hidden_states
            )

        return {'sample': noise_pred}


class ComfyUIModelWrapper:
    """
    Wrapper that makes ComfyUI MODEL/CLIP/VAE work with diffusers-style code.

    This provides a unified interface so the stereo generation code can work
    with either diffusers pipelines or ComfyUI model inputs.
    """

    def __init__(self, model, clip, vae, device="cuda"):
        """
        Initialize the wrapper with ComfyUI model components.

        Args:
            model: ComfyUI MODEL (ModelPatcher wrapping diffusion model)
            clip: ComfyUI CLIP (text encoder)
            vae: ComfyUI VAE (encoder/decoder)
            device: Device to run on
        """
        import comfy.model_management

        self.comfy_model = model
        self.comfy_clip = clip
        self.comfy_vae = vae
        self._device = torch.device(device)

        # Load models to GPU using ComfyUI's model management
        comfy.model_management.load_models_gpu([model])

        # Extract the underlying UNet
        self.unet = UNetWrapper(model, device)

        # Create wrapped components with diffusers-compatible interfaces
        self.vae = VAEWrapper(vae, device)
        self.text_encoder = TextEncoderWrapper(clip, device)
        self.tokenizer = TokenizerWrapper(clip)

        # Create a DDIM scheduler compatible with our code
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )

        # Store model config for compatibility checks
        self._model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect the model architecture type."""
        try:
            # Check latent channels - SD1/2 use 4, SDXL uses 4 but different size
            model_config = self.comfy_model.model.model_config
            class_name = model_config.__class__.__name__

            if "SD1" in class_name or "SD10" in class_name:
                return "SD1"
            elif "SD2" in class_name or "SD20" in class_name:
                return "SD2"
            elif "SDXL" in class_name:
                return "SDXL"
            elif "Flux" in class_name:
                return "FLUX"
            else:
                # Fallback: check unet input channels
                unet = self.comfy_model.model.diffusion_model
                in_channels = unet.in_channels if hasattr(unet, 'in_channels') else 4
                if in_channels == 4:
                    return "SD1"  # Assume SD1.x compatible
                return "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    @property
    def device(self):
        """Return the actual device where the model is loaded."""
        try:
            return next(self.comfy_model.model.diffusion_model.parameters()).device
        except StopIteration:
            return self._device

    @property
    def model_type(self) -> str:
        return self._model_type

    def is_supported(self) -> bool:
        """Check if this model type is supported for stereo generation."""
        return self._model_type in SUPPORTED_MODEL_TYPES

    def get_unsupported_message(self) -> str:
        """Get error message for unsupported models."""
        return (
            f"Model type '{self._model_type}' is not yet supported for stereo generation. "
            f"Currently supported: {', '.join(SUPPORTED_MODEL_TYPES)}. "
            f"SDXL and FLUX support may be added in future updates."
        )
